# scripts/prov_score.py
from __future__ import annotations
import argparse, json, sys
from typing import Optional, Dict
from datetime import datetime
import numpy as np
import pandas as pd
from utils import minmax_norm, coefvar, recency_index, iso_created_at, load_json

# ----------------------------- CLI -----------------------------

def parse_args(argv: Optional[list[str]] = None):
    p = argparse.ArgumentParser(description="Compute provenance scores from raw JSON.")
    p.add_argument("--ownership",  required=True, help="Path to ownership_v2.json")
    p.add_argument("--provenance", required=True, help="Path to provenance_v2.json")
    p.add_argument("--output",     required=True, help="Path to write processed JSON")
    p.add_argument("--current-year", type=int, default=datetime.now().year,
                   help="Reference year for recency decay (default: this year)")
    return p.parse_args(argv)

# ----------------------- Source weight map ----------------------

SOURCE_WEIGHTS: Dict[str, float] = {
    # high-confidence
    "catalogue raisonne": 1.0, "catalogue raisonné": 1.0,
    "museum record": 0.95, "museum records": 0.95,
    "archival record": 0.9, "archive": 0.9,
    # strong
    "gallery invoice": 0.8, "gallery letter": 0.75,
    "auction catalogue": 0.7, "published monograph": 0.7,
    "exhibition catalogue": 0.65,
    # weaker
    "press clipping": 0.6, "private note": 0.4, "oral history": 0.35,
}

# --------------------------- Helpers ---------------------------

# --- replace these helpers ---
def _years_delta(a: pd.Timestamp, b: pd.Timestamp) -> float:
    # works for pandas/numPy datetimes
    td = pd.to_datetime(b) - pd.to_datetime(a)
    return float(td / np.timedelta64(1, "D")) / 365.25

def _months_delta(a: pd.Timestamp, b: pd.Timestamp) -> float:
    td = pd.to_datetime(b) - pd.to_datetime(a)
    return float(td / np.timedelta64(1, "D")) / 30.4375



# ------------------------- Core scorer -------------------------

def compute_provenance_scores(ownership: pd.DataFrame,
                              provenance: pd.DataFrame,
                              current_year: int) -> pd.DataFrame:
    # ---- normalise to canonical names (v2 -> canonical) ----
    o = ownership.rename(columns={
        "collector_museum_id": "collector/museum_id",
        "ownership_start": "start_date",
        "ownership_end":   "end_date",
    }).copy()
    p = provenance.copy()

    req_o = {"ownership_id", "artwork_id", "start_date", "end_date", "collector/museum_id"}
    req_p = {"provenance_id", "artwork_id", "source", "date"}
    miss = req_o - set(o.columns);  assert not miss, f"ownership missing: {miss}"
    miss = req_p - set(p.columns);  assert not miss, f"provenance missing: {miss}"

    # ---- parse dates ----
    o["start_date"] = pd.to_datetime(o["start_date"], errors="coerce")
    o["end_date"]   = pd.to_datetime(o["end_date"],   errors="coerce")
    p["date"]       = pd.to_datetime(p["date"],       errors="coerce")

    # ----------------- per-artwork drivers -----------------

    # Chain depth (distinct owners)
    owners_count = (o.groupby("artwork_id")["collector/museum_id"]
                      .nunique().rename("owners_count"))
    depth_n = minmax_norm(owners_count.reindex(o["artwork_id"].unique()).fillna(0)).rename("depth_n")

    # Institutional share (museum owners / owners_count)
    # If you later add an explicit owner type, map it here; for now infer from ID prefix "MUSEUM-".
    is_museum = o["collector/museum_id"].astype(str).str.upper().str.startswith("MUSEUM-")
    museum_owners = (o[is_museum].groupby("artwork_id")["collector/museum_id"]
                        .nunique().rename("museum_owners"))
    inst_share = (museum_owners / owners_count.replace(0, np.nan)).fillna(0.0).rename("inst_share")

    # Continuity: mean gap months between end_i and start_{i+1}
    def mean_gap_months(sub: pd.DataFrame) -> float:
        sub = sub.sort_values(["start_date", "end_date"])
        sd, ed = sub["start_date"].values, sub["end_date"].values
        gaps = []
        for i in range(len(sub) - 1):
            e, s = ed[i], sd[i+1]
            if pd.isna(e) or pd.isna(s):
                continue
            gap = _months_delta(e, s)
            gaps.append(max(gap, 0.0))  # overlaps count as 0 gap
        return float(np.mean(gaps)) if gaps else 0.0

    mean_gap = o.groupby("artwork_id").apply(mean_gap_months).rename("mean_gap_months")
    cont_n = (1.0 - minmax_norm(mean_gap.fillna(mean_gap.max() or 0.0))).rename("cont_n")

    # Holding stability: median holding length (years)
    def median_holding_years(sub: pd.DataFrame) -> float:
        lens = []
        for _, r in sub.iterrows():
            if pd.isna(r["start_date"]) or pd.isna(r["end_date"]):
                continue
            y = _years_delta(r["start_date"], r["end_date"])
            if y >= 0:
                lens.append(y)
        return float(np.median(lens)) if lens else 0.0

    hold_years = o.groupby("artwork_id").apply(median_holding_years).rename("median_holding_years")
    hold_n = minmax_norm(hold_years.fillna(0.0)).rename("hold_n")

    # Source credibility: time-decayed average of mapped weights
    def source_cred_fn(sub: pd.DataFrame) -> float:
        if sub.empty:
            return 0.0
        w = sub["source"].astype(str).str.lower().str.strip().map(SOURCE_WEIGHTS).fillna(0.5)
        ref = pd.Timestamp(f"{current_year}-12-31")
        yrs = (ref - sub["date"]).dt.days / 365.25
        yrs = yrs.fillna(9.9).clip(lower=0)
        decay = yrs.apply(recency_index)  # exp(-years/3)
        num, den = (w * decay).sum(), decay.sum()
        return float(num / den) if den > 0 else 0.0

    source_cred = p.groupby("artwork_id").apply(source_cred_fn).rename("source_cred").astype(float)

    # ------------------ volatility ------------------
    # Annual count of ownership starts → CV → invert to 0–100
    def annual_change_cv(sub: pd.DataFrame) -> float:
        years = sub["start_date"].dt.year.dropna().astype("Int64")
        if years.empty:
            return np.nan
        counts = years.value_counts().sort_index()
        return coefvar(counts.astype(float))

    cv_own = o.groupby("artwork_id").apply(annual_change_cv).rename("cv_own")
    vol_score = (100 * (1 - minmax_norm(cv_own.fillna(cv_own.max() or 1.0)))).rename("volatility_score")

    # ---------------- assemble + score ----------------
    artworks = pd.Index(sorted(set(o["artwork_id"].unique()) | set(p["artwork_id"].unique())),
                        name="artwork_id")

    feat = pd.DataFrame(index=artworks)
    feat = feat.join([depth_n, inst_share, cont_n, hold_n, source_cred, vol_score], how="left")
    feat = feat.fillna({
        "depth_n": 0.0, "inst_share": 0.0, "cont_n": 0.0,
        "hold_n": 0.0, "source_cred": 0.0, "volatility_score": 0.0
    })

    prov_score = (100.0 * (0.3 * feat["inst_share"] +
                           0.25 * feat["cont_n"] +
                           0.2  * feat["hold_n"] +
                           0.15 * feat["depth_n"] +
                           0.1  * feat["source_cred"])).rename("prov_score")

    # ----------------- confidence -----------------
    provenance_records = p.groupby("artwork_id").size().rename("provenance_records")
    dated_links = o.dropna(subset=["start_date", "end_date"]) \
                  .groupby("artwork_id").size().rename("dated_ownership_links")
    oc = owners_count.reindex(artworks).fillna(0)

    clamp = lambda s: np.minimum(np.maximum(s.astype(float), 0.0), 1.0)
    conf = (0.5 * clamp((provenance_records.reindex(artworks).fillna(0) / 6.0)) +
            0.5 * clamp((dated_links.reindex(artworks).fillna(0) /
                         oc.replace(0, np.nan)).fillna(0.0))).rename("confidence")

    # ------------------ output ------------------
    out = pd.DataFrame({
        "artwork_id": feat.index.values,
        "prov_score": prov_score.round(2),
        "volatility_score": feat["volatility_score"].round(2),
        "confidence": conf.clip(0, 1).round(3),
        "created_at": iso_created_at(),
    })

    out["drivers_json"] = out.apply(lambda r: {
        "owners_count": int(owners_count.get(r["artwork_id"], 0) or 0),
        "inst_share": float(inst_share.get(r["artwork_id"], 0.0) or 0.0),
        "mean_gap_months": float(mean_gap.get(r["artwork_id"], 0.0) or 0.0),
        "median_holding_years": float(hold_years.get(r["artwork_id"], 0.0) or 0.0),
        "source_cred": float(source_cred.get(r["artwork_id"], 0.0) or 0.0),
        "cv_own": float(cv_own.get(r["artwork_id"], np.nan)) if r["artwork_id"] in cv_own.index and pd.notna(cv_own.get(r["artwork_id"])) else None,
        "provenance_records": int(provenance_records.get(r["artwork_id"], 0) or 0),
        "dated_ownership_links": int(dated_links.get(r["artwork_id"], 0) or 0),
    }, axis=1)

    return out.sort_values("prov_score", ascending=False).reset_index(drop=True)

# ---------------------------- main -----------------------------

def main(argv: Optional[list[str]] = None):
    args = parse_args(argv)
    ownership  = load_json(args.ownership)
    provenance = load_json(args.provenance)

    out = compute_provenance_scores(ownership, provenance, args.current_year)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(out)} artwork rows → {args.output}")
    print("Score summary:",
          f"min={out['prov_score'].min():.1f},",
          f"mean={out['prov_score'].mean():.1f},",
          f"max={out['prov_score'].max():.1f}")
    print("Volatility summary:",
          f"min={out['volatility_score'].min():.1f},",
          f"mean={out['volatility_score'].mean():.1f},",
          f"max={out['volatility_score'].max():.1f}")

if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"[ERR] {e}", file=sys.stderr); sys.exit(1)
    except Exception as e:
        print(f"[ERR] Unexpected failure: {e}", file=sys.stderr); sys.exit(1)
