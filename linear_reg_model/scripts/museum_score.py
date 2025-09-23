from __future__ import annotations
import argparse, json, sys
from typing import Optional
from datetime import datetime
import numpy as np
import pandas as pd
from utils import minmax_norm, coefvar, recency_index, iso_created_at, load_json, to_quarter

def parse_args(argv: Optional[list[str]] = None):
    p = argparse.ArgumentParser(description="Compute museum prestige scores from raw JSON.")
    p.add_argument("--museums",  required=True, help="Path to museums JSON")
    p.add_argument("--loans",    required=True, help="Path to artwork_loan JSON")
    p.add_argument("--output",   required=True, help="Path to write processed JSON")
    p.add_argument("--current-year", type=int, default=datetime.now().year,
                   help="Current year used for 3y/12q windows (default: this year)")
    return p.parse_args(argv)

def compute_museum_scores(museums: pd.DataFrame,
                          loans: pd.DataFrame,
                          current_year: int) -> pd.DataFrame:
    # --- required columns ---
    req_m = {"museum_id", "name", "location", "age", "annual_visitors"}
    req_l = {"loan_id", "artwork_id", "loan_date", "loan_location", "loan_length", "loaner_id"}
    miss = req_m - set(museums.columns); assert not miss, f"museums missing: {miss}"
    miss = req_l - set(loans.columns);   assert not miss, f"loans missing: {miss}"

    # Interpret loans.loaner_id as the museum receiving/issuing the loan we care about
    df = loans.rename(columns={"loaner_id": "museum_id"}).merge(
        museums[["museum_id"]], on="museum_id", how="inner"
    )

    # Clean/parse types
    df["loan_date"] = pd.to_datetime(df["loan_date"], errors="coerce")
    df = df.dropna(subset=["loan_date"])
    df["loan_length"] = pd.to_numeric(df["loan_length"], errors="coerce")  # months
    museums = museums.copy()
    museums["age"] = pd.to_numeric(museums["age"], errors="coerce")
    museums["annual_visitors"] = pd.to_numeric(museums["annual_visitors"], errors="coerce")

    # ---- feature components (museum level) ----
    # Visitor reach: vis_n = minmax(log1p(vis))
    vis_log = museums.set_index("museum_id")["annual_visitors"].pipe(np.log1p)
    vis_n = minmax_norm(vis_log).rename("vis_n")

    # Institution age: age_n = minmax(age)
    age_n = minmax_norm(museums.set_index("museum_id")["age"]).rename("age_n")

    # Exhibition intensity: loans in last 3y → minmax
    start_3y = pd.Timestamp(f"{current_year-3}-01-01")
    df3 = df[df["loan_date"] >= start_3y]
    loans_3y = df3.groupby("museum_id").size().rename("loans_3y")
    loans_n = minmax_norm(loans_3y.reindex(museums["museum_id"]).fillna(0)).rename("loans_n")

    # Duration weight: average loan length (months) → minmax
    avg_len = df.groupby("museum_id")["loan_length"].mean().rename("avg_loan_length")
    len_n = minmax_norm(avg_len.reindex(museums["museum_id"]).fillna(0)).rename("len_n")

    # ---- volatility ----
    # Quarterly loan counts over last 12 quarters → CV → 0–100 with inversion
    df_q = df.copy()
    df_q["quarter_end"] = to_quarter(df_q["loan_date"])
    q_counts = (df_q.groupby(["museum_id", "quarter_end"])
                    .size()
                    .reset_index(name="count"))

    def cv_last_12(sub: pd.DataFrame) -> float:
        sub = sub.sort_values("quarter_end").tail(12)
        return coefvar(sub["count"])

    cv_loans = (q_counts.groupby("museum_id", group_keys=False)[["quarter_end", "count"]]
                        .apply(cv_last_12)
                        .rename("cv_loans"))
    vol_norm = minmax_norm(cv_loans.fillna(cv_loans.max() or 1.0))
    volatility_score = (100 * (1 - vol_norm)).rename("volatility_score")

    # ---- confidence ----
    # years since last loan
    last_loan = df.groupby("museum_id")["loan_date"].max()
    years_since = (pd.Timestamp(f"{current_year}-12-31") - last_loan).dt.days / 365.25
    rec_i = years_since.apply(lambda y: recency_index(y if pd.notna(y) else 9.9)).rename("recency_i")

    # conf = 0.5 * clamp(loans_3y / 20, 0, 1) + 0.5 * recency_index(years_since_last_loan)
    clamp = lambda s: np.minimum(np.maximum(s.astype(float), 0.0), 1.0)
    conf = (0.5 * clamp((loans_3y.reindex(museums["museum_id"]).fillna(0) / 20.0)) +
            0.5 * rec_i.reindex(museums["museum_id"]).fillna(0.0)).rename("confidence")

    # ---- assemble feature table (museum-indexed) ----
    feat = pd.DataFrame(index=museums["museum_id"].values)
    feat = feat.join([vis_n, age_n, loans_n, len_n, volatility_score], how="left")

    # score = 100 * (0.4*vis_n + 0.2*age_n + 0.3*loans_n + 0.1*len_n)
    museum_score = (100.0 * (0.4*feat["vis_n"].fillna(0) +
                             0.2*feat["age_n"].fillna(0) +
                             0.3*feat["loans_n"].fillna(0) +
                             0.1*feat["len_n"].fillna(0))).rename("museum_score")

    # tiers (optional, mirrors your style)
    def vol_tier(x: float) -> str:
        if x >= 66.6667: return "Low Volatility"
        if x >= 33.3333: return "Medium Volatility"
        return "High Volatility"

    # ---- output table ----
    out = pd.DataFrame({
        "museum_id": feat.index,
        "museum_score": museum_score.round(2),
        "volatility_score": feat["volatility_score"].round(2),
        "confidence": conf.clip(0,1).round(3),
        "created_at": iso_created_at(),
    })

    # drivers blob
    out["drivers_json"] = out.apply(lambda r: {
        "vis_log1p": float(vis_log.get(r["museum_id"], 0.0) or 0.0),
        "age_years": float(museums.set_index("museum_id").loc[r["museum_id"], "age"])
                     if r["museum_id"] in museums.set_index("museum_id").index and
                        pd.notna(museums.set_index("museum_id").loc[r["museum_id"], "age"]) else 0.0,
        "loans_3y": int(loans_3y.get(r["museum_id"], 0) or 0),
        "avg_loan_length_months": float(avg_len.get(r["museum_id"], np.nan)) if r["museum_id"] in avg_len.index and pd.notna(avg_len.get(r["museum_id"])) else None,
        "cv_loans_12q": float(cv_loans.get(r["museum_id"], np.nan)) if r["museum_id"] in cv_loans.index and pd.notna(cv_loans.get(r["museum_id"])) else None,
    }, axis=1)

    # add museum meta
    out = out.merge(museums[["museum_id", "name", "location"]], on="museum_id", how="left")

    # sort and return
    out = out.sort_values("museum_score", ascending=False).reset_index(drop=True)
    return out

def main(argv: Optional[list[str]] = None):
    args = parse_args(argv)
    museums = load_json(args.museums)
    loans    = load_json(args.loans)

    out = compute_museum_scores(museums, loans, args.current_year)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(out)} museum rows → {args.output}")
    print("Score summary:",
          f"min={out['museum_score'].min():.1f},",
          f"mean={out['museum_score'].mean():.1f},",
          f"max={out['museum_score'].max():.1f}")
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
