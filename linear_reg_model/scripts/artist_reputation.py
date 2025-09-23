from __future__ import annotations
import argparse, json, sys
from typing import Optional, List
from datetime import datetime
import numpy as np
import pandas as pd
from utils import minmax_norm, coefvar, recency_index, iso_created_at, load_json, to_quarter

def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="Compute artist reputation scores from raw JSON.")
    p.add_argument("--artists",        required=True, help="Path to artists JSON")
    p.add_argument("--artist_dealers", required=True, help="Path to artists_dealers JSON")
    p.add_argument("--dealers",        required=True, help="Path to dealers JSON")
    p.add_argument("--earnings",       required=True, help="Path to earnings JSON (dealer_id, year, earnings)")
    p.add_argument("--loans",          required=True, help="Path to artwork_loans JSON")
    p.add_argument("--auctions",       required=True, help="Path to auctions JSON")
    p.add_argument("--biddata",        required=True, help="Path to bid_data JSON")
    p.add_argument("--artworks",       required=False, help="Path to artworks JSON (optional; artwork_id→artist_id)")
    p.add_argument("--output",         required=True, help="Path to write processed JSON")
    p.add_argument("--current-year",   type=int, default=datetime.now().year,
                   help="Current year for 3y/8q windows (default: this year)")
    return p.parse_args(argv)

def _ensure_artist_id(df: pd.DataFrame, artworks: Optional[pd.DataFrame], key_col: str) -> pd.DataFrame:
    if "artist_id" in df.columns:
        return df
    if artworks is None or key_col not in df.columns:
        raise AssertionError("Missing artist_id and no artworks mapping available.")
    return df.merge(artworks[["artwork_id", "artist_id"]], on="artwork_id", how="left")

def _dealer_latest_earnings(earn: pd.DataFrame, current_year: int) -> pd.Series:
    return (earn.loc[earn["year"] == current_year]
                .groupby("dealer_id")["earnings"]
                .sum()
                .rename("latest_earnings"))

def _dealer_cagr(earn: pd.DataFrame, current_year: int, t: int = 3) -> pd.Series:
    w = range(current_year - t + 1, current_year + 1)
    e = earn[earn["year"].isin(w)].copy()
    def _cagr(g: pd.DataFrame) -> float:
        g = g.sort_values("year")
        if len(g) < 2: return np.nan
        E0, Et = g.iloc[0]["earnings"], g.iloc[-1]["earnings"]
        years = g.iloc[-1]["year"] - g.iloc[0]["year"]
        if years <= 0 or E0 <= 0: return np.nan
        return (Et / E0) ** (1 / years) - 1
    return (e.groupby("dealer_id")[["year", "earnings"]]
              .apply(_cagr)
              .rename("dealer_cagr"))

def _simple_slope(series: pd.Series) -> float:
    y = series.dropna().astype(float)
    n = len(y)
    if n < 2: return 0.0
    x = np.arange(n, dtype=float)
    x_mean, y_mean = x.mean(), y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0: return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)

def compute_artist_scores(artists: pd.DataFrame,
                          artist_dealers: pd.DataFrame,
                          dealers: pd.DataFrame,
                          earnings: pd.DataFrame,
                          loans: pd.DataFrame,
                          auctions: pd.DataFrame,
                          biddata: pd.DataFrame,
                          current_year: int,
                          artworks: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    req_art = {"artist_id", "name"}
    req_ad  = {"artist_id", "dealer_id"}
    req_d   = {"dealer_id", "gallery_id", "number_of_artists"}
    req_e   = {"dealer_id", "year", "earnings"}
    req_b   = {"bid_data_id", "auction_id", "number_of_bids", "reserve_price", "final_price"}
    miss = req_art - set(artists.columns);        assert not miss, f"artists missing: {miss}"
    miss = req_ad - set(artist_dealers.columns);  assert not miss, f"artist_dealers missing: {miss}"
    miss = req_d - set(dealers.columns);          assert not miss, f"dealers missing: {miss}"
    miss = req_e - set(earnings.columns);         assert not miss, f"earnings missing: {miss}"
    miss = req_b - set(biddata.columns);          assert not miss, f"biddata missing: {miss}"
    assert "loan_date" in loans.columns, "loans missing 'loan_date'"
    assert {"auction_id", "artwork_id"}.issubset(auctions.columns), "auctions missing 'auction_id' and/or 'artwork_id'"

    loans = loans.copy()
    auctions = auctions.copy()
    biddata = biddata.copy()
    loans["loan_date"] = pd.to_datetime(loans["loan_date"], errors="coerce")
    auctions["date_of_auction"] = pd.to_datetime(auctions.get("date_of_auction", pd.NaT), errors="coerce")
    biddata["reserve_price"]   = pd.to_numeric(biddata["reserve_price"], errors="coerce")
    biddata["final_price"]     = pd.to_numeric(biddata["final_price"], errors="coerce")
    biddata["number_of_bids"]  = pd.to_numeric(biddata["number_of_bids"], errors="coerce")

    artworks_df = None
    if artworks is not None:
        assert {"artwork_id", "artist_id"}.issubset(artworks.columns), "artworks needs artwork_id, artist_id"
        artworks_df = artworks[["artwork_id", "artist_id"]].drop_duplicates()

    loans = _ensure_artist_id(loans, artworks_df, key_col="artwork_id")
    auctions = auctions.merge(biddata[["auction_id", "number_of_bids", "reserve_price", "final_price"]],
                              on="auction_id", how="left")
    auctions = _ensure_artist_id(auctions, artworks_df, key_col="artwork_id")

    start_3y = pd.Timestamp(f"{current_year-3}-01-01")
    loans_3y = (loans.loc[loans["loan_date"] >= start_3y]
                     .groupby("artist_id")
                     .size()
                     .rename("loans_3y"))
    inst_n = minmax_norm(loans_3y).rename("inst_n")

    auc3 = auctions[auctions["date_of_auction"] >= start_3y].copy()

    sales_3y = auc3.groupby("artist_id").size().rename("sales_3y")
    sales_n  = minmax_norm(sales_3y).rename("sales_n")

    mean_bids = auc3.groupby("artist_id")["number_of_bids"].mean().fillna(0.0)
    bids_n = minmax_norm(mean_bids).rename("bids_n")

    has_reserve = (auc3["reserve_price"].notna()) & (auc3["reserve_price"] > 0)
    prem = ((auc3["final_price"] - auc3["reserve_price"]) / auc3["reserve_price"]).where(has_reserve)
    prem_med = auc3.assign(prem=prem).groupby("artist_id")["prem"].median().fillna(0.0)
    prem_n = minmax_norm(prem_med).rename("prem_n")

    ad = artist_dealers[["artist_id", "dealer_id"]].dropna().drop_duplicates()
    latest_e = _dealer_latest_earnings(earnings, current_year)
    cagr     = _dealer_cagr(earnings, current_year, t=3)

    dealer_feat = pd.DataFrame({
        "latest_earnings_log1p": np.log1p(latest_e),
        "dealer_cagr": cagr
    })

    artist_dealer = (ad.merge(dealer_feat, left_on="dealer_id", right_index=True, how="left")
                       .groupby("artist_id", as_index=True)
                       .agg({"latest_earnings_log1p":"median", "dealer_cagr":"median"}))

    earn_level_n = minmax_norm(artist_dealer["latest_earnings_log1p"].fillna(0.0)).rename("earn_level_n")
    cagr_med_n   = minmax_norm(artist_dealer["dealer_cagr"].fillna(0.0)).rename("cagr_med_n")
    dealer_strength = (0.6*earn_level_n + 0.4*cagr_med_n).rename("dealer_strength")

    q = auc3.copy()
    q["quarter_end"] = to_quarter(q["date_of_auction"])
    q_median = (q.groupby(["artist_id","quarter_end"])["final_price"]
                  .median()
                  .reset_index())

    mom = (q_median.sort_values("quarter_end")
                  .groupby("artist_id")["final_price"]
                  .apply(lambda s: _simple_slope(np.log1p(s.tail(8))))
                  .rename("mom"))
    mom_n = minmax_norm(mom.fillna(0.0)).rename("mom_n")

    cv = (q_median.sort_values("quarter_end")
                  .groupby("artist_id")["final_price"]
                  .apply(lambda s: coefvar(s.tail(12)))
                  .rename("cv"))
    vol_norm = minmax_norm(cv.fillna(cv.max() or 1.0))
    volatility_score = (100 * (1 - vol_norm)).rename("volatility_score")

    last_sale = auc3.groupby("artist_id")["date_of_auction"].max()
    years_since = (pd.Timestamp(f"{current_year}-12-31") - last_sale).dt.days / 365.25
    rec_i = years_since.apply(lambda y: recency_index(y if pd.notna(y) else 9.9))
    conf = (0.5 * np.minimum((sales_3y.reindex(artists["artist_id"]).fillna(0) / 15.0), 1.0) +
            0.5 * rec_i.reindex(artists["artist_id"]).fillna(0.0)).rename("confidence")

    unique_artist_ids = artists["artist_id"].drop_duplicates().tolist()
    
    feat = pd.DataFrame(index=pd.Index(unique_artist_ids, name="artist_id"))
    feat = feat.join([inst_n, sales_n, bids_n, prem_n, dealer_strength, mom_n, volatility_score], how="left")
    feat = feat.fillna({
        "inst_n": 0.0, "sales_n": 0.0, "bids_n": 0.0, "prem_n": 0.0,
        "dealer_strength": 0.0, "mom_n": 0.0, "volatility_score": 0.0
    })

    reputation_score = (100.0 * (
        0.25*feat["inst_n"] +
        0.20*feat["sales_n"] +
        0.15*feat["bids_n"] +
        0.15*feat["prem_n"] +
        0.15*feat["dealer_strength"] +
        0.10*feat["mom_n"]
    )).rename("reputation_score")

    feat_with_scores = feat.copy()
    feat_with_scores["reputation_score"] = reputation_score
    feat_with_scores["confidence"] = conf.reindex(feat_with_scores.index).fillna(0.0).clip(0,1)
    
    feat_with_scores = feat_with_scores.reset_index()
    
    out = pd.DataFrame({
        "artist_id": feat_with_scores["artist_id"],
        "reputation_score": feat_with_scores["reputation_score"].round(2),
        "volatility_score": feat_with_scores["volatility_score"].round(2),
        "confidence": feat_with_scores["confidence"].round(3),
        "created_at": iso_created_at()
    })

    out = out.merge(artists[["artist_id","name","nationality"]], on="artist_id", how="left")

    feat_indexed = feat
    out["drivers_json"] = out["artist_id"].apply(lambda aid: {
        "inst_n": float(feat_indexed.at[aid, "inst_n"]) if aid in feat_indexed.index else 0.0,
        "sales_n": float(feat_indexed.at[aid, "sales_n"]) if aid in feat_indexed.index else 0.0,
        "bids_n": float(feat_indexed.at[aid, "bids_n"]) if aid in feat_indexed.index else 0.0,
        "prem_n": float(feat_indexed.at[aid, "prem_n"]) if aid in feat_indexed.index else 0.0,
        "dealer_strength": float(feat_indexed.at[aid, "dealer_strength"]) if aid in feat_indexed.index else 0.0,
        "mom_n": float(feat_indexed.at[aid, "mom_n"]) if aid in feat_indexed.index else 0.0,
        "cv_quarterly": float(cv.get(aid, np.nan)) if aid in cv.index else None,
        "sales_3y": int(sales_3y.get(aid, 0) or 0),
        "loans_3y": int(loans_3y.get(aid, 0) or 0),
    })

    return out.sort_values("reputation_score", ascending=False).reset_index(drop=True)

def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    artists        = load_json(args.artists)
    artist_dealers = load_json(args.artist_dealers)
    dealers        = load_json(args.dealers)
    earnings       = load_json(args.earnings)
    loans          = load_json(args.loans)
    auctions       = load_json(args.auctions)
    biddata        = load_json(args.biddata)
    artworks       = load_json(args.artworks) if args.artworks else None

    out = compute_artist_scores(artists, artist_dealers, dealers, earnings,
                                loans, auctions, biddata, args.current_year, artworks)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(out)} artist rows → {args.output}")
    print("Score summary:",
          f"min={out['reputation_score'].min():.1f},",
          f"mean={out['reputation_score'].mean():.1f},",
          f"max={out['reputation_score'].max():.1f}")
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