import argparse, json, sys
from typing import List, Optional
from datetime import datetime
import numpy as np
import pandas as pd
from utils import safe_ratio, to_quarter, coefvar, minmax_norm, recency_index, iso_created_at, load_json

def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser()
    p.add_argument("--galleries", required=True)
    p.add_argument("--dealers",   required=True)
    p.add_argument("--earnings",  required=True)
    p.add_argument("--top_city",  required=True)
    p.add_argument("--output",    required=True)
    p.add_argument("--current-year", type=int, default=datetime.now().year)
    return p.parse_args(argv)

def dealer_cagr(g: pd.DataFrame, current_year: int, t: int = 3) -> float:
    window = range(current_year - t + 1, current_year + 1)
    g = g[g["year"].isin(window)].sort_values("year")
    if len(g) < 2:
        return np.nan
    E0, Et = g.iloc[0]["earnings"], g.iloc[-1]["earnings"]
    years = g.iloc[-1]["year"] - g.iloc[0]["year"]
    if E0 > 0 and years > 0:
        return (Et / E0) ** (1 / years) - 1
    return np.nan

def compute_gallery_score(galleries: pd.DataFrame,
                          dealers: pd.DataFrame,
                          earnings: pd.DataFrame,
                          top_city: List[str],
                          current_year: int) -> pd.DataFrame:

    # required columns
    req_g = {"gallery_id", "location", "name"}
    req_d = {"dealer_id", "gallery_id", "dealer_name", "number_of_artists"}
    req_e = {"earnings_id", "dealer_id", "year", "earnings"}
    miss = req_g - set(galleries.columns); assert not miss, f"galleries missing: {miss}"
    miss = req_d - set(dealers.columns);   assert not miss, f"dealers missing: {miss}"
    miss = req_e - set(earnings.columns);  assert not miss, f"earnings missing: {miss}"

    # merge core tables
    df = (
        dealers.merge(galleries[["gallery_id", "location", "name"]],
                      on="gallery_id", how="left")
               .merge(earnings[["dealer_id", "year", "earnings"]],
                      on="dealer_id", how="inner")
    )

    # feature guards
    has_shows = "shows" in df.columns

    # binary top city
    df["is_top_city"] = df["location"].isin(top_city).astype(int)

    # group by dealer
    grp = df.groupby("dealer_id", dropna=False)

    roster_size = grp["number_of_artists"].sum().rename("roster_size")

    latest_year_earning = grp.apply(
        lambda g: g.loc[g["year"] == current_year, "earnings"].sum()
    ).rename("latest_year_earning")

    earnings_growth = grp.apply(lambda g: dealer_cagr(g, current_year)).rename("earnings_growth")

    shows_n = grp.apply(
        lambda g: g.loc[g["year"] >= current_year - 2, "shows"].sum() if has_shows else 0
    ).rename("shows_n")

    top_city_flag = df.groupby("dealer_id")["is_top_city"].max().rename("is_top_city")

    # volatility over last 5y (per dealer)
    df5 = df[df["year"] >= current_year - 4]
    cv_per_dealer = df5.groupby("dealer_id")["earnings"].apply(coefvar)
    volatility_score = (100 * (1 - minmax_norm(cv_per_dealer))).rename("volatility_score")

    feat = pd.DataFrame(index=grp.groups.keys())
    feat = feat.join([roster_size, latest_year_earning, earnings_growth, shows_n, top_city_flag, volatility_score], how="left")

    meta = (df.drop_duplicates("dealer_id")
              .set_index("dealer_id")[["gallery_id", "name", "location"]])
    feat = feat.join(meta, how="left")

    # fill NA baselines
    feat = feat.fillna({
        "roster_size": 0,
        "latest_year_earning": 0.0,
        "earnings_growth": 0.0,
        "shows_n": 0,
        "is_top_city": 0,
        "volatility_score": 0.0
    })

    # normalisations for score formula
    roster_n = minmax_norm(feat["roster_size"]).rename("roster_n")
    earn_n   = minmax_norm(feat["latest_year_earning"]).rename("earn_n")
    cagr_n   = minmax_norm(feat["earnings_growth"].fillna(0.0)).rename("cagr_n")
    shows_nn = minmax_norm(feat["shows_n"]).rename("shows_n_norm")

    # gallery_score using weights
    gallery_score = (100.0 * (
        0.35 * earn_n +
        0.25 * cagr_n +
        0.15 * feat["is_top_city"].astype(float) +
        0.10 * roster_n +
        0.15 * shows_nn
    )).rename("gallery_score")

    dealers_per_gallery = dealers.groupby("gallery_id").size().rename("dealer_count")
    feat = feat.join(dealers_per_gallery, on="gallery_id")

    years_with_earnings = df.groupby("dealer_id")["year"].nunique().rename("years_with_earnings")

    # clamp(x/5, 0, 1)
    clamp = lambda s: np.minimum(np.maximum(s.astype(float) / 5.0, 0.0), 1.0)
    conf = (0.5 * clamp(feat["dealer_count"].fillna(0)) +
            0.5 * clamp(years_with_earnings.reindex(feat.index).fillna(0))).rename("confidence")

    # final output table
    out = pd.DataFrame({
        "dealer_id": feat.index.values,
        "gallery_id": feat["gallery_id"].values,
        "gallery_name": feat["name"].values,
        "location": feat["location"].values,
        "gallery_score": gallery_score.round(2),
        "volatility_score": feat["volatility_score"].round(2),
        "roster_size": feat["roster_size"].astype(int),
        "latest_year_earning": feat["latest_year_earning"].round(2),
        "earnings_growth": feat["earnings_growth"].round(4),
        "shows_n": feat["shows_n"].astype(int),
        "is_top_city": feat["is_top_city"].astype(int),
        "confidence": conf.clip(0, 1).round(3),
        "created_at": iso_created_at()
    })

    # drivers
    out["drivers_json"] = out.apply(lambda r: {
        "roster_size": int(feat.loc[r["dealer_id"], "roster_size"]),
        "latest_year_earning": float(feat.loc[r["dealer_id"], "latest_year_earning"]),
        "earnings_growth": float(feat.loc[r["dealer_id"], "earnings_growth"]),
        "shows_last3y": int(feat.loc[r["dealer_id"], "shows_n"]),
        "cv_5y": float(cv_per_dealer.get(r["dealer_id"], np.nan)),
    }, axis=1)

    # sorting by score
    out = out.sort_values("gallery_score", ascending=False).reset_index(drop=True)
    return out

def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    galleries = load_json(args.galleries)
    dealers   = load_json(args.dealers)
    earnings  = load_json(args.earnings)
    top_city  = load_json(args.top_city)
    
    if isinstance(top_city, pd.DataFrame):
        if top_city.shape[1] != 1:
            print("isnt single column", file=sys.stderr)
            sys.exit(1)
        top_city = top_city.iloc[:, 0].dropna().astype(str).tolist()
    elif isinstance(top_city, pd.Series):
        top_city = top_city.dropna().astype(str).tolist()
    else:
        try:
            top_city = list(top_city)
        except Exception:
            sys.exit(1)

    out = compute_gallery_score(galleries, dealers, earnings, top_city, args["current_year"] if isinstance(args, dict) else args.current_year)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
