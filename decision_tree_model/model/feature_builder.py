from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

def _log(msg: str) -> None:
  ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
  print(f"[INFO {ts} UTC] {msg}")

# convert series to numeric, replace errors with Nan
def safe_num(s: pd.Series, *, clip_low: float | None = None) -> pd.Series:
  x = pd.to_numeric(s, errors="coerce")
  if clip_low is not None:
      x = x.clip(lower=clip_low)
  return x

# apply safe log
def safe_log1p(s: pd.Series) -> pd.Series:
    return np.log1p(safe_num(s, clip_low=0.0).fillna(0.0))

def build_feature_mart_all(raw_path: str, current_year: int = 2024) -> pd.DataFrame:
   
  base = Path(raw_path)

  artworks = pd.read_json(base / "artworks_v2_large.json")
  auctions = pd.read_json(base / "auctions_v2.json")
  biddata = pd.read_json(base / "bid_data_v2.json")
  houses = pd.read_json(base / "auction_houses_v2.json")
  artists = pd.read_json(base / "artists_v2_large.json")
  artists_dealers = pd.read_json(base / "artists_dealers_v2.json")
  dealers = pd.read_json(base / "dealers_v2.json")
  earnings = pd.read_json(base / "earnings_v2.json")
  collectors = pd.read_json(base / "collectors_v2.json")
  ownership = pd.read_json(base / "ownership_v2.json")
  museums = pd.read_json(base / "museums_v2.json")
  loans = pd.read_json(base / "artwork_loans_v2.json")
  provenance = pd.read_json(base / "provenance_v2.json")
  top_cities = pd.read_json(base / "top_art_cities.json", typ='series').tolist()

  auc = auctions.copy()
  auc["date_of_auction"] = pd.to_datetime(auc["date_of_auction"], errors="coerce")

  bid = biddata.copy()
  for col in ["reserve_price", "final_price", "number_of_bids"]:
    bid[col] = safe_num(bid[col])
  
  df = (
     auc.merge(bid, on="auction_id", how="left")
     .merge(artworks, on="artwork_id", how="left")
     .merge(houses, on="auction_house_id", how="left")
  )

  # rows with target
  df = df[safe_num(df["final_price"]).notna()].copy()
  df["y_price"] = safe_num(df["final_price"])
  df["y_log_price"] = safe_log1p(df["final_price"])

  # get artwork features
  df["artwork_length"] = safe_num(df["artwork_length"])
  df["artwork_width"] = safe_num(df["artwork_width"])
  df["artwork_height"] = safe_num(df.get("artwork_height", np.nan))

  df["area"] = df["artwork_length"] * df["artwork_width"]
  df["log_area"] = safe_log1p(df["area"])

  df["volume"] = df["artwork_length"] * df["artwork_width"] * df["artwork_height"]
  df["log_volume"] = safe_log1p(df["volume"])

  df["year_created"] = safe_num(df["year_created"]).fillna(current_year)
  df["artwork_age"] = (df["date_of_auction"].dt.year - df["year_created"]).clip(lower=0)

  df["signed_flag"] = df["signed"].astype(str).str.lower().isin(
    ["1", "true", "yes", "y", "t"]
  ).astype(int)

  df["log_insurance"] = safe_log1p(df["insurance_value"])

  # Auction features
  df["season_q"] = df["date_of_auction"].dt.quarter.astype("Int64").astype(str)
  df["auction_year"] = df["date_of_auction"].dt.year
  df["reserve_gt0"] = (safe_num(df["reserve_price"]).fillna(0) > 0).astype(int)
  
  with np.errstate(divide="ignore", invalid="ignore"):
    df["premium"] = np.where(
      df["reserve_price"].fillna(0) > 0,
      (df["final_price"] - df["reserve_price"]) / df["reserve_price"].replace(0, np.nan),
      np.nan,
    )

  df["is_top_city"] = df["location"].isin(top_cities).astype(int)

  # artist + dealer features
  df = df.merge(artists[["artist_id", "nationality"]], on="artist_id", how="left")
  df = df.merge(artists_dealers, on="artist_id", how="left")
  df = df.merge(dealers, on="dealer_id", how="left")

  # dealer earnings
  dealer_agg = earnings.groupby("dealer_id")["earnings"].mean().reset_index()
  dealer_agg.rename(columns={"earnings": "dealer_avg_earnings"}, inplace=True)
  df = df.merge(dealer_agg, on="dealer_id", how="left")
  df["log_dealer_earnings"] = safe_log1p(df["dealer_avg_earnings"])

  # ownernship + collectors
  owner_counts = ownership.groupby("artwork_id").size().reset_index(name="ownership_count")
  df = df.merge(owner_counts, on="artwork_id", how="left")

  # museum exposure
  loan_counts = loans.groupby("artwork_id").size().reset_index(name="loan_count")
  df = df.merge(loan_counts, on="artwork_id", how="left")

  museum_agg = museums.groupby("museum_id")["annual_visitors"].mean().reset_index()

  # provenance
  prov_counts = provenance.groupby("artwork_id").size().reset_index(name="prov_count")
  df = df.merge(prov_counts, on="artwork_id", how="left")

  _log(f"Feature mart built with shape {df.shape}, {len(df.columns)} columns.")
  return df