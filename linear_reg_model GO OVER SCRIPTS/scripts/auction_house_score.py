import argparse, json, sys
from typing import List, Optional 
from datetime import datetime
import numpy as np
import pandas as pd
from utils import safe_ratio, to_quarter, coefvar, minmax_norm, recency_index, iso_created_at, load_json

def parse_args(argv: Optional[List[str]] = None):
  p = argparse.ArgumentParser()
  p.add_argument("--houses",   required=True)
  p.add_argument("--auctions", required=True)
  p.add_argument("--biddata",  required=True)
  p.add_argument("--bidders",  required=False)
  p.add_argument("--output",   required=True)
  p.add_argument("--current-year", type=int, default=datetime.now().year)
  return p.parse_args(argv)

def compute_house_scores(houses: pd.DataFrame,
                         auctions: pd.DataFrame,
                         biddata: pd.DataFrame,
                         current_year: int) -> pd.DataFrame:
  

  req_h = {"auction_house_id", "auction_house_name", "location", "auction_house_rating"}
  req_a = {"auction_id", "auction_house_id", "artwork_id", "date_of_auction"}
  req_b = {"bid_data_id", "auction_id", "number_of_bids", "reserve_price", "final_price"}
 
  miss = req_h - set(houses.columns);  assert not miss, f"houses missing: {miss}"
  miss = req_a - set(auctions.columns); assert not miss, f"auctions missing: {miss}"
  miss = req_b - set(biddata.columns);  assert not miss, f"bid_data missing: {miss}"


  df = (biddata.merge(auctions[["auction_id", "auction_house_id","date_of_auction"]], # merge the biddata auctions and houses so it is tied to a house and a data
                      on="auction_id", how="left") # and filtered to only valid houses before i can compute any metrics
                      .merge(houses[["auction_house_id"]], on="auction_house_id", how="inner"))
  
  df["date_of_auction"] = pd.to_datetime(df["date_of_auction"], errors="coerce")
  df = df.dropna(subset=["date_of_auction"]) # drop any null

  # reserve mask (true only when numeric is greater than 0)
  rp = pd.to_numeric(df["reserve_price"], errors="coerce")
  fp = pd.to_numeric(df["final_price"], errors="coerce")

  df["has_reserve"] = rp.notna() & (rp > 0)
  df["sold"] = (~df["has_reserve"]) | (fp >= rp)
  df["premium"] = np.where(df["has_reserve"] & fp.notna(), (fp - rp) / rp, np.nan)

  #3yr window
  start_3y = pd.Timestamp(f"{current_year-3}-01-01") # start of the 3 years
  df3 = df[df["date_of_auction"] >= start_3y] # create a copy of the data with only auctions that happened post 3yr

  #total hamme in last 3yr - use log to reduce skew
  depth = (df3.groupby("auction_house_id")["final_price"]
              .sum().pipe(np.log1p).rename("depth"))
  
  grp = df.groupby("auction_house_id", dropna=False) # group all auction houses by id
  lots_total = grp.size().rename("lots_total") # get total lots from house
  lots_with_reserve=grp["has_reserve"].sum().rename("lots_with_reserve") # get total lots with a reserve
  lots_sold = grp["sold"].sum().rename("lots_sold") # get total lots sold
  clearance=safe_ratio(lots_sold, lots_with_reserve) # calcaulte the ration to lots sold and lots with reserve
  avg_bids = grp["number_of_bids"].mean().rename("avg_bids") # calculate the number of average bids per year
  median_premium = grp["premium"].median().rename("median_premium") # calculate the median premium

  # volatility
  df_q = df.copy() # create a copy
  df_q["quarter_end"] = to_quarter(df_q["date_of_auction"]) # add column that contains the end of the quarter
  q_median = (df_q.groupby(["auction_house_id", "quarter_end"])["final_price"].median().reset_index()) # within each group (house, quarter) it takes the median hammer price

  #compute coefvar per house over last 12 q
  def cv_last_12(sub: pd.DataFrame) -> float:
    sub = sub.sort_values("quarter_end").tail(12)
    return coefvar(sub["final_price"])
  
  cv = (
    q_median
      .groupby("auction_house_id", group_keys=False)[["quarter_end","final_price"]]
      .apply(cv_last_12)
      .rename("cv")
  ) # split the table into one mini dataset per house then run the volatility function. sorting by quarter. 

  feat = pd.DataFrame(index=grp.groups.keys())
  feat = feat.join([lots_total, lots_with_reserve, lots_sold, clearance, avg_bids, median_premium], how='left')
  feat = feat.join(depth, how='left')
  feat = feat.join(cv, how='left')
  feat = feat.fillna({
    "median_premium": 0.0,
    "depth": 0.0
  })

  clr_n = minmax_norm(feat["lots_sold"] / feat["lots_with_reserve"].replace(0, np.nan)).fillna(0) # fraction of reserved lots that actually sold
  bids_n = minmax_norm(feat["avg_bids"]) # normalised average bods
  prem_n = minmax_norm(feat["median_premium"]) # median premium
  depth_n = minmax_norm(feat["depth"]) # log of total hammer in last 3 years -> bigger houses higher raw depth

  # cv -> higher is worse volatility
  cv_norm = minmax_norm(feat["cv"].fillna(feat["cv"].max() or 1.0))
  vol_score = (1 - cv_norm) * 100.0 # calculate volatility score

  auction_house_score = 100.0 * (0.30*clr_n + 0.25*bids_n + 0.25*prem_n + 0.20*depth_n)

  def tier(x: float) -> str: # function to calculate the tier
    if x >= 66.6667: return "Tier 1"
    if x >= 33.3333: return "Tier 2"
    return "Tier 3"

  def vol_tier(x: float) -> str: # function to calculate volatily tier
    if x >= 66.6667: return "Low Volatility"
    if x >= 33.3333: return "Medium Volatility"
    return "High Volatility"
  
  last_sale = df.groupby("auction_house_id")["date_of_auction"].max() # get last sale
  years_since = (pd.Timestamp(f"{current_year}-12-31") - last_sale).dt.days / 365.25 # get years since last sale
  rec_i = years_since.apply(lambda y: recency_index(y if pd.notna(y) else 9.9)) # weight each one based on how recent it was
  conf = 0.6 * np.minimum(feat["lots_total"]/50.0, 1.0) + 0.4 * rec_i.fillna(0.0) # calculate the confidence score

  out = pd.DataFrame({ # form the output for the json
    "auction_house_id": feat.index.values,
    "auction_house_score": auction_house_score.round(2),
    "auction_house_tier": auction_house_score.apply(tier),
    "volatility_score": vol_score.round(2),
    "volatility_tier": vol_score.apply(vol_tier),
    "clearance_rate": (feat["lots_sold"] / feat["lots_with_reserve"].replace(0, np.nan)).fillna(0).round(3),
    "avg_bids": feat["avg_bids"].round(2),
    "median_premium": feat["median_premium"].round(3),
    "depth_index": depth_n.round(3),
    "confidence": conf.clip(0,1).round(3),
    "created_at": iso_created_at()
  })

  out["drivers_json"] = out.apply(lambda r: { # the drivers for the output
    "lots_total": int(lots_total.get(r["auction_house_id"], 0) or 0),
    "lots_with_reserve": int(lots_with_reserve.get(r["auction_house_id"], 0) or 0),
    "lots_sold": int(lots_sold.get(r["auction_house_id"], 0) or 0),
    "cv_last12": float(feat.loc[r["auction_house_id"], "cv"]) if pd.notna(feat.loc[r["auction_house_id"], "cv"]) else None
  }, axis=1)

  out = out.merge(houses[["auction_house_id","auction_house_name","location"]], 
          on="auction_house_id", how="left")
  
  out = out.sort_values("auction_house_score", ascending=False).reset_index(drop=True)
  return out

def main(argv: Optional[List[str]] = None):
  args = parse_args(argv)
  houses = load_json(args.houses)
  auctions = load_json(args.auctions)
  biddata = load_json(args.biddata)

  if args.bidders:
    bidders = load_json(args.bidders)
    if not {"bid_data_id","bid_amount"}.issubset(bidders.columns):
      sys.exit(1)
    else:
      counts = (bidders.groupby("bid_data_id").size() # counts the number of bidders
                 .rename("number_of_bids_from_bidders"))
      biddata = biddata.merge(counts, left_on="bid_data_id", right_index=True, how="left") # merge the biddata onto counts
      # prefer explicit number_of_bids column; fill from bidders where missing
      biddata["number_of_bids"] = biddata["number_of_bids"].fillna(biddata["number_of_bids_from_bidders"])
      biddata.drop(columns=["number_of_bids_from_bidders"], inplace=True, errors="ignore") 

  out = compute_house_scores(houses, auctions, biddata, args.current_year)

  with open(args.output, "w", encoding="utf-8") as f:
    json.dump(out.to_dict(orient="records"), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
  main()