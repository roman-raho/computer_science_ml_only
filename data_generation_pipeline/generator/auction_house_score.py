import json
import argparse
import sys
from typing import List
import pandas as pd
from utils import iso_created_at, minmax_norm
import numpy as np

W_AV = 0.7
W_LY = 0.3

def parse_args(argv: List[str] = None): # setting up the cmd line interface the run the python file
  p = argparse.ArgumentParser(description="Compute provenance_score from raw provenance JSON")
  p.add_argument("--input", required=True, help="Path to raw auction house JSON (array of objects)")
  p.add_argument("--output", required=True,help="Path to write processed auction_house_score JSON.")
  p.add_argument("--current-year",type=int, default=2025,help="Fixed current year for age calculation (default: 2024).")
  return p.parse_args(argv)

def main(argv: List[str] = None):
  args = parse_args(argv)

  try:
    raw = pd.read_json(args.input)
  except ValueError as e:
    print(f"Error: Failed to read JSON file from {args.input}: {e}", file=sys.stderr) # else print error to the console in format standard erro
    sys.exit(1)

  required_cols = {"auction_house_id", "founded_year", "annual_auctions"}
  missing = required_cols - set(raw.columns)
  if missing:
    print(f"Error: Missing required columns in input {sorted(missing)}", file=sys.stderr) 
    sys.exit(1) # end program if they are missing

  grp = raw.groupby("auction_house_id", dropna=False)
  earliest_year = grp["founded_year"].min().fillna(args.current_year).astype(int).rename("earliest_year")
  annual_auctions = grp["annual_auctions"].max().fillna(0).rename("annual_auctions")

  age_years = (args.current_year - earliest_year).clip(lower=0).rename("age_years")

  drivers = pd.concat([
    earliest_year,
    age_years,
    annual_auctions
  ],axis=1).reset_index()

  drivers["v_norm"] = minmax_norm(drivers["annual_auctions"])
  drivers["l_norm"] = minmax_norm(drivers["age_years"])

  score = 100 * (W_AV * drivers["v_norm"] + W_LY * drivers["l_norm"])

  tier = np.where(
    score >= 100*(2/3), "Tier 1",
    np.where(score >= 100*(1/3), "Tier 2", "Tier 3")
  )

  created_at = iso_created_at()
  out = pd.DataFrame({
    "auction_house_id": drivers["auction_house_id"],
    "auction_house_score": score,
    "auction_house_tier": tier,
    "drivers_json": drivers.apply(lambda r: {
      "earliest_year": int(r["earliest_year"]),
      "age_years": int(r["age_years"]),
      "annual_auctions": int(r["annual_auctions"]),
    }, axis=1),
    "created_at": created_at,
  })

  out_records = out.to_dict(orient="records") # create a dic in type of records for each row
  with open(args.output, "w",encoding = "utf-8") as f:
    json.dump(out_records, f, ensure_ascii=False, indent=2) # store in file path provided
  
  print(f"Wrote {len(out_records)} rows -> {args.output}")
  print(f"Score summary: min={int(score.min())}, mean={round(float(score.mean()),1)}, max={int(score.max())}")

if __name__ == "__main__": # run code
  main()