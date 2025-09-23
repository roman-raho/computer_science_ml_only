import json
import argparse
import sys
from typing import List
import pandas as pd
from utils import iso_created_at, minmax_norm

INSTITUTION_KWRDS = ["museum", "institute", "university"] # for a score of 1 or 0
BONUS_METHODS = ["commissioned", "bequest", "gift"] # for a score of 1 or 0

W_RC = 0.6 # the weight for record count 
W_AGE = 0.4 # the weight for age
BONUS_PER_FLAG = 0.05 # the weight for insitutions

def parse_args(argv: List[str] = None): # setting up the cmd line interface the run the python file
  p = argparse.ArgumentParser(description="Compute provenance_score from raw provenance JSON")
  p.add_argument("--input", required=True, help="Path to raw provenance JSON (array of objects)")
  p.add_argument("--output", required=True,help="Path to write processed provenance_score JSON.")
  p.add_argument("--current-year",type=int, default=2024,help="Fixed current year for age calculation (default: 2024).")
  return p.parse_args(argv)

def compute_flags(df: pd.DataFrame) -> pd.DataFrame: # compute the flags for the inst and bonus
  owner_1 = df["owner"].fillna("").str.lower() # get the owner
  method_1 = df["acquisition_method"].fillna("").str.lower() # get the method

  inst = owner_1.apply(lambda x: any(k in x for k in INSTITUTION_KWRDS)).astype(int) # if exisit on owner 1 else 0
  meth = method_1.apply(lambda x: any(x == k or k in x for k in BONUS_METHODS)).astype(int) # if exists on the method 1 else 0

  return pd.DataFrame({ # return a dataframe with the flags and method
    "institutional_flag": inst,
    "method_bonus_flag": meth,
  })

def main(argv: List[str] = None): # main to accept arguements from cmd line
  args = parse_args(argv) # use function defined before

  try:
    raw = pd.read_json(args.input) # attempt to read from file provided
  except ValueError as e:
    print(f"Error: Failed to read JSON file from {args.input}: {e}", file=sys.stderr) # else print error to the console in format standard erro
    sys.exit(1)

  required_cols = {"artwork_id", "owner", "acquisition_date","acquisition_method"} # check that all required columns are present
  missing = required_cols - set(raw.columns) # if they are not missing will be true
  if missing:
    print(f"Error: Missing required columns in input {sorted(missing)}", file=sys.stderr) 
    sys.exit(1) # end program if they are missing

  years = pd.to_datetime(raw["acquisition_date"], errors="coerce").dt.year # get the years as a datatime from the raw acquisition date
  raw = raw.assign(_years=years) # assign the _years columns to the years series we just defined

  flags=compute_flags(raw) # compute the flags from the raw data
  raw = pd.concat([raw,flags],axis=1) # add it to the raw dataframe

  grp = raw.groupby("artwork_id",dropna=False) # group the artworks by their IDs

  records_count = grp.size().rename("records_count") # count the number of records 
  earliest_year = grp["_years"].min().fillna(args.current_year).astype(int).rename("earliest_year") # get the earliest year from the _years column
  age_years = (args.current_year - earliest_year).clip(lower=0).rename("age_years") # compute the earliest time the work was shown

  institutional_flag = grp["institutional_flag"].max().rename("institutional_flag") # if one institution exists on the artwork then mark it as 1 else 0
  method_bonus_flag = grp["method_bonus_flag"].max().rename("method_bonus_flag") # same as inst logic

  drivers = pd.concat([ # build the drivers dataframe
    records_count,
    earliest_year,
    age_years,
    institutional_flag,
    method_bonus_flag
  ], axis=1).reset_index()

  drivers["rc_norm"] = minmax_norm(drivers["records_count"]) # calculate the normalised values
  drivers["age_norm"] = minmax_norm(drivers["age_years"])

  base = W_RC * drivers["rc_norm"] + W_AGE * drivers["age_norm"] # calcualte the base score as defined before
  bonus = BONUS_PER_FLAG * drivers["institutional_flag"] + BONUS_PER_FLAG * drivers["method_bonus_flag"] # calculate bonus score
  final = (base + bonus).clip(upper=1.0) # calculate the final score max at 1
  score = (final * 100).round().astype(int) # convert it to correct type

  created_at = iso_created_at()
  out = pd.DataFrame({ # constrcut JSON row
    "artwork_id": drivers["artwork_id"],
    "provenance_score": score,
    "drivers_json": drivers.apply(lambda r: {
      "records_count": int(r["records_count"]),
      "earliest_year": int(r["earliest_year"]),
      "age_years": int(r["age_years"]),
      "institutional_flag": int(r["institutional_flag"]),
      "method_bonus_flag": int(r["method_bonus_flag"]),
    }, axis=1),
    "created_at": created_at
  })

  out_records = out.to_dict(orient="records") # create a dic in type of records for each row
  with open(args.output, "w",encoding = "utf-8") as f:
    json.dump(out_records, f, ensure_ascii=False, indent=2) # store in file path provided
  
  print(f"Wrote {len(out_records)} rows -> {args.output}")
  print(f"Score summary: min={int(score.min())}, mean={round(float(score.mean()),1)}, max={int(score.max())}")

if __name__ == "__main__": # run code
  main()