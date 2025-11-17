import json
import argparse
import sys
from typing import List
import pandas as pd
from utils import iso_created_at, minmax_norm

# scores for bonus 1/0
INSTITUTION_KWRDS = ["museum", "institute", "university"] 
BONUS_METHODS = ["commissioned", "bequest", "gift"] 

W_RC = 0.6  
W_AGE = 0.4
BONUS_PER_FLAG = 0.05

def parse_args(argv: List[str] = None): # setting up the cmd line interface the run the python file
  p = argparse.ArgumentParser()
  p.add_argument("--input", required=True)
  p.add_argument("--output", required=True)
  p.add_argument("--current-year",type=int, default=2024)
  return p.parse_args(argv)

def compute_flags(df: pd.DataFrame) -> pd.DataFrame:
  owner_1 = df["owner"].fillna("").str.lower() 
  method_1 = df["acquisition_method"].fillna("").str.lower()

  # if exisit on owner 1 else 0
  inst = owner_1.apply(lambda x: any(k in x for k in INSTITUTION_KWRDS)).astype(int) 
  meth = method_1.apply(lambda x: any(x == k or k in x for k in BONUS_METHODS)).astype(int)

  return pd.DataFrame({
    "institutional_flag": inst,
    "method_bonus_flag": meth,
  })

def main(argv: List[str] = None):
  args = parse_args(argv) 

  # exception handling
  try:
    raw = pd.read_json(args.input)
  except ValueError as e:
    print(f"couldnt to read JSON file from {args.input}: {e}", file=sys.stderr)
    sys.exit(1)

  required_cols = {"artwork_id", "owner", "acquisition_date","acquisition_method"}
  missing = required_cols - set(raw.columns)
  
  # handling missing case
  if missing:
    print(f"missing required columns in input {sorted(missing)}", file=sys.stderr) 
    sys.exit(1) # end program if they are missing

  years = pd.to_datetime(raw["acquisition_date"], errors="coerce").dt.year
  raw = raw.assign(_years=years) 

  flags=compute_flags(raw)
  raw = pd.concat([raw,flags],axis=1)

  grp = raw.groupby("artwork_id",dropna=False)  

  records_count = grp.size().rename("records_count")
  earliest_year = grp["_years"].min().fillna(args.current_year).astype(int).rename("earliest_year")
  age_years = (args.current_year - earliest_year).clip(lower=0).rename("age_years")

  institutional_flag = grp["institutional_flag"].max().rename("institutional_flag")
  method_bonus_flag = grp["method_bonus_flag"].max().rename("method_bonus_flag")

  # build drivers
  drivers = pd.concat([
    records_count,
    earliest_year,
    age_years,
    institutional_flag,
    method_bonus_flag
  ], axis=1).reset_index()

  drivers["rc_norm"] = minmax_norm(drivers["records_count"])
  drivers["age_norm"] = minmax_norm(drivers["age_years"])

  # use predefined formulas
  base = W_RC * drivers["rc_norm"] + W_AGE * drivers["age_norm"]
  bonus = BONUS_PER_FLAG * drivers["institutional_flag"] + BONUS_PER_FLAG * drivers["method_bonus_flag"]
  final = (base + bonus).clip(upper=1.0)
  score = (final * 100).round().astype(int)

  # create output
  created_at = iso_created_at()
  out = pd.DataFrame({
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

  out_records = out.to_dict(orient="records")
  with open(args.output, "w",encoding = "utf-8") as f:
    json.dump(out_records, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
  main()