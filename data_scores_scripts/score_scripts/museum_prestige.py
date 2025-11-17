import json
import argparse
import math
import sys
from datetime import datetime, timezone
from typing import List
import pandas as pd
from utils import iso_created_at, minmax_norm
import numpy as np

W_AV = 0.7
W_AG = 0.3

def parse_args(argv: List[str] = None):
  p = argparse.ArgumentParser()
  p.add_argument("--input", required=True)
  p.add_argument("--output", required=True)
  return p.parse_args(argv)

def main(argv: List[str] = None):
  args = parse_args(argv)

  # exception handling
  try:
    raw = pd.read_json(args.input)
  except ValueError as e:
    print(f"couldnt to read JSON file from {args.input}: {e}", file=sys.stderr) # else print error to the console in format standard error
    sys.exit(1)
  
  required_cols = {"museum_id", "age", "annual_visitors"}
  missing = required_cols - set(raw.columns)

  if missing:
    print(f"missing required columns in input {sorted(missing)}", file=sys.stderr) 
    sys.exit(1)
  
  # data cleaning
  raw["annual_visitors"] = pd.to_numeric(raw["annual_visitors"], errors="coerce").fillna(0).clip(lower=0)
  raw["age"] = pd.to_numeric(raw["age"], errors="coerce").fillna(0).clip(lower=0)
  
  grp = raw.groupby("museum_id",dropna=False)

  annual_visitors = grp["annual_visitors"].max().rename("annual_visitors")
  age = grp["age"].max().rename("age")

  visitors_log = np.log10(annual_visitors.clip(lower=1)).rename("visitors_log")

  # building drivers
  drivers = pd.concat([
    annual_visitors,
    visitors_log,
    age
  ], axis=1).reset_index()

  drivers["av_norm"] = minmax_norm(drivers["visitors_log"])
  drivers["age_norm"] = minmax_norm(drivers["age"])

  base = W_AG * drivers["age_norm"] + W_AV * drivers["av_norm"]
  score = (base * 100).round().astype(int)

  # output
  created_at = iso_created_at()
  out = pd.DataFrame({
    "museum_id": drivers["museum_id"],
    "museum_score": score,
    "drivers_json": drivers.apply(lambda r: {
      "age": int(r["age"]),
      "annual_visitors": int(r["annual_visitors"]),
      "visitors_log": int(r["visitors_log"])
    }, axis=1),
    "created_at": created_at
  })

  out_records = out.to_dict(orient="records")
  with open(args.output, "w", encoding="utf-8") as f:
    json.dump(out_records,f,ensure_ascii=False,indent=2)

if __name__ == "__main__":
  main()
