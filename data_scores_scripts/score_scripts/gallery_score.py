import json
import argparse
import sys
from typing import List, Iterable
import pandas as pd
from utils import iso_created_at, minmax_norm

W_TC = 0.4

def parse_args(argv: List[str] = None):
  p = argparse.ArgumentParser()
  p.add_argument("--input", required=True)
  p.add_argument("--output", required=True)
  p.add_argument("--top_15_cities",required=True)
  return p.parse_args(argv)

def load_top_cities(path: str) -> Iterable[str]:
    
    # read data
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {city.strip().lower() for city in data}


# function to compute if city is top 10
def compute_flags(df: pd.DataFrame, top_cities: Iterable[str]) -> pd.DataFrame:
  location_norm = df["location"].fillna("").str.strip().str.lower()

  cities_flag = location_norm.apply(lambda x: 1 if x in top_cities else 0)

  return pd.DataFrame({
    "cities_flag": cities_flag,
  })

def main(argv: List[str] = None):
  args = parse_args(argv)

  # exception handling
  try:
    raw = pd.read_json(args.input)
  except ValueError as e:
    print(f"couldnt to read JSON file from {args.input}: {e}", file=sys.stderr) # else print error to the console in format standard erro
    sys.exit(1)

  top_cities = load_top_cities(args.top_15_cities)

  required_cols = {"gallery_id", "location", "number_of_artists"}
  missing = required_cols - set(raw.columns)

  if missing:
    print(f"missing required columns in input {sorted(missing)}", file=sys.stderr) 
    sys.exit(1)
  
  # use function defined earlier
  flags = compute_flags(raw, top_cities)
  raw = pd.concat([raw,flags], axis=1)

  # data cleaning
  grp = raw.groupby("gallery_id", dropna=False)

  number_of_artists=grp["number_of_artists"].max().rename("number_of_artists")
  top_city = grp["cities_flag"].max().rename("cities_flag")

  drivers = pd.concat([number_of_artists, top_city], axis=1).reset_index()

  drivers["ta_norm"] = minmax_norm(drivers["number_of_artists"])

  base = drivers["ta_norm"]
  bonus = W_TC * drivers["cities_flag"]
  final = (base+bonus).clip(upper=1)
  score = (final*100).round().astype(int)

  #prep output
  created_at = iso_created_at()
  out = pd.DataFrame({
    "gallery_id": drivers["gallery_id"],
    "gallery_score":score,
    "drivers_json": drivers.apply(lambda r: {
      "number_of_artist": int(r["number_of_artists"]),
      "cities_flag": int(r["cities_flag"]),
    }, axis=1),
    "created_at": created_at
  })

  out_records = out.to_dict(orient="records")
  with open(args.output, "w", encoding="utf-8") as f:
    json.dump(out_records, f, ensure_ascii=False, indent=2)

if __name__ == "__main__": # run code
  main()