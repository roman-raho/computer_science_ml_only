import json
import argparse
import sys
from typing import List, Iterable
import pandas as pd
from utils import iso_created_at, minmax_norm

W_TC = 0.4

def parse_args(argv: List[str] = None):
  p = argparse.ArgumentParser(description="Compute provenance_score from raw provenance JSON")
  p.add_argument("--input", required=True, help="Path to raw gallery JSON (array of objects)")
  p.add_argument("--output", required=True,help="Path to write processed gallery_score JSON.")
  p.add_argument("--top_15_cities",required=True,help="Path to top 15 art market cities JSON.")
  return p.parse_args(argv)

def load_top_cities(path: str) -> Iterable[str]:
  try:
    data = pd.read_json(path)
    if isinstance(data, pd.DataFrame):
      if data.shape[1] == 1:
        vals = data.iloc[:, 0].astype(str).tolist()
      else:
        vals = pd.unique(data.astype(str).values.ravel("K").tolist())
    else:
      vals = pd.Series(data).astype(str).tolist()
  except ValueError:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        vals = list(map(str, raw.values()))
    else:
        vals = list(map(str, raw))
  return {v.strip().lower() for v in vals}


def compute_flags(df: pd.DataFrame, top_cities: Iterable[str]) -> pd.DataFrame:
  location_norm = df["location"].fillna("").str.strip().str.lower()

  cities_flag = location_norm.apply(lambda x: 1 if x in top_cities else 0)

  return pd.DataFrame({
    "cities_flag": cities_flag,
  })

def main(argv: List[str] = None):
  args = parse_args(argv)

  try:
    raw = pd.read_json(args.input)
  except ValueError as e:
    print(f"Error: Failed to read JSON file from {args.input}: {e}", file=sys.stderr) # else print error to the console in format standard erro
    sys.exit(1)

  top_cities = load_top_cities(args.top_15_cities)

  required_cols = {"gallery_id", "location", "number_of_artists"}
  missing = required_cols - set(raw.columns)

  if missing:
    print(f"Error: Missing required columns in input {sorted(missing)}", file=sys.stderr) 
    sys.exit(1)
  
  flags = compute_flags(raw, top_cities)
  raw = pd.concat([raw,flags], axis=1)

  grp = raw.groupby("gallery_id", dropna=False)

  number_of_artists=grp["number_of_artists"].max().rename("number_of_artists")
  top_city = grp["cities_flag"].max().rename("cities_flag")

  drivers = pd.concat([number_of_artists, top_city], axis=1).reset_index()

  drivers["ta_norm"] = minmax_norm(drivers["number_of_artists"])

  base = drivers["ta_norm"]
  bonus = W_TC * drivers["cities_flag"]
  final = (base+bonus).clip(upper=1)
  score = (final*100).round().astype(int)

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

  print(f"Wrote {len(out_records)} rows -> {args.output}")
  print(f"Score summary: min={int(score.min())}, mean={round(float(score.mean()),1)}, max={int(score.max())}")

if __name__ == "__main__": # run code
  main()