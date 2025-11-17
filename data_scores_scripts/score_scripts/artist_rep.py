import json
import argparse
import sys
from typing import List
import pandas as pd
from utils import iso_created_at, minmax_norm
import numpy as np

# define set weights at the top so that code is easier to read

W_V = 0.4
W_P = 0.25
W_NA = 0.2
W_S = 0.1
W_A = 0.05

def parse_args(argv: List[str] = None): # setting up the cmd line interface the run the python file
  p = argparse.ArgumentParser()
  p.add_argument("--input_artist", required=True)
  p.add_argument("--input_artwork", required=True)
  p.add_argument("--input_provenance", required=True)
  p.add_argument("--output", required=True)
  p.add_argument("--current_year",type=int, default=2025)
  return p.parse_args(argv)

def main(argv: List[str] = None):
  args = parse_args(argv)

  # try to read files
  try:
    raw_artist = pd.read_json(args.input_artist)
    raw_artwork = pd.read_json(args.input_artwork)
    raw_provenance = pd.read_json(args.input_provenance)
  except ValueError as e:
    print(f"Couldnt read JSON file from: {args.input}: {e}", file=sys.stderr) # else print error to the console in format standard erro
    sys.exit(1)

  # check if all required columns are there
  required_artist_cols = {"artist_id", "name", "nationality"}
  required_artwork_cols = {"artwork_id","artist_id","insurance_value","signed","year_created","artwork_length","artwork_width","artwork_height"}
  required_provenance_cols = {"artwork_id", "provenance_score"}

  missing_artist = required_artist_cols - set(raw_artist.columns)
  missing_artwork = required_artwork_cols - set(raw_artwork.columns)
  missing_provenance = required_provenance_cols - set(raw_provenance.columns)

  if missing_artist or missing_artwork or missing_provenance:
    sys.exit(1)
  
  # some data cleaning
  grp_artworks = raw_artwork.groupby("artist_id",dropna=False)
  avg_insurance_value = grp_artworks["insurance_value"].mean().rename("avg_insurance_value")
  avg_insurance_value_log1p = np.log1p(avg_insurance_value.fillna(0)).rename("avg_insurance_value_log1p")
  signed_ratio = grp_artworks["signed"].mean().rename("signed_ratio").fillna(0.0)
  
  avg_artwork_age = (
    (args.current_year - raw_artwork["year_created"])
    .groupby(raw_artwork["artist_id"])
    .mean()
    .rename("avg_artwork_age")
    .fillna(0)
  )

  prov_join = raw_provenance[["artwork_id","provenance_score"]].merge(
    raw_artwork[["artwork_id","artist_id"]], on="artwork_id", how="inner"
  )

  avg_provenance_score = (
    prov_join.groupby("artist_id")["provenance_score"].mean()
    .rename("avg_provenance_score")
    .fillna(0)
  ) 

  num_artworks = grp_artworks.size().rename("num_artworks").astype(int)

  artists = raw_artist.set_index("artist_id")[["name", "nationality"]]
  drivers = artists.join(
    [
      num_artworks,
      avg_insurance_value,
      avg_insurance_value_log1p,      
      signed_ratio,                              
      avg_artwork_age,        
      avg_provenance_score,
    ],
    how="outer"
  ).reset_index()  # keeps "artist_id"

  # assign drivers
  drivers["num_artworks"] = drivers.get("num_artworks", pd.Series(index=drivers.index)).fillna(0).astype(int)
  drivers["avg_insurance_value"] = drivers.get("avg_insurance_value", pd.Series(index=drivers.index)).fillna(0.0).astype(float)
  drivers["avg_insurance_value_log1p"] = drivers.get("avg_insurance_value_log1p", pd.Series(index=drivers.index)).fillna(0.0).astype(float)
  drivers["signed_ratio"] = drivers.get("signed_ratio", pd.Series(index=drivers.index)).fillna(0.0).astype(float)
  drivers["avg_artwork_age"] = drivers.get("avg_artwork_age", pd.Series(index=drivers.index)).fillna(0.0).astype(float)
  drivers["avg_provenance_score"] = drivers.get("avg_provenance_score", pd.Series(index=drivers.index)).fillna(0.0).astype(float)
  drivers["num_artworks"] = drivers["num_artworks"].fillna(0).astype(int)
  drivers["cnt_norm"] = minmax_norm(drivers["num_artworks"])
  drivers["val_norm"] = minmax_norm(drivers["avg_insurance_value_log1p"])
  drivers["sig_norm"] = minmax_norm(drivers["signed_ratio"])
  drivers["age_norm"] = minmax_norm(drivers["avg_artwork_age"])
  drivers["prov_norm"] = minmax_norm(drivers["avg_provenance_score"])

  # calculate score from defined formula from design
  score = round(100 * (W_V * drivers["val_norm"] + W_P*drivers["prov_norm"] + W_NA*drivers["cnt_norm"] + W_S*drivers["sig_norm"] + W_A*drivers["age_norm"]))

  created_at = iso_created_at()
  # prepare output
  out = pd.DataFrame({
    "artist_id": drivers["artist_id"],
    "reputation_score": score,
    "drivers_json": drivers.apply(lambda r: {
      "name": str(r["name"]),
      "nationality": str(r["nationality"]),
      "avg_insurance_value": int(r["avg_insurance_value"]),
      "signed_ratio": float(r["signed_ratio"]),
      "avg_artwork_age": int(r["avg_artwork_age"]),
      "avg_provenance_score": int(r["avg_provenance_score"]),
      "num_artworks": int(r["num_artworks"]),
    }, axis=1),
    "created_at": created_at
  })

  out_records = out.to_dict(orient="records")
  with open(args.output, "w",encoding = "utf-8") as f:
    json.dump(out_records, f, ensure_ascii=False, indent=2) # store in file path provided

if __name__ == "__main__": # run code
  main()
 