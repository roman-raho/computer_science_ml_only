import json, pandas as pd
from pathlib import Path
import numpy as np
import argparse

if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("--structured_json", required=True)
  p.add_argument("--out_json", required=True)
  p.add_argument("--test_frac", type=float, default=0.2)
  p.add_argument("--seed", type=int, default=42)
  args = p.parse_args()

  df = pd.read_json(args.structured_json)
  if "artwork_id" not in df.columns:
    raise ValueError("Structured JSON must have artwork ID in it")
  
  rng = np.random.default_rng(args.seed)

  # unique artwork IDs only
  new_df = df["artwork_id"].dropna().astype(str).unique()
  rng.shuffle(new_df)

  n_test = int(args.test_frac * len(new_df))
  test_ids = set(new_df[:n_test])

  split = {item: ("test" if item in test_ids else "train") for item in new_df}

  Path(args.out_json).write_text(json.dumps(split, indent=2))