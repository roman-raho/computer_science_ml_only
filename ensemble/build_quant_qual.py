import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np

def main():
  p = argparse.ArgumentParser()
  p.add_argument("--structured_json", required=True)
  p.add_argument("--split_json", required=True)
  p.add_argument("--quant_linear_csv", required=True)
  p.add_argument("--quant_tree_csv", required=True)
  p.add_argument("--qual_path", required=True)
  p.add_argument("--out_dir", required=True)
  args = p.parse_args()

  out = Path(args.out_dir)
  out.mkdir(parents=True, exist_ok=True)
  
  # load the actual true price
  true = pd.read_json(args.structured_json)[["artwork_id", "final_price"]].copy()
  true["artwork_id"] = true["artwork_id"].astype(str)
  true["y_true"] = np.log(true["final_price"].clip(lower=1e-9))
  true = true[["artwork_id", "y_true"]]
  true = true.drop_duplicates(subset=["artwork_id"], keep="first")

  # load the split information
  split = pd.Series(json.load(open(args.split_json)), name="stage")
  split.index = split.index.astype(str)

  # load predictions
  q_lin = pd.read_csv(args.quant_linear_csv).rename(columns={"y_pred_log": "y_quant_linear"})
  q_tree = pd.read_csv(args.quant_tree_csv).rename(columns={"y_pred_log": "y_quant_tree"})

  q_qual = pd.read_csv(args.qual_path)
  q_qual = q_qual.rename(columns={"y_pred": "y_qual", "y_pred_log": "y_qual"})


  # ensure that the IDs are strings
  for df in (q_lin, q_tree, q_qual):
    df["artwork_id"] = df["artwork_id"].astype(str)

  # if there aer duplicates average them
  def avg_duplicates(df, col):
    if df["artwork_id"].duplicated().any():
      return df.groupby("artwork_id", as_index=False)[col].mean()
    else:
      return df
    
  q_lin = avg_duplicates(q_lin, "y_quant_linear")
  q_tree = avg_duplicates(q_tree, "y_quant_tree")
  q_qual = avg_duplicates(q_qual, "y_qual")

  # merge them all
  df = (true
    .merge(q_lin, on="artwork_id", how="inner")
    .merge(q_tree, on="artwork_id", how="inner")
    .merge(q_qual, on="artwork_id", how="inner")
    .merge(split, left_on="artwork_id", right_index=True, how="left"))
  
  df = df.dropna(subset=["y_true","y_quant_linear","y_quant_tree","y_qual","stage"]).copy()


  for stage in ["train", "test"]:
    part = df[df["stage"] == stage]
    X = part[["y_quant_linear", "y_quant_tree", "y_qual"]].reset_index(drop=True)
    y = part[["y_true"]].reset_index(drop=True)
    X.to_parquet(out / f"X_{stage}.parquet", index=False)
    y.to_parquet(out / f"y_{stage}.parquet", index=False)

if __name__ == "__main__":
  main()
