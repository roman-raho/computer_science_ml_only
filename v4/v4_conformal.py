from __future__ import annotations
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# utility functions

def _log1p_safe(a):
  a = np.asarray(a, dtype=float)
  a = np.clip(a, 0.0, None)
  return np.log1p(a)

def _expm1(a):
  return np.expm1(a)

def _abs_log_residuals(y_true, y_pred):
  y_true = np.asarray(y_true, dtype=float)
  y_pred = np.asarray(y_pred, dtype=float)
  y_pred = np.clip(y_pred, 0.0, None)
  return np.abs(_log1p_safe(y_true) - _log1p_safe(y_pred))

def _coverage(y_true, lo, hi):
  y_true = np.asarray(y_true, dtype=float)
  return float(np.mean((y_true >= lo) & (y_true <= hi)))

# split data

def _load_fm_and_split(csv_path: Path, target_col: str, val_size: float, random_state: int):
  df = pd.read_csv(csv_path)
  if target_col not in df.columns:
      raise ValueError(f"Target col '{target_col}' not found in {csv_path}. Columns: {df.columns.tolist()}")
  y = df[target_col].values
  X = df.drop(columns=[target_col])
  X = X.select_dtypes(include=[np.number]).fillna(0.0).values
  _, X_val, _, y_val = train_test_split(
      X, y, test_size=val_size, random_state=random_state
  )
  return X_val, y_val

# predict models 
def _fit_conformal_q(model_path: Path, fm_csv: Path, target_col: str,
                     val_size: float, random_state: int, alpha: float):
  X_val, y_val = _load_fm_and_split(fm_csv, target_col, val_size, random_state)
  model = joblib.load(model_path)
  mu_val = model.predict(X_val)
  res = _abs_log_residuals(y_val, mu_val)
  q = float(np.quantile(res, alpha)) # get number within alpha
  lo = _expm1(np.maximum(0.0, _log1p_safe(mu_val) - q))
  hi = _expm1(_log1p_safe(mu_val) + q)
  cov = _coverage(y_val, lo, hi)
  return q, cov # return the error radius in log space + the actual coverage on validation 

def main():
  parser = argparse.ArgumentParser(description="Build conformal bands (log-space) for LR and RF models")
  parser.add_argument("--lr_model", required=True, type=Path, help="Path to lr .joblib")
  parser.add_argument("--lr_fm", required=True, type=Path, help="Path to lr feature mart CSV")
  parser.add_argument("--rf_model", required=True, type=Path, help="Path to rf .joblib")
  parser.add_argument("--rf_fm", required=True, type=Path, help="Path to rf feature mart CSV")
  parser.add_argument("--target_col", default="price", help="name of target column default - price")
  parser.add_argument("--val_size", type=float, default=0.2, help="Validation size fraction = 0.2")
  parser.add_argument("--alpha", type=float, default=0.90, help="qantile level = 0.90")
  parser.add_argument("--random_state", type=int, default=42, help="Random seed - 42")
  parser.add_argument("--out_json", type=Path, default=Path("v4_conformal.json"),
                        help="Output JSON file (default = ./v4_conformal.json)")
  args = parser.parse_args()

  # train models
  q_lr, cov_lr = _fit_conformal_q(args.lr_model, args.lr_fm, args.target_col,
                                    args.val_size, args.random_state, args.alpha)
  q_rf, cov_rf = _fit_conformal_q(args.rf_model, args.rf_fm, args.target_col,
                                    args.val_size, args.random_state, args.alpha)
  
  # return payload
  payload = {
    "alpha": args.alpha,
    "target": "log1p",
    "q_lr": q_lr,
    "cov_lr": cov_lr,
    "q_rf": q_rf,
    "cov_rf": cov_rf,
    "meta": {
      "val_size": args.val_size,
      "random_state": args.random_state,
      "lr_fm_csv": str(args.lr_fm),
      "rf_fm_csv": str(args.rf_fm),
      "lr_model": str(args.lr_model),
      "rf_model": str(args.rf_model),
    }
  }
  
  with open(args.out_json, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

  print(f"saved -> {args.out_json}")

if __name__ == "__main__":
    main()