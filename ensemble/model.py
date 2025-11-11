import pandas as pd
import numpy as np
import argparse
import json, os
from typing import List, Optional
from joblib import dump

from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold


def parse_args(argv: Optional[List[str]]):
  p = argparse.ArgumentParser()

  p.add_argument("--training_data_prefix", required=True)
  p.add_argument("--testing_data_prefix", required=True)
  p.add_argument("--output_prefix", default="ensemble/model_outputs")

  return p.parse_args(argv)

def train_model(X_train: pd.DataFrame, 
                X_test: pd.DataFrame, 
                y_train: pd.DataFrame, 
                y_test: pd.DataFrame,
                output_prefix: str):
  
  y_train = np.ravel(y_train.squeeze()) # turn the y into a Series
  y_test  = np.ravel(y_test.squeeze())  

  if any(len(obj) == 0 for obj in (X_train, X_test, y_train, y_test)): # check if empty
    raise ValueError("You must provide non-empty datasets.")
  
  model = ElasticNet(max_iter=10000, fit_intercept=True, random_state=42) # use an ElasticNet model

  param_grid = { 
    "alpha": np.logspace(-3, 1.5, 12),
    "l1_ratio": [0.05, 0.2, 0.5, 0.8, 0.95],
  }
  
  inner = KFold(n_splits=5, shuffle=True, random_state=42) # divide training into equal parts

  gscv = GridSearchCV( # form with cross validation
    estimator=model,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=inner,
    n_jobs=-1,
    refit=True,
    verbose=0
  )

  gscv.fit(X_train, y_train) # fit the training data

  yhat_test = gscv.predict(X_test)

  price_true = np.expm1(y_test) # get the actual training data
  price_pred = np.expm1(yhat_test)

  abs_resid_log = np.abs(y_test - yhat_test)

  alpha = 0.2 # 80% intervals
  q_hat_log = float(np.quantile(abs_resid_log, 1 - alpha))

  pct_err = np.abs(price_pred - price_true) / np.maximum(price_true, 1e-6)
  within_20 = float((pct_err <= 0.20).mean())

  mae = float(mean_absolute_error(y_test, yhat_test)) # metrics
  r2 = float(r2_score(y_test, yhat_test))

  rng = np.random.default_rng(42)
  y_shuf = y_train.copy()
  rng.shuffle(y_shuf) # shuffle for dummy results

  gscv_shuf = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=inner,
    n_jobs=-1,
    refit=True,
    verbose=0
  )

  gscv_shuf.fit(X_train, y_shuf) # fit to data
  yhat_shuf = gscv_shuf.predict(X_test)
  r2_shuf = float(r2_score(y_test, yhat_shuf))

  return { # return metrics and model
    "best_params": getattr(gscv, "best_params_", None),
    "mae_log": mae,
    "r2_log": r2,
    "pct_within_20pct_price": within_20,
    "r2_shuffle_guard": r2_shuf,
    "best_estimator_": gscv.best_estimator_,
    "search": gscv,
    "y_test": y_test,
    "yhat_test": yhat_test,
    "X_test_cols": X_test.columns,
    "alpha": alpha,
    "q_hat_log": q_hat_log,
    "abs_resid_log": abs_resid_log 
  }

def save_model_artifacts(
    output_prefix: str,
    search,
    X_test_cols,
    y_test: np.ndarray,
    yhat_test: np.ndarray,
    within_20: float,
    mae_log: float,
    r2_log: float,
    alpha: float,
    q_hat_log: float,
    abs_resid_log: float,
    r2_shuffle_guard: float,
  ):
  out_dir = os.path.dirname(output_prefix)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)

  # model save
  dump(getattr(search, "best_estimator_", search),
    f"{output_prefix}_model.joblib")
  
  # predictions
  pd.DataFrame({"y_true_log": y_test, "y_pred_log": yhat_test}).to_csv(
    f"{output_prefix}_test_predictions.csv", index=False
  )

  metrics = {
    "best_params": getattr(search, "best_params_", None),
    "cv_best_mae_log": float(-search.best_score_) if hasattr(search, "best_score_") else None,
    "test_mae_log": float(mae_log),
    "test_r2_log": float(r2_log),
    "pct_within_20pct_price": float(within_20),
    "r2_shuffle_guard": float(r2_shuffle_guard),
    "n_splits": getattr(search.cv, "n_splits", None),
    "interval_alpha": float(alpha),
    "q_hat_log": float(q_hat_log),
    "abs_resid_log_p80": float(np.quantile(abs_resid_log, 0.80)),
    "abs_resid_log_mean": float(np.mean(abs_resid_log)),
  }

  with open(f"{output_prefix}_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

  # cv grid results
  if hasattr(search, "cv_results_"):
    pd.DataFrame(search.cv_results_).to_csv(
      f"{output_prefix}_cv_results.csv", index=False
    )

  with open(f"{output_prefix}_feature_columns.json", "w", encoding="utf-8") as f:
    json.dump({"columns": list(X_test_cols)}, f, indent=2)

def main(argv: Optional[List[str]] = None):
  argv = parse_args(argv)
  
  X_train = pd.read_parquet(f"{argv.training_data_prefix}/X_train.parquet")
  X_test = pd.read_parquet(f"{argv.testing_data_prefix}/X_test.parquet")
  y_train = pd.read_parquet(f"{argv.training_data_prefix}/y_train.parquet")
  y_test = pd.read_parquet(f"{argv.testing_data_prefix}/y_test.parquet")

  out = train_model(X_train, X_test, y_train, y_test, argv.output_prefix)

  save_model_artifacts(
    output_prefix=argv.output_prefix,
    search=out["search"],
    X_test_cols=out["X_test_cols"],
    y_test=out["y_test"],
    yhat_test=out["yhat_test"],
    within_20=out["pct_within_20pct_price"],
    mae_log=out["mae_log"],
    r2_log=out["r2_log"],
    alpha = out["alpha"],
    q_hat_log = out["q_hat_log"],
    abs_resid_log=out["abs_resid_log"],
    r2_shuffle_guard=out["r2_shuffle_guard"],
  )

if __name__ == "__main__":
  main()