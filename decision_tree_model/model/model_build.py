import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# sklearn imports
from sklearn import __version__ as skl_version
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from joblib import dump

# import feature mart builder
from feature_builder import build_feature_mart_all

# model training

def train_tree_models(
  df: pd.DataFrame, train_end_year: int, output_prefix: str
) -> Dict:
  train_df = df[df["auction_year"] <= train_end_year].copy()
  test_df = df[df["auction_year"] > train_end_year].copy()

  # if empty
  if len(train_df) == 0 or len(test_df) == 0:
    raise RuntimeError("Train/Test split is empty. Check train_end_year.")
  
  # feature set
  numeric_cols = [
        "log_area", "log_volume", "artwork_age", "signed_flag", "reserve_gt0",
        "log_insurance", "dealer_avg_earnings", "ownership_count",
        "loan_count", "prov_count", "auction_house_rating"
    ]
  
  numeric_cols = [c for c in numeric_cols if c in df.columns]

  # get catogerical columns
  cat_cols = []
  for c in ["medium", "nationality", "season_q", "location"]:
    if c in df.columns:
      cat_cols.append(c)

  # build number and categorical pipeline seperatly
  num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
  ])

  cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
  ])

  # put together
  pre = ColumnTransformer([
    ("num", num_pipe, numeric_cols),
    ("cat", cat_pipe, cat_cols)
  ], remainder="drop", verbose_feature_names_out=False)

  # define models
  models = {
    "decision_tree": DecisionTreeRegressor(random_state=42),
    "random_forest": RandomForestRegressor(random_state=42, n_jobs=-1)
  }

  # grids
  grids = {
    "decision_tree": {"model__max_depth": [5, 10, 20, None]},
    "random_forest": {
      "model__n_estimators": [50, 100, 200],
      "model__max_depth": [10, 20, None],
    }
  }

  results = {}
  inner = TimeSeriesSplit(n_splits=5)

  for name, model in models.items():
    pipe = Pipeline([("pre", pre), ("model", model)])
   
  # full pipeline
  gscv = GridSearchCV(
    estimator=pipe,
    param_grid=grids[name],
    scoring="neg_mean_absolute_error",
    cv=inner,
    n_jobs=-1,
    refit=True
  )

  # generate training and testing
  X_train = train_df[numeric_cols + cat_cols]
  y_train = train_df["y_log_price"].values
  X_test = test_df[numeric_cols + cat_cols]
  y_test = test_df["y_log_price"].values
  gscv.fit(X_train, y_train)
  yhat_test = gscv.predict(X_test)

  # evaluate
  price_true = np.expm1(y_test)
  price_pred = np.expm1(yhat_test)
  pct_err = np.abs(price_pred - price_true) / np.maximum(price_true, 1e-6)
  within_20 = float((pct_err <= 0.20).mean())
  mae = float(mean_absolute_error(y_test, yhat_test))
  r2 = float(r2_score(y_test, yhat_test))

  results[name] = {
    "best_params": gscv.best_params_,
    "test_mae_log": mae,
    "test_r2_log": r2,
    "pct_within_20pct_price": within_20,
  }

  # Saving to CSV
  X_all = df[numeric_cols + cat_cols]
  yhat_all = gscv.predict(X_all)

  prediction = pd.DataFrame({
    "artwork_id": df["artwork_id"].astype(str),
    "y_pred_log": yhat_all
  })

  prediction = prediction.groupby("artwork_id", as_index=False)["y_pred_log"].mean()

  prediction_path = f"{output_prefix}_{name}_predictions.csv"
  prediction.to_csv(prediction_path, index=False)

  # save model + report
  dump(gscv.best_estimator_, f"{output_prefix}_{name}.joblib")
  with open(f"{output_prefix}_{name}_report.json", "w") as f:
    json.dump(results[name], f, indent=2)
  
  return results

# main

def main(argv: Optional[List[str]] = None):
  parser = argparse.ArgumentParser()
  parser.add_argument("--raw-path", required=True)
  parser.add_argument("--current-year", type=int, default=datetime.now().year)
  parser.add_argument("--train-end-year", type=int, required=True)
  parser.add_argument("--output-prefix", required=True)
  args = parser.parse_args(argv)


  try:
    df = build_feature_mart_all(args.raw_path, args.current_year)
  except Exception as e:
    sys.exit(1)

  try:
    df.to_csv(f"{args.output_prefix}_feature_mart.csv", index=False)
  except Exception as e:
    print(f"failed save feature mart: {e}", file=sys.stderr)


if __name__ == "__main__":
  try:
    main()
  except Exception as e:
    print(f"failed: {e}", file=sys.stderr)
    sys.exit(1)
