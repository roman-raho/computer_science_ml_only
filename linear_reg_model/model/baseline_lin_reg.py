from __future__ import annotations
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# sklearn
from sklearn import __version__ as skl_version
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from joblib import dump
from feature_mart import build_feature_mart

# used for the console to output messages
def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[INFO {ts} UTC] {msg}")


# a helper function to read json files - with a try catch block
def read_json(path: str) -> pd.DataFrame:
    try:
        return pd.read_json(path)
    except ValueError: # works if the type doesnt correspon to the value
        try:
            return pd.read_json(path, lines=True) # incase in the format without [] around it
        except Exception as e: # final error handled
            print(f"[ERR] reading {path}: {e}", file=sys.stderr)
            raise

# convers unknown type to numbers (floats or ints)
# if something cannot be converted it is replaced with a NaN
# * is used to show that it can only be called with a keyword
def safe_num(s: pd.Series, *, clip_low: Optional[float] = None) -> pd.Series: # clip low sets the minimum floor on the numbers
    x = pd.to_numeric(s, errors="coerce")
    if clip_low is not None:
        x = x.clip(lower=clip_low)
    return x

# determine the log values of a series
# use log as it is zero safe
def safe_log1p(s: pd.Series) -> pd.Series:
    return np.log1p(safe_num(s, clip_low=0.0).fillna(0.0))

# used to check the version of skicealn
# makes sure user using it has correct verrsion
def sklearn_version_ge(v: str) -> bool:
    def as_tuple(ver: str) -> tuple:
        parts = ver.split("+")[0].split(".")
        return tuple(int(p) for p in parts[:3]) # returns the individual numbers in a tuple
    return as_tuple(skl_version) >= as_tuple(v) # compares them

# depending on the version that the user is using they may need a different type of one hot encoder
# returns the correct value for sparse_output
def make_ohe() -> OneHotEncoder:
    if sklearn_version_ge("1.2.0"):
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# build the model
def nested_elasticnet_report(
    df: pd.DataFrame, train_end_year: int, output_prefix: str # take in the train end year so you can determine the split
) -> Dict:
    train_df = df[df["year"] <= train_end_year].copy() # split the datasets
    test_df = df[df["year"] > train_end_year].copy()

    if len(train_df) == 0 or len(test_df) == 0: # if they are enmpty return
        raise RuntimeError(
            "Train/Test split is empty. Check --train-end-year and your dates and you data."
        )

    # list out the numeric columns
    numeric_cols = [
        "log_area",
        "age",
        "signed_flag",
        "reserve_gt0",
        "artist_rep_score",
        "artist_volatility",
        "prov_score",
        "prov_volatility",
        "auction_house_score",
        "house_volatility",
    ]

    # make sure that each nc is in the dataset
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    # in the same way create the categorical columns list
    cat_cols: List[str] = []
    for c in ["medium", "region", "season_q"]:
        if c in df.columns:
            cat_cols.append(c)

    # 1. fit all nans with median as it is robust to outliars
    # 2. normalise all numeric features
    num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ])

    # fill all missing values with missing
    cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("ohe", make_ohe()), # convert into numeric values
        ])

    pre = ColumnTransformer( # allowing me to apply different preprocessing steps to different subsets of columns in one go
        transformers=[
            ("num", num_pipe, numeric_cols), # apply to the numeric columsn
            ("cat", cat_pipe, cat_cols), # apply to the categorical columsn
        ],
        remainder="drop", # only output the specific columns i tell it to
        verbose_feature_names_out=False, # leave the columns names as i had them
    )

    model = ElasticNet(max_iter=10000, fit_intercept=True, random_state=42) # prepare model

    param_grid = {
        "model__alpha": np.logspace(-3, 1.5, 12),
        "model__l1_ratio": [0.05, 0.2, 0.5, 0.8, 0.95],
    }

    pipe = Pipeline([("pre", pre), ("model", model)]) # prepare pipeline -> process data -> train model

    inner = TimeSeriesSplit(n_splits=5)

    gscv = GridSearchCV( # create the full pipeline
        estimator=pipe, # use the pipeline we defined
        param_grid=param_grid, # use the params we define
        scoring="neg_mean_absolute_error", # metric to determine if model is good
        cv=inner, # time series splt for cross validation
        n_jobs=-1, # run in parallel in all CPU cores
        refit=True, # after CV retrain on all training data with the ebst params
        verbose=0, # 0 = silent
    )

    X_train = train_df[numeric_cols + cat_cols] # get the training data from the correct columns
    y_train = train_df["y_log_price"].values # NEW get the output

    _log("Fitting GridSearchCV (time-aware)") # log to console
    gscv.fit(X_train, y_train) # fit the data

    X_test = test_df[numeric_cols + cat_cols] # prepare testing data
    y_test = test_df["y_log_price"].values # NEW
    yhat_test = gscv.predict(X_test) # test

    price_true = np.expm1(y_test) # undo logs
    price_pred = np.expm1(yhat_test)
    pct_err = np.abs(price_pred - price_true) / np.maximum(price_true, 1e-6) # compute percentage error for each work
    within_20 = float((pct_err <= 0.20).mean()) # returns the amount of predictions by % that are within 20% error

    mae = float(mean_absolute_error(y_test, yhat_test)) # calculate mae and r2
    r2 = float(r2_score(y_test, yhat_test))

    rng = np.random.default_rng(42) # shuffle training targets - features an targets are mismatched
    y_shuf = y_train.copy()
    rng.shuffle(y_shuf)

    gscv_shuf = GridSearchCV( # train model on this nonsense data - model shouldnt learn anything useful
        pipe, param_grid, scoring="neg_mean_absolute_error", cv=inner, n_jobs=-1, refit=True
    )
    gscv_shuf.fit(X_train, y_shuf)
    yhat_shuf = gscv_shuf.predict(X_test)
    r2_shuf = float(r2_score(y_test, yhat_shuf)) # should be close to 0

    # all info about how preprocessing was done
    preproc = gscv.best_estimator_.named_steps["pre"]
    # the learned coef and the inter
    model_best = gscv.best_estimator_.named_steps["model"]

    # generate a data dictionary
    feat_names_num = numeric_cols
    feat_names_cat: List[str] = []
    if cat_cols:
        enc = preproc.named_transformers_["cat"]
        feat_names_cat = list(enc.get_feature_names_out(cat_cols))
    feat_names = feat_names_num + feat_names_cat

    # make table
    coefs = model_best.coef_
    coef_df = pd.DataFrame({"feature": feat_names, "coef": coefs}).sort_values(
        "coef", ascending=False
    )

    # residuals table of the models perforamnce
    resid_df = pd.DataFrame(
        {
            "artwork_id": test_df["artwork_id"].values,
            "auction_id": test_df["auction_id"].values,
            "date_of_auction": test_df["date_of_auction"].astype(str).values,
            "y_true_log": y_test,
            "y_pred_log": yhat_test,
            "price_true": price_true,
            "price_pred": price_pred,
            "pct_error": pct_err,
        }
    ).sort_values("pct_error", ascending=False)

    # save how my numeric features where standardised
    num_pipe = preproc.named_transformers_["num"]
    scaler = num_pipe.named_steps["scaler"] # get the preprocessing pipeline
    std_summary: Dict[str, Dict[str, float]] = {}
    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        for i, col in enumerate(numeric_cols): # chcek if they are there and then add them to dictionary
            std_summary[col] = {
                "mean": float(scaler.mean_[i]),
                "scale": float(scaler.scale_[i]),
            }

    # ensure output folder exists
    out_dir = os.path.dirname(output_prefix)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    coef_df.to_csv(f"{output_prefix}_coefficients.csv", index=False)
    resid_df.to_csv(f"{output_prefix}_residuals.csv", index=False)
    with open(f"{output_prefix}_feature_std.json", "w", encoding="utf-8") as f:
        json.dump(std_summary, f, indent=2)

    # save the fitted pipeline for reuse
    dump(gscv.best_estimator_, f"{output_prefix}_model.joblib")

    report = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_end_year": int(train_df["year"].max()) if len(train_df) else None,
        "best_params": gscv.best_params_,
        "cv_best_mae": float(-gscv.best_score_),
        "test_mae_log": mae,
        "test_r2_log": r2,
        "pct_within_20pct_price": round(within_20, 3),
        "leakage_guard_r2": r2_shuf,
        "sklearn_version": skl_version,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(f"{output_prefix}_baseline_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(
        description="Feature mart + ElasticNet baseline for artwork price prediction."
    )

    # raw
    p.add_argument("--artworks", required=True, help="artworks_v2_large.json")
    p.add_argument("--auctions", required=True, help="auctions_v2.json")
    p.add_argument("--biddata", required=True, help="bid_data_v2.json")
    p.add_argument("--houses", required=True, help="auction_houses_v2.json")

    # processed scores
    p.add_argument("--artist_rep", required=True, help="processed/artist_reputation_score.json")
    p.add_argument("--gallery", required=True, help="processed/gallery_score.json")
    p.add_argument("--museum", required=True, help="processed/museum_score.json")
    p.add_argument("--provenance", required=True, help="processed/provenance_score.json")
    p.add_argument("--house_score", required=True, help="processed/auction_house_score.json")

    # config
    p.add_argument(
        "--current-year",
        type=int,
        default=datetime.now().year,
        help="Reference year (for age calc).",
    )
    p.add_argument(
        "--train-end-year",
        type=int,
        required=True,
        help="All auctions <= this year are train; later years are test.",
    )
    p.add_argument(
        "--output-prefix",
        required=True,
        help="Prefix for outputs (e.g., data/processed/baseline_v2)",
    )

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)

    _log("Loading input tables…")
    artworks = read_json(args.artworks) # get data
    auctions = read_json(args.auctions)
    biddata = read_json(args.biddata)
    houses = read_json(args.houses)

    scores = { # get scores
        "artist_rep": read_json(args.artist_rep),
        "gallery": read_json(args.gallery),
        "museum": read_json(args.museum),
        "provenance": read_json(args.provenance),
        "house_score": read_json(args.house_score),
    }

    _log("Building feature mart") # log to console
    mart = build_feature_mart( 
        artworks, auctions, biddata, houses, scores, args.current_year
    )

    mart_out = f"{args.output_prefix}_feature_mart.csv"
    mart.to_csv(mart_out, index=False)
    _log(f"Wrote feature mart -> {mart_out}")

    _log("Training ElasticNet baseline") # train model
    report = nested_elasticnet_report(mart, args.train_end_year, args.output_prefix)

    print("Baseline report:")
    for k, v in report.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"[ERR] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERR] Unexpected failure: {e}", file=sys.stderr)
        sys.exit(1)
