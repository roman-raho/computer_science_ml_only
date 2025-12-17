import argparse
import json
import os
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

def safe_log1p(s: pd.Series) -> pd.Series:
    return np.log1p(pd.to_numeric(s, clip_low=0.0, errors="coerce").fillna(0.0))

# build the model
def nested_elasticnet_report(
    df: pd.DataFrame, train_end_year: int, output_prefix: str # take in the train end year so you can determine the split
) -> Dict:
    train_df = df[df["year"] <= train_end_year].copy() # split the datasets
    test_df = df[df["year"] > train_end_year].copy()

    if len(train_df) == 0 or len(test_df) == 0: # if they are enmpty return
        raise RuntimeError(
            "train/test data is empty"
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

    cat_cols: List[str] = []
    for c in ["medium", "region", "season_q"]:
        if c in df.columns:
            cat_cols.append(c)

    num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ])

    # fill all missing values with missing
    cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")), # convert into numeric values
        ])

    pre = ColumnTransformer( # allowing me to apply different preprocessing steps to different subsets of columns in one go
        transformers=[
            ("num", num_pipe, numeric_cols), # apply to the numeric columsn
            ("cat", cat_pipe, cat_cols), # apply to the categorical columsn
        ],
        remainder="drop", # only output the specific columns i tell it to
        verbose_feature_names_out=False, # leave the columns names as i had them
    )

    model = ElasticNet(max_iter=10000, fit_intercept=True, random_state=42)

    param_grid = {
        "model__alpha": np.logspace(-3, 1.5, 12),
        "model__l1_ratio": [0.05, 0.2, 0.5, 0.8, 0.95],
    }

    pipe = Pipeline([("pre", pre), ("model", model)]) # prepare pipeline -> process data -> train model

    inner = TimeSeriesSplit(n_splits=5)

    gscv = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=inner,
        n_jobs=-1, # run in parallel in all CPU cores
        refit=True, # after CV retrain on all training data with the ebst params
        verbose=0, # 0 = silent
    )

    X_train = train_df[numeric_cols + cat_cols]
    y_train = train_df["y_log_price"].values

    gscv.fit(X_train, y_train)

    X_test = test_df[numeric_cols + cat_cols]
    y_test = test_df["y_log_price"].values
    yhat_test = gscv.predict(X_test)

    price_true = np.expm1(y_test)
    price_pred = np.expm1(yhat_test)

    # metrics
    pct_err = np.abs(price_pred - price_true) / np.maximum(price_true, 1e-6)
    within_20 = float((pct_err <= 0.20).mean())

    mae = float(mean_absolute_error(y_test, yhat_test))
    r2 = float(r2_score(y_test, yhat_test))

    # save predictions to csv
    X_all = df[numeric_cols + cat_cols]
    yhat_all = gscv.predict(X_all)
    
    predictions = pd.DataFrame({
        "artwork_id": df["artwork_id"].astype(str),
        "y_pred_log": yhat_all
    })

    predictions = predictions.groupby("artwork_id", as_index=False)["y_pred_log"].mean()

    prediction_path = f"{output_prefix}_baseline_prediction.csv"
    predictions.to_csv(prediction_path, index=False)

    rng = np.random.default_rng(42)
    y_shuf = y_train.copy()
    rng.shuffle(y_shuf)

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
    
    # store mean and std for each feature
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
        "sklearn_version": skl_version,
        "created_at": datetime.now(timezone.utc),
    }
    with open(f"{output_prefix}_baseline_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser()

    # raw
    p.add_argument("--artworks", required=True)
    p.add_argument("--auctions", required=True)
    p.add_argument("--biddata", required=True)
    p.add_argument("--houses", required=True)

    # processed scores
    p.add_argument("--artist_rep", required=True)
    p.add_argument("--gallery", required=True)
    p.add_argument("--museum", required=True)
    p.add_argument("--provenance", required=True)
    p.add_argument("--house_score", required=True)

    # config
    p.add_argument(
        "--current-year",
        type=int,
        default=datetime.now().year,
    )
    p.add_argument(
        "--train-end-year",
        type=int,
        required=True,
    )
    p.add_argument(
        "--output-prefix",
        required=True,
    )

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)

    artworks = pd.read_json(args.artworks) # get data
    auctions = pd.read_json(args.auctions)
    biddata = pd.read_json(args.biddata)
    houses = pd.read_json(args.houses)

    scores = { # get scores
        "artist_rep": pd.read_json(args.artist_rep),
        "gallery": pd.read_json(args.gallery),
        "museum": pd.read_json(args.museum),
        "provenance": pd.read_json(args.provenance),
        "house_score": pd.read_json(args.house_score),
    }

    mart = build_feature_mart( 
        artworks, auctions, biddata, houses, scores, args.current_year
    )

    mart_out = f"{args.output_prefix}_feature_mart.csv"
    mart.to_csv(mart_out, index=False)


if __name__ == "__main__":
    main()

