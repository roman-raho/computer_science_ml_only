import argparse
import json
import os
import sys
from dataclasses import dataclass
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


# used for the console to output messages
def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[INFO {ts} UTC] {msg}")


# a helper function to read json files - with a try catch block
def read_json(path: str) -> pd.DataFrame:
    try:
        return pd.read_json(path)
    except ValueError:  # works if the type doesnt correspon to the value
        try:
            return pd.read_json(
                path, lines=True
            )  # incase in the format without [] around it
        except Exception as e:  # final error handled
            print(f"[ERR] reading {path}: {e}", file=sys.stderr)
            raise


# convers unknown type to numbers (floats or ints)
# if something cannot be converted it is replaced with a NaN
# * is used to show that it can only be called with a keyword
def safe_num(
    s: pd.Series, *, clip_low: Optional[float] = None
) -> pd.Series:  # clip low sets the minimum floor on the numbers
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
        return tuple(
            int(p) for p in parts[:3]
        )  # returns the individual numbers in a tuple

    return as_tuple(skl_version) >= as_tuple(v)  # compares them


# depending on the version that the user is using they may need a different type of one hot encoder
# returns the correct value for sparse_output
def make_ohe() -> OneHotEncoder:
    if sklearn_version_ge("1.2.0"):
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_feature_mart(  # takes in all tables that we will use features from
    artworks: pd.DataFrame,
    auctions: pd.DataFrame,
    biddata: pd.DataFrame,
    houses: pd.DataFrame,
    scores: Dict[str, pd.DataFrame],
    current_year: int,
) -> pd.DataFrame:  # returns one large dataframe

    need_art = {  # all required columns
        "artwork_id",
        "artist_id",
        "year_created",
        "medium",
        "signed",
        "artwork_length",
        "artwork_width",
    }
    need_auc = {"auction_id", "artwork_id", "auction_house_id", "date_of_auction"}
    need_bid = {"auction_id", "reserve_price", "final_price", "number_of_bids"}

    miss = need_art - set(
        artworks.columns
    )  # check if any columns are missing on tables we are checking
    if miss:  # if miss has values and is true log to the console
        raise AssertionError(f"artworks missing: {sorted(miss)}")
    miss = need_auc - set(auctions.columns)
    if miss:
        raise AssertionError(f"auctions missing: {sorted(miss)}")
    miss = need_bid - set(biddata.columns)
    if miss:
        raise AssertionError(f"bid_data missing: {sorted(miss)}")

    auc = (
        auctions.copy()
    )  # make the copy so dont mutate the original auctions reference
    auc["date_of_auction"] = pd.to_datetime(
        auc["date_of_auction"], errors="coerce"
    )  # change to correct format

    bid = biddata.copy()
    for col in ["reserve_price", "final_price", "number_of_bids"]:
        bid[col] = safe_num(bid[col])  # conver all to actual numbers

    # core merge - auction rows with realised price
    df = (
        auc.merge(
            bid[["auction_id", "reserve_price", "final_price", "number_of_bids"]],
            on="auction_id",
            how="left",
        )
        .merge(artworks, on="artwork_id", how="left")
        .merge(
            houses[["auction_house_id", "location"]], on="auction_house_id", how="left"
        )
    )

    # keep only rows with a valid final price
    df = df[safe_num(df["final_price"]).notna()].copy()

    # geometry + simple transforms
    df["artwork_length"] = safe_num(df["artwork_length"])
    df["artwork_width"] = safe_num(df["artwork_width"])
    df["area"] = df["artwork_length"] * df["artwork_width"]
    df["log_area"] = safe_log1p(df["area"])  # take log of area to stablise scale

    df["year_created"] = safe_num(df["year_created"]).fillna(current_year)
    df["age"] = (current_year - df["year_created"]).clip(lower=0)

    # cleanup the text
    df["medium"] = df["medium"].astype(str).str.lower().str.strip()

    # signed flag
    df["signed_flag"] = (
        df["signed"]
        .astype(str)
        .str.lower()
        .isin(["1", "true", "yes", "y", "t"])
        .astype(int)
    )

    # calendar features
    df["season_q"] = df["date_of_auction"].dt.quarter
    df["year"] = df["date_of_auction"].dt.year

    # price-derived features
    df["reserve_gt0"] = (safe_num(df["reserve_price"]).fillna(0) > 0).astype(
        int
    )  # get the reserve price correctly formatted

    # compute premium with simple formula for percentage
    with np.errstate(divide="ignore", invalid="ignore"):  # silence runtime errors
        df["premium"] = np.where(
            (safe_num(df["reserve_price"]).fillna(0) > 0)
            & safe_num(df["final_price"]).notna(),
            (safe_num(df["final_price"]) - safe_num(df["reserve_price"]))
            / np.maximum(safe_num(df["reserve_price"]).replace(0, np.nan), 1e-12),
            np.nan,
        )
    df["mean_bids"] = safe_num(df["number_of_bids"])  # already numeric

    # join scores getting ready
    art_rep = (
        scores["artist_rep"][
            ["artist_id", "reputation_score", "volatility_score"]
        ].rename(
            columns={
                "reputation_score": "artist_rep_score",  # rename for clarity
                "volatility_score": "artist_volatility",
            }
        )
        if "artist_rep" in scores
        and not scores[
            "artist_rep"
        ].empty  # if the dictionary actually contains the artist rep scores go ahead
        else pd.DataFrame(
            columns=["artist_id", "artist_rep_score", "artist_volatility"]
        )  # else return an empty df
    )

    prov = (
        scores["provenance"][["artwork_id", "prov_score", "volatility_score"]].rename(
            columns={"volatility_score": "prov_volatility"}
        )
        if "provenance" in scores and not scores["provenance"].empty
        else pd.DataFrame(columns=["artwork_id", "prov_score", "prov_volatility"])
    )

    ah = (
        scores["house_score"][
            ["auction_house_id", "auction_house_score", "volatility_score"]
        ].rename(columns={"volatility_score": "house_volatility"})
        if "house_score" in scores and not scores["house_score"].empty
        else pd.DataFrame(
            columns=["auction_house_id", "auction_house_score", "house_volatility"]
        )
    )

    # merge the scores onto the end of the df
    df = df.merge(art_rep, on="artist_id", how="left")
    df = df.merge(prov, on="artwork_id", how="left")
    df = df.merge(ah, on="auction_house_id", how="left")

    # target
    df["y_log_price"] = safe_log1p(df["final_price"])  # log1p to include possible zeros
    df["y_price"] = safe_num(df["final_price"]).fillna(0.0)  # NEW

    # simple region from location , last value
    df["region"] = (
        df["location"].astype(str).str.split(",").str[-1].str.strip().str.lower()
    )

    # categorical casting for encoder
    # uses pandas nullable integer type
    # conver back to strings - the one hot encode treats these strings as categorical lavels
    df["season_q"] = df["season_q"].astype("Int64").astype(str)

    # drop any rows without date or target
    df = df.dropna(subset=["date_of_auction", "y_log_price"]).copy()

    _log(f"Feature mart shape: {df.shape}; columns: {list(df.columns)}")
    return df  # return clean dataset
