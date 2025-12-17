from typing import Dict

import numpy as np
import pandas as pd

# determine the log values of a series
def safe_log1p(s: pd.Series) -> pd.Series:
    return np.log1p(pd.to_numeric(s, clip_low=0.0, errors="coerce").fillna(0.0))

def build_feature_mart(
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
        bid[col] = pd.to_numeric(bid[col])  # conver all to actual numbers

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
    df = df[pd.to_numeric(df["final_price"]).notna()].copy()

    # geometry + simple transforms
    df["artwork_length"] = pd.to_numeric(df["artwork_length"])
    df["artwork_width"] = pd.to_numeric(df["artwork_width"])
    df["area"] = df["artwork_length"] * df["artwork_width"]
    df["log_area"] = safe_log1p(df["area"])  # take log of area to stablise scale

    df["year_created"] = pd.to_numeric(df["year_created"]).fillna(current_year)
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

    # price derivd features
    df["reserve_gt0"] = (pd.to_numeric(df["reserve_price"]).fillna(0) > 0).astype(
        int
    )  # get the reserve price correctly formatted

    # compute premium with simple formula for percentage
    with np.errstate(divide="ignore", invalid="ignore"):  # silence runtime errors
        df["premium"] = np.where(
            (pd.to_numeric(df["reserve_price"]).fillna(0) > 0)
            & pd.to_numeric(df["final_price"]).notna(),
            (pd.to_numeric(df["final_price"]) - pd.to_numeric(df["reserve_price"]))
            / np.maximum(pd.to_numeric(df["reserve_price"]).replace(0, np.nan), 1e-12),
            np.nan,
        )
    df["mean_bids"] = pd.to_numeric(df["number_of_bids"])  # already numeric

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
    df["y_price"] = pd.to_numeric(df["final_price"]).fillna(0.0)  # NEW

    df["region"] = (
        df["location"].astype(str).str.split(",").str[-1].str.strip().str.lower()
    )
    
    # conver back to strings - the one hot encode treats these strings as categorical lavels
    df["season_q"] = df["season_q"].astype("Int64").astype(str)

    # drop any rows without date or target
    df = df.dropna(subset=["date_of_auction", "y_log_price"]).copy()

    return df  # return clean dataset
