import pandas as pd
from pathlib import Path
import json

RAW = Path("data/raw")

auctions = pd.read_json(RAW / "auctions_v2.json")
bids = pd.read_json(RAW / "bid_data_v2.json")

# merge them by auction id
merged = auctions.merge(
  bids[["auction_id", "final_price"]],
  on="auction_id",
  how="inner"
)

# add artist id
arts = pd.read_json(RAW / "artworks_v2_large.json")[["artwork_id", "artist_id"]]
merged = merged.merge(arts, on="artwork_id", how="left")

target = merged[["artwork_id", "artist_id", "final_price"]].dropna()
target = target[target["final_price"] > 0]

OUT = Path("data/processed/structured_with_target.json")
OUT.parent.mkdir(parents=True, exist_ok=True)
target.to_json(OUT, orient="records", indent=2)
