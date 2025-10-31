import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import hashlib
import argparse

def parse_args():
  parser = argparse.ArgumentParser(description="Build qualitative feature mart.")
  parser.add_argument("--structured", required=True, help="Path to structured JSON file")
  parser.add_argument("--text", required=True, help="Path to text JSON file")
  parser.add_argument("--output", required=True, help="Where to save the feature matrix (.parquet)")
  parser.add_argument("--artifacts_dir", required=True, help="Where to store TF-IDF/PCA/Scaler artifacts")
  return parser.parse_args()


def build_feature_mart(structured_path, text_path, output_path, artifacts_dir):
  np.random.seed(42) # generated random seed
  artifacts_dir.mkdir(parents=True, exist_ok=True)

  # load the data
  structured = pd.read_json(structured_path)
  text = pd.read_json(text_path)

  # merge data
  df = pd.merge(structured, text, on=["artwork_id", "artist_id"], how="inner").reset_index(drop=True)
  
  # UPDATE - to prevent leakages
  artists = df["artist_id"].dropna().unique()
  rng = np.random.default_rng(42)
  rng.shuffle(artists)
  cut = int(0.8 * len(artists)) if len(artists) > 1 else 1
  train_artists = set(artists[:cut])
  is_train = df["artist_id"].isin(train_artists)

  df_train = df[is_train].copy()
  df_val = df[~is_train].copy()


  # OLD CODE
  # tf-idf vectorisastion of text
  # tfidf = TfidfVectorizer(max_features=300, stop_words="english")
  # text_vectors = tfidf.fit_transform(df["text"]).toarray()

  # UPDATED CODE
  tfidf = TfidfVectorizer(max_features=300, stop_words="english")
  text_vectors_train = tfidf.fit_transform(df_train["text"]).toarray()
  text_vectors_val = tfidf.transform(df_val["text"]).toarray()

  # pca for dimension reducation
  pca = PCA(n_components=24, random_state=42)
 
  # OLD CODE
  # text_reduced = pca.fit_transform(text_vectors)
  # text_cols = [f"text_{i}" for i in range(text_reduced.shape[1])]
  # text_features = pd.DataFrame(text_reduced, columns=text_cols)

  # UPDATED CODE
  text_reduced_train = pca.fit_transform(text_vectors_train)
  text_reduced_val = pca.transform(text_vectors_val)
  text_cols = [f"text_{i}" for i in range(text_reduced_train.shape[1])]
  text_train = pd.DataFrame(text_reduced_train, columns = text_cols, index=df_train.index)
  text_val = pd.DataFrame(text_reduced_val, columns=text_cols, index=df_val.index)

  # scale structured numeric features
  numeric_cols = ["press_index", "institution_score", "gallery_tier_max", "exhibition_count"]
  scaler = StandardScaler()

  # UPDATED CODE
  struct_scaled_train = scaler.fit_transform(df_train[numeric_cols])
  struct_scaled_val = scaler.transform(df_val[numeric_cols])
  struct_cols = [f"num_{col}" for col in numeric_cols]
  struct_train = pd.DataFrame(struct_scaled_train, columns=struct_cols, index=df_train.index)
  struct_val = pd.DataFrame(struct_scaled_val, columns=struct_cols, index=df_val.index)

  # OLD CODE
  # struct_scaled = scaler.fit_transform(df[numeric_cols])
  # struct_cols = [f"num_{col}" for col in numeric_cols]
  # struct_features = pd.DataFrame(struct_scaled, columns=struct_cols)

  # combine all features
  # id_cols = df[["artwork_id", "artist_id"]].reset_index(drop=True)
  feats_train = pd.concat([df_train[["artwork_id","artist_id"]], struct_train, text_train], axis=1)#
  feats_val = pd.concat([df_val[["artwork_id","artist_id"]], struct_val, text_val], axis=1)
  X = pd.concat([feats_train, feats_val], axis=0).sort_index().reset_index(drop=True)

  # save outputs (with ids)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  X.to_parquet(output_path, index=False)
  joblib.dump(tfidf, artifacts_dir / "tfidf.pkl")
  joblib.dump(pca, artifacts_dir / "pca.pkl")
  joblib.dump(scaler, artifacts_dir / "scaler.pkl")

  metadata = {
    "n_samples": len(X),
    "n_features": X.shape[1],
    "n_text_components": text_reduced_train.shape[1],
    "n_structured": len(numeric_cols),
    "structured_path": str(structured_path),
    "text_path": str(text_path),
    "output_path": str(output_path),
  }

  with open(artifacts_dir / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

if __name__ == "__main__":
  args = parse_args()
  build_feature_mart(
    structured_path=Path(args.structured),
    text_path=Path(args.text),
    output_path=Path(args.output),
    artifacts_dir=Path(args.artifacts_dir)
  )