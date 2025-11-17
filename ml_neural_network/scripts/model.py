import json, argparse, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks

def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument("--features", required=True)
  p.add_argument("--structured", required=True)
  p.add_argument("--target", default="final_price")
  p.add_argument("--outdir", required=True)
  p.add_argument("--seed", type=int, default=42)
  return p.parse_args()

def pct_within_20(true, pred):
  true = np.asarray(true); pred = np.asarray(pred)
  denom = np.maximum(np.abs(true), 1e-6)
  pct_err = np.abs(pred - true) / denom # compute the percentage error
  return float((pct_err <= 0.20).mean())

def main():
  args = parse_args()
  tf.random.set_seed(args.seed)

  # create the output directory
  outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

  X = pd.read_parquet(args.features)

  # bring in target from structured JSON
  structured = pd.read_json(args.structured)

  # expect these ID keys 
  required_keys = ["artwork_id", "artist_id"]
  if not all(k in structured.columns for k in required_keys):
    raise ValueError(f"structured.json must contain {required_keys}")
  if not all(k in X.columns for k in required_keys):
    raise ValueError(f"features.parquet must contain {required_keys}")
  

  df = X.merge(structured[required_keys + [args.target]], on=required_keys, how="inner")
  if df[args.target].isna().any(): # remove any null prices
    df = df.dropna(subset=[args.target])

  # features
  drop_cols = required_keys + [args.target]
  feature_cols = [c for c in df.columns if c not in drop_cols]
  X_mat = df[feature_cols].astype("float32").values
  y = df[args.target].astype("float32").values

  # predict log price to stablise
  y = np.log1p(y)

  X_train, X_val, y_train, y_val = train_test_split(
    X_mat, y, test_size=0.2, random_state=args.seed
  )

  # small feed forward model
  model = keras.Sequential([
    layers.Input(shape=(X_mat.shape[1],)),
    layers.Dense(64,activation="relu", kernel_regularizer=regularizers.l2(1e-4)), # 64 neuron value - each neuron applies the ReLU function - adds a small penatly term to the weights
    layers.Dropout(0.15), # regularisation - 15% of the neuros during training in the previous layer are randomly dropped at each step
    layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-4)), # 32 neurons, again using ReLU and l2 reg
    layers.Dense(1, activation="linear") # output layer
  ])

  model.compile(optimizer="adam", loss="mse", metrics=["mae"])

  es = callbacks.EarlyStopping(monitor="val_mae", patience=15, restore_best_weights=True) # prevent overfitting by stopping training if weights dont improve outcome

  hist = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=300,
    batch_size=32,
    verbose=0, # silence
    callbacks=[es] # early stopping
  )

  # evaluate on holdout
  y_pred_val = model.predict(X_val, verbose=0).reshape(-1)
  y_val_real = np.expm1(y_val)
  y_pred_real = np.expm1(y_pred_val)

  mae = float(mean_absolute_error(y_val_real, y_pred_real))
  r2 = float(r2_score(y_val_real, y_pred_real))
  within_20 = pct_within_20(y_val_real, y_pred_real)

  # save artificats
  model_path = outdir / "nn_model.keras"
  model.save(model_path)

  hist_path = outdir / "nn_history.json"
  with open(hist_path, "w") as f:
    json.dump({k: [float(x) for x in v] for k, v in hist.history.items()}, f, indent=2)

  report = {
    "seed": args.seed,
    "n_features": int(X_mat.shape[1]),
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "metrics": {
            "val_mae_price": mae,
            "val_r2_price": r2,
            "pct_within_20pct_price": within_20
        },
        "feature_cols_sample": feature_cols[:10],
        "model_hash_sha1": hashlib.sha1(model_path.read_bytes()).hexdigest()
  }

  # save to csv artist
  y_pred_all_log = model.predict(X_mat, verbose=0).reshape(-1)

  pred = pd.DataFrame({
    "artwork_id": df["artwork_id"].astype(str),
    "y_pred_log": y_pred_all_log
  })

  pred = pred.groupby("artwork_id", as_index=False)["y_pred_log"].mean()

  pred_path = outdir / "qual_preds.csv"
  pred.to_csv(pred_path, index=False)

  with open (outdir / "nn_report.json", "w") as f:
    json.dump(report, f, indent=2)

if __name__ == "__main__":
  main()

