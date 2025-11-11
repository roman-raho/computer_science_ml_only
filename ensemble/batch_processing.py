import os, json, time
import numpy as np
import pandas as pd
from joblib import load
from datetime import datetime, timezone
from supabase import create_client, Client
import argparse
from typing import Optional, List

def parse_args(argv: Optional[List[str]]):
  p = argparse.ArgumentParser()
  p.add_argument("--model_prefix", required=True)
  p.add_argument("--feature_path", required=True)

  return p.parse_args(argv)

def main(argv: Optional[List[str]] = None):
  argv = parse_args(argv)

  # supabase setup
  url = "https://xotakuqckqcikadxnpsl.supabase.co"
  key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhvdGFrdXFja3FjaWthZHhucHNsIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NzIzNjk0OSwiZXhwIjoyMDcyODEyOTQ5fQ.V6NsstER0GnVU7jUEhJ5QBBk8RaNrZM8580ygRoRW5o"
  supabase: Client = create_client(url, key)


  model = load(f"{argv.model_prefix}/model_outputs_model.joblib")

  with open(f"{argv.model_prefix}/model_outputs_feature_columns.json", "r", encoding="utf-8") as f:
    cols = json.load(f)["columns"]

  with open(f"{argv.model_prefix}/model_outputs_metrics.json", "r", encoding="utf-8") as f:
    m = json.load(f)

  q_hat_log = float(m["q_hat_log"])
  alpha = float(m["interval_alpha"])

  # load the feature data
  df1 = pd.read_parquet(f"{argv.feature_path}/X_train.parquet")
  df2 = pd.read_parquet(f"{argv.feature_path}/X_test.parquet")

  df = pd.concat([df1, df2], ignore_index=True)
  df = df.reset_index()
  df = df.rename(columns={"index": "artwork_id"})
  df["artwork_id"] = df["artwork_id"].apply(lambda x: f"ARTW-{x:06d}")
  assert "artwork_id" in df.columns, "feature table needs to include artwork_id"

  for c in cols:
    if c not in df.columns:
      df[c] = 0.0
  X = df[cols].to_numpy()

  # predict + intervals
  y_pred_log = model.predict(X)
  lower_log = y_pred_log - q_hat_log
  upper_log = y_pred_log + q_hat_log

  value = np.expm1(y_pred_log)
  lower = np.expm1(lower_log)
  upper = np.expm1(upper_log)
  confidence = np.full_like(value, fill_value=(1-alpha), dtype=float)

  # build upload rows
  fetched_at = datetime.now(timezone.utc).isoformat()
  rows = []
  for aid, v, lo, hi, conf in zip(df["artwork_id"], value, lower, upper, confidence):
    rows.append({
      "artwork_id": str(aid),
      "ensemble_value": float(v),
      "lower": float(lo),
      "upper": float(hi),
      "confidence": float(conf),
      "fetched_at": fetched_at
    })

  # upload in batches
  start = time.time()
  batch_size = 500
  for i in range(0, len(rows), batch_size):
    chunk = rows[i:i+batch_size]
    data, count = supabase.table("validation_current").upsert(chunk).execute()

  print("UPLOAD COMPELTE")

if __name__ == "__main__":
  main()