import pandas as pd
import json
import numpy as np

# df1 = pd.read_parquet("ensemble/outputs/X_train.parquet")
# df2 = pd.read_parquet("ensemble/outputs/X_test.parquet")
# print(len(df1))
# print(len(df2))

# load predictions
q_lin = pd.read_csv("quant_data/output-ml/__baseline_prediction.csv").rename(columns={"y_pred_log": "y_quant_linear"})
print("LENGTH OF LINEAR: " + str(len(q_lin)))
q_tree = pd.read_csv("quant_data/output-decision-tree/__decision_tree_predictions.csv").rename(columns={"y_pred_log": "y_quant_tree"})
print("LENGTH OF DECISION TREE: " + str(len(q_tree)))

q_qual = pd.read_csv("ml_neural_network/data/artifacts/qual_preds.csv")
q_qual = q_qual.rename(columns={"y_pred": "y_qual", "y_pred_log": "y_qual"})
print("LENGTH OF QUAL: " + str(len(q_qual)))

# load the actual true price
true = pd.read_json("ml_neural_network/data/processed/structured_with_target.json")[["artwork_id", "final_price"]].copy()
true["artwork_id"] = true["artwork_id"].astype(str)
true["y_true"] = np.log(true["final_price"].clip(lower=1e-9))
true = true[["artwork_id", "y_true"]]
true = true.drop_duplicates(subset=["artwork_id"], keep="first")

# load the split information
split = pd.Series(json.load(open("ensemble/outputs/split.json")), name="stage")
split.index = split.index.astype(str)

df = (true
    .merge(q_lin, on="artwork_id", how="inner")
    .merge(q_tree, on="artwork_id", how="inner")
    .merge(q_qual, on="artwork_id", how="inner")
    .merge(split, left_on="artwork_id", right_index=True, how="left"))

# debugging
print(f"TRUE length: {len(true)}")
print(f"TRUE unique IDs: {true['artwork_id'].nunique()}")
print(f"LINEAR unique IDs: {q_lin['artwork_id'].nunique()}")
print(f"TREE unique IDs: {q_tree['artwork_id'].nunique()}")
print(f"QUAL unique IDs: {q_qual['artwork_id'].nunique()}")

# After each merge:
temp1 = true.merge(q_lin, on="artwork_id", how="inner")
print(f"After merge with linear: {len(temp1)}")

temp2 = temp1.merge(q_tree, on="artwork_id", how="inner")
print(f"After merge with tree: {len(temp2)}")

temp3 = temp2.merge(q_qual, on="artwork_id", how="inner")
print(f"After merge with qual: {len(temp3)}")

print(f"TRUE length after dedup: {len(true)}")
