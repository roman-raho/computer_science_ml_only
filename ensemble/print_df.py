import pandas as pd

df1 = pd.read_parquet("ensemble/outputs/X_train.parquet")
df2 = pd.read_parquet("ensemble/outputs/X_test.parquet")
print(len(df1) + len(df2))