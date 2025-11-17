from datetime import datetime, timezone
import pandas as pd

# saw this on stack overflow and thought it would be cool to use it
def iso_created_at() -> str:
  return datetime.now(timezone.utc).isoformat()

def minmax_norm(series: pd.Series) -> pd.Series:
  if series.empty:
    return series.astype(float) # if it is empty return series with 0s
  
  minv = series.min() # get the min of the series
  maxv = series.max() # get the max of the series

  if pd.isna(minv) or pd.isna(maxv) or maxv == minv: # if the min is a Nan or the max is or the max = min, hence diving by 0 then bail out and fill series with 0
    return pd.Series([0.0] * len(series), index=series.index, dtype=float)
  
  return (series - minv) / (maxv - minv) # else return normalised series
