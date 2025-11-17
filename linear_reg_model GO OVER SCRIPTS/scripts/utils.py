from __future__ import annotations
from datetime import datetime
import numpy as np
import pandas as pd
import sys


def iso_created_at() -> str:
  return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ") #format the time had to look it up

def winsorize_series(s: pd.Series, lower=0.01, upper=0.99) -> pd.Series: # limit extreme percentiles
  if s.empty:
    return s
  
  low, high = s.quantile(lower), s.quantile(upper)
  return s.clip(lower=low, upper=high)

def minmax_norm(s: pd.Series) -> pd.Series: # normalisation
  s = winsorize_series(s.astype(float))
  mn, mx = s.min(), s.max()
  if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
    return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
  return (s - mn) / (mx - mn)

def coefvar(x: pd.Series) -> float: # calculate the variance coefficient 
  x = x.dropna()
  if len(x) < 2: # if its 1 or 0 cannot calcualte anything from that
    return np.nan
  m = x.mean() # calculate the mean
  sd = x.std(ddof=1) # set degress of freedom to 1
  if not np.isfinite(m) or m == 0:
    return np.nan
  return float(sd / abs(m))

def recency_index(years_since_last: float) -> float:
    years_since_last = max(years_since_last, 0.0)
    return float(np.exp(-years_since_last / 3.0)) # divide by 3 in order to limit values

def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series: # calcualte the safe ratio
  den = den.replace(0, np.nan)
  return num.astype(float) / den.astype(float)

def to_quarter(ts: pd.Series) -> pd.Series:
  return pd.PeriodIndex(ts, freq="Q").to_timestamp(how="end") # quarter endpoints

def load_json(path: str) -> pd.DataFrame: # load the json
  try:
    return pd.read_json(path)
  except ValueError as e:
    sys.exit(1)  