import numpy as np
import json
from pathlib import Path
import joblib

from v4_conformal import _log1p_safe, _expm1 # reuse utils

def load_conformal_params(path: str | Path = "v4_conformal.json"):
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)
  
def interval_from_q(mu: float, q: float) -> tuple[float, float]:
  log_mu = _log1p_safe(mu)
  lo = _expm1(np.maximum(0.0, log_mu - q))
  hi = _expm1(log_mu + q)
  return float(lo), float(hi)

def blend_prediction(x, lr_model, rf_model, q_lr: float, q_rf: float) -> dict:
  # run both models, build intervals, blend point + interval, and assign confidence
  mu_lr = float(lr_model.predict([x])[0])
  mu_rf = float(rf_model.predict([x])[0])
  lo_lr, hi_lr = interval_from_q(mu_lr, q_lr)
  lo_rf, hi_rf = interval_from_q(mu_rf, q_rf)

  # blended point = median of the two
  mu_blend = float(np.median([mu_lr, mu_rf]))
  lo_blend = float(min(lo_lr, lo_rf))
  hi_blend = float(max(hi_lr, hi_rf))

  # confidence badge from relative width
  width_rel = (hi_blend - lo_blend) / max(hi_blend, 1e-9)
  if width_rel <= 0.20:
      conf = "High"
  elif width_rel <= 0.40:
      conf = "Medium"
  else:
      conf = "Low"
  return {
      "lr": {"mu": mu_lr, "lo": lo_lr, "hi": hi_lr},
      "rf": {"mu": mu_rf, "lo": lo_rf, "hi": hi_rf},
      "blend": {"mu": mu_blend, "lo": lo_blend, "hi": hi_blend, "confidence": conf},
  }