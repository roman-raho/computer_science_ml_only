# ml/v4/v4_test_run.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# ---------- config: paths relative to project root ----------
LR_MODEL = Path("ml/quant_data/output-ml/baseline_v2_model.joblib")
LR_FM    = Path("ml/quant_data/output-ml/baseline_v2_feature_mart.csv")

RF_MODEL = Path("ml/quant_data/output-decision-tree/_random_forest.joblib")
RF_FM    = Path("ml/quant_data/output-decision-tree/_feature_mart.csv")

CONFORMAL_JSON = Path("ml/v4/v4_conformal.json")
TARGET_COL = "price"   # change if your target column is named differently

# ---------- utils (same logic as v4_blend, but dual inputs) ----------
def _log1p_safe(a):
    import numpy as np
    a = np.asarray(a, dtype=float)
    a = np.clip(a, 0.0, None)
    return np.log1p(a)

def _expm1(a):
    import numpy as np
    return np.expm1(a)

def interval_from_q(mu: float, q: float) -> tuple[float, float]:
    log_mu = _log1p_safe(mu)
    lo = _expm1(np.maximum(0.0, log_mu - q))
    hi = _expm1(log_mu + q)
    return float(lo), float(hi)

def confidence_from_bounds(lo: float, hi: float) -> str:
    width_rel = (hi - lo) / max(hi, 1e-9)
    if width_rel <= 0.20:
        return "High"
    if width_rel <= 0.40:
        return "Medium"
    return "Low"

# ---------- load artefacts ----------
def load_conformal_params(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_numeric_matrix(csv_path: Path, target_col: str):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[target_col]).select_dtypes(include="number").fillna(0.0).values
    y = df[target_col].values
    return X, y

def main():
    # models
    lr = joblib.load(LR_MODEL)
    rf = joblib.load(RF_MODEL)

    # conformal q's
    p = load_conformal_params(CONFORMAL_JSON)
    q_lr, q_rf = float(p["q_lr"]), float(p["q_rf"])

    # feature marts (separate per model)
    X_lr, y_lr = load_numeric_matrix(LR_FM, TARGET_COL)
    X_rf, y_rf = load_numeric_matrix(RF_FM, TARGET_COL)

    # pick rows (index 0 from each mart is fine for UT01)
    x_lr = X_lr[0]
    x_rf = X_rf[0]

    # predictions
    mu_lr = float(lr.predict([x_lr])[0])
    mu_rf = float(rf.predict([x_rf])[0])

    # per-model intervals
    lo_lr, hi_lr = interval_from_q(mu_lr, q_lr)
    lo_rf, hi_rf = interval_from_q(mu_rf, q_rf)

    # blend: median point + union interval
    mu_blend = float(np.median([mu_lr, mu_rf]))
    lo_blend = float(min(lo_lr, lo_rf))
    hi_blend = float(max(hi_lr, hi_rf))
    conf = confidence_from_bounds(lo_blend, hi_blend)

    result = {
        "lr": {"mu": mu_lr, "lo": lo_lr, "hi": hi_lr},
        "rf": {"mu": mu_rf, "lo": lo_rf, "hi": hi_rf},
        "blend": {"mu": mu_blend, "lo": lo_blend, "hi": hi_blend, "confidence": conf},
        "meta": {"q_lr": q_lr, "q_rf": q_rf, "source_rows": {"lr": 0, "rf": 0}},
    }

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
