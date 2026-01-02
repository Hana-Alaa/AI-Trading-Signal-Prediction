# ==============================================================
# PHASE 8 – PAPER TRADING CORE + PHASE 9 – FastAPI REST API
# ==============================================================

import sys
import warnings
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------
# Path setup – adds root of project so "config" can be imported
# --------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from config import MODELS_DIR  # ✅ now config import should work

# --------------------------------------------------------------
# Calibrated model wrapper
# --------------------------------------------------------------
class CalibratedModelWrapper:
    def __init__(self, base_model, iso_model):
        self.base_model = base_model
        self.iso_model = iso_model

    def predict_proba(self, X):
        base = self.base_model.predict_proba(X)[:, 1]
        calibrated = self.iso_model.predict(base)
        return np.vstack([1 - calibrated, calibrated]).T

    def predict(self, X, thr=0.5):
        return (self.predict_proba(X)[:, 1] >= thr).astype(int)


# --------------------------------------------------------------
# Utility – Load model and metadata
# --------------------------------------------------------------
def load_model_and_params():
    import __main__  # needed for joblib loading custom class
    __main__.CalibratedModelWrapper = CalibratedModelWrapper
    try:
        model = joblib.load(MODELS_DIR / "model_target_hit_final_calibrated.pkl")
        meta = json.load(open(MODELS_DIR / "metadata.json"))
        params = meta.get("optimized_params", {})
        features = meta["target_hit_model"]["features"]
        print("✅ Model & parameters loaded successfully.")
        return model, params, features
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load model or metadata: {e}")


# --------------------------------------------------------------
# Signal generation logic
# --------------------------------------------------------------
def generate_signal(model, params, features, user_input):
    """
    user_input: dict (from JSON) OR pandas DataFrame (from FastAPI POST)
    """
    # Handle both dict or DataFrame inputs
    if isinstance(user_input, pd.DataFrame):
        df = user_input.copy()
    else:
        df = pd.DataFrame([user_input])

    # Normalize column names
    rename_map = {
        "rsi": "RSI",
        "rsi_3d": "rsi_3d",
        "candle_wick": "upper_wick"
    }
    df.rename(columns=rename_map, inplace=True)

    # Fill any missing expected features
    for c in features:
        if c not in df.columns:
            df[c] = 0.0

    X = df[features]

    # ---- Predict ----
    proba = float(model.predict_proba(X)[:, 1][0])
    thr = params.get("best_threshold", 0.5)
    ev = params.get("expected_value", 0.0)
    rr = params.get("best_reward_risk", 2.0)
    rf = params.get("best_risk_fraction", 0.002)

    signal = bool(proba >= thr)
    conf = "high" if proba > 0.8 else "medium" if proba > 0.6 else "low"

    return {
        "enter_trade": signal,
        "target_probability": round(proba, 3),
        "expected_outcome": "target" if signal else "no_trade",
        "confidence_score": conf,
        "expected_pnl_avg": round(ev, 3),
        "threshold_used": thr,
        "risk_fraction_applied": rf,
        "reward_risk_ratio_applied": rr,
        "timestamp": datetime.now().isoformat()
    }


# ==============================================================
# PHASE 9 – FastAPI REST API for Trading Signal Prediction
# ==============================================================

from fastapi import FastAPI
from pydantic import BaseModel

warnings.filterwarnings("ignore")

# --------------------------------------------------------------
# Create FastAPI app + Load model
# --------------------------------------------------------------
app = FastAPI(title="Trading Signal REST API", version="1.0")

# Load model once at startup
model, params, features = load_model_and_params()

# --------------------------------------------------------------
# Pydantic schema for incoming data
# --------------------------------------------------------------
class Candle(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime = datetime.now()

# --------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------
@app.get("/")
def root():
    """Simple health‑check endpoint."""
    return {"status": "ok", "message": "Trading Signal API running ✅"}


@app.post("/predict")
def predict_signal(data: Candle):
    """
    Receives a candle (open/high/low/close/volume).
    Returns a trading‑signal recommendation with probability & confidence.
    """
    try:
        df = pd.DataFrame([data.dict()])
        df.set_index("timestamp", inplace=True)
        result = generate_signal(model, params, features, df)
        return result if result else {"error": "Failed to produce signal."}
    except Exception as e:
        return {"error": f"{e}"}