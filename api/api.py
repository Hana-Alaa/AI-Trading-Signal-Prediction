# ==============================================================
# PHASE 8 – PAPER TRADING CORE
# PHASE 9 – FastAPI REST API
# ==============================================================

import sys
import json
import warnings
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

# --------------------------------------------------------------
# Path setup
# --------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import MODELS_DIR
from decision.decision_layer import TradingDecisionEngine

warnings.filterwarnings("ignore")

# --------------------------------------------------------------
# Load models & metadata
# --------------------------------------------------------------
model_target = joblib.load(
    MODELS_DIR / "model_target_hit_final_calibrated.pkl"
)

model_stop = joblib.load(
    MODELS_DIR / "model_stop_hit_final_calibrated.pkl"
)

meta = json.load(open(MODELS_DIR / "metadata.json"))
FEATURES = meta["target_hit_model"]["features"]

ENTRY_THR = meta["optimized_params"]["entry_thr"]
STOP_THR  = meta["optimized_params"]["stop_thr"]

# --------------------------------------------------------------
# Decision Engine
# --------------------------------------------------------------
decision_engine = TradingDecisionEngine(
    entry_threshold=ENTRY_THR,
    stop_threshold=STOP_THR,
    load_models=False  
)

# --------------------------------------------------------------
# FastAPI App
# --------------------------------------------------------------
app = FastAPI(
    title="Trading Signal Prediction API",
    version="1.0"
)

# --------------------------------------------------------------
# Request Schema 
# --------------------------------------------------------------
class Candle(BaseModel):
    entry_price: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    rsi: float
    rsi_3d: float
    candle_body: float
    candle_wick: float

# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------
def prepare_features(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    df.rename(
        columns={
            "rsi": "RSI",
            "candle_wick": "upper_wick"
        },
        inplace=True
    )

    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features: {missing}")

    return df[FEATURES].astype(float)

def build_api_response(decision: dict) -> dict:
    p_target = decision["p_target"]
    p_stop   = decision["p_stop"]
    enter    = decision["decision"] == "ENTER"

    confidence = (
        "high" if p_target >= 0.8 else
        "medium" if p_target >= 0.6 else
        "low"
    )

    return {
        "enter_trade": enter,
        "target_probability": round(p_target, 2),
        "stop_probability": round(p_stop, 2),
        "expected_outcome": "target" if enter else "no_trade",
        # "estimated_time_to_event": "45 minutes",  
        "confidence_score": confidence
    }

# --------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------
@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Trading Signal API running ✅"
    }

@app.post("/predict")
def predict_signal(candle: Candle):
    try:
        X = prepare_features(candle.dict())

        p_target = float(model_target.predict_proba(X)[0, 1])
        p_stop   = float(model_stop.predict_proba(X)[0, 1])

        decision = decision_engine.decide(
            p_target=p_target,
            p_stop=p_stop
        )

        return build_api_response(decision)

    except Exception as e:
        return {"error": str(e)}