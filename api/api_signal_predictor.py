# ==============================================================
# PHASE 9 – FastAPI REST API for Trading Signal Prediction
# ==============================================================

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys, warnings, joblib, json

# --- Setup ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from config import MODELS_DIR
from api.phase8_paper_trading_api import (
    generate_signal,
    load_model_and_params
)

warnings.filterwarnings("ignore")

# ==============================================================
#  FastAPI APP + Load Model
# ==============================================================
app = FastAPI(title="Trading Signal REST API", version="1.0")

# Load model only once at startup
model, params, features = load_model_and_params()

# ==============================================================
#  Request Schema
# ==============================================================
class Candle(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime = datetime.now()

# ==============================================================
#  Endpoints
# ==============================================================

@app.get("/")
def root():
    """Simple health‑check endpoint."""
    return {"status": "ok", "message": "Trading Signal API running ✅"}

@app.post("/predict")
def predict_signal(data: Candle):    
    """
    Receives a list of candles (each one is a dict of open/high/low/close/volume).
    Returns a trading signal recommendation with confidence.
    """
    try:
        df = pd.DataFrame([data.dict()])
        df.set_index("timestamp", inplace=True)
        result = generate_signal(model, params, features, df)
        return result if result else {"error": "Failed to produce signal."}
    except Exception as e:
        return {"error": f"{e}"}