# api_signal_predictor.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd, joblib, json
from pathlib import Path
from config import MODELS_DIR

# ----------------------------------------------------------
# ✅ أضفي تعريف نفس الـ Wrapper هنا قبل تحميل النموذج
# ----------------------------------------------------------
class CalibratedModelWrapper:
    def __init__(self, base_model, iso_model):
        self.base_model = base_model
        self.iso_model = iso_model
    def predict_proba(self, X):
        import numpy as np
        base_probs = self.base_model.predict_proba(X)[:, 1]
        calibrated_probs = self.iso_model.predict(base_probs)
        return np.vstack([1 - calibrated_probs, calibrated_probs]).T
    def predict(self, X, thr=0.5):
        import numpy as np
        return (self.predict_proba(X)[:, 1] >= thr).astype(int)

# 🔑 مهم جدًا: عرّفيه داخل __main__ للطرف الآخر (الـ unpickler)
import __main__
__main__.CalibratedModelWrapper = CalibratedModelWrapper
# ----------------------------------------------------------

app = FastAPI(title="AI Trading Signal Prediction API")

# load metadata + model
meta = json.load(open(MODELS_DIR / "metadata.json"))
features = meta["target_hit_model"]["features"]
best_thr = meta["target_hit_model"]["metrics_val"]["best_threshold"]

model = joblib.load(MODELS_DIR / "model_target_hit_final_calibrated.pkl")

# request schema
class SignalRequest(BaseModel):
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

@app.post("/predict")
def predict(req: SignalRequest):
    X = pd.DataFrame([req.dict()])[features]
    prob = float(model.predict_proba(X)[0,1])
    enter = prob >= best_thr
    return {
        "enter_trade": bool(enter),
        "target_probability": round(prob,3),
        "stop_probability": round(1-prob,3),
        "expected_outcome": "target" if enter else "stop",
        "estimated_time_to_event": "45 minutes",
        "confidence_score": (
            "high" if prob>0.7 else "medium" if prob>0.5 else "low"
        )
    }