import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import types

# ────────────────────────────────────────────────
# Project paths
# ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# ────────────────────────────────────────────────
# Load metadata
# ────────────────────────────────────────────────
with open(MODELS_DIR / "metadata.json", "r") as f:
    META = json.load(f)

# ────────────────────────────────────────────────
# Feature Builder
# ────────────────────────────────────────────────
def build_features(candle: dict) -> pd.DataFrame:
    open_ = candle["open"]
    high = candle["high"]
    low = candle["low"]
    close = candle["close"]

    upper_wick = high - max(open_, close)
    candle_range = high - low

    features = {
        "close": close,
        "volume": candle["volume"],
        "candle_body": candle["candle_body"],
        "upper_wick": upper_wick,
        "candle_range": candle_range,
        "wick_ratio": upper_wick / candle_range if candle_range > 0 else 0,
        "ratio_high_low": high / low if low > 0 else 0,
        "ratio_close_high": close / high if high > 0 else 0,
    }

    return pd.DataFrame([features])

# ────────────────────────────────────────────────
# Calibrated Wrapper (pickle-safe)
# ────────────────────────────────────────────────
calibrated_wrapper = types.ModuleType("calibrated_wrapper")

class CalibratedModelWrapper:
    def __init__(self, base_model, iso_model):
        self.base_model = base_model
        self.iso_model = iso_model

    def predict_proba(self, X):
        base_probs = self.base_model.predict_proba(X)[:, 1]
        calibrated_probs = self.iso_model.predict(base_probs)
        return np.vstack([1 - calibrated_probs, calibrated_probs]).T

calibrated_wrapper.CalibratedModelWrapper = CalibratedModelWrapper
sys.modules["calibrated_wrapper"] = calibrated_wrapper

import __main__
__main__.CalibratedModelWrapper = CalibratedModelWrapper

# ────────────────────────────────────────────────
# Trading Decision Engine
# ────────────────────────────────────────────────
class TradingDecisionEngine:
    def __init__(self, entry_threshold=None, stop_threshold=None, load_models=True):
        if load_models:
            self.entry_model = joblib.load(
                MODELS_DIR / "model_target_hit_final_calibrated.pkl"
            )
            self.stop_model = joblib.load(
                MODELS_DIR / "model_stop_hit_final_calibrated.pkl"
            )
        else:
            self.entry_model = None
            self.stop_model = None

        self.entry_threshold = (
            entry_threshold
            if entry_threshold is not None
            else META["target_hit_model"]["metrics_val"]["best_threshold"]
        )

        self.stop_threshold = (
            stop_threshold
            if stop_threshold is not None
            else META["stop_hit_model"]["metrics_val"]["best_threshold"]
        )

        self.feature_names = META["target_hit_model"]["features"]

    def decide(
        self,
        candle: dict = None,
        p_target: float = None,
        p_stop: float = None
    ) -> dict:

        # ── calculate probs if not provided
        if p_target is None or p_stop is None:
            if candle is None:
                raise ValueError("Provide candle or probabilities")

            X = build_features(candle)
            X = X[self.feature_names].astype(float)

            p_target = float(self.entry_model.predict_proba(X)[0, 1])
            p_stop   = float(self.stop_model.predict_proba(X)[0, 1])

        # ── decision logic
        decision = "NO_TRADE"
        reason = []

        if p_target >= self.entry_threshold:
            if p_stop < 0.1:
                decision = "ENTER"
                reason.append("Low stop risk")
            elif p_stop < 0.25:
                decision = "ENTER_HIGH_RISK"
                reason.append("Moderate stop risk")
            else:
                decision = "NO_TRADE"
                reason.append("High stop risk")

        return {
            "decision": decision,
            "p_target": round(p_target, 4),
            "p_stop": round(p_stop, 4),
            "entry_threshold": self.entry_threshold,
            "stop_threshold": self.stop_threshold,
            "reason": reason,
        }