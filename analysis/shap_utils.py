import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import __main__

# Needed for loading calibrated models
class CalibratedModelWrapper:
    def __init__(self, base_model, iso_model):
        self.base_model = base_model
        self.iso_model = iso_model

    def predict_proba(self, X):
        base = self.base_model.predict_proba(X)[:, 1]
        calibrated = self.iso_model.predict(base)
        return np.vstack([1 - calibrated, calibrated]).T

__main__.CalibratedModelWrapper = CalibratedModelWrapper

def run_shap_analysis(
    model_path: Path,
    X: pd.DataFrame,
    feature_cols: list,
    model_name: str,
    reports_dir: Path,
    sample_size: int = 1000
):
    model = joblib.load(model_path)
    base_model = model.base_model if hasattr(model, "base_model") else model

    try:
        explainer = shap.TreeExplainer(base_model)
    except Exception:
        explainer = shap.Explainer(base_model, X)

    sample = X.sample(min(sample_size, len(X)), random_state=42)
    shap_output = explainer(sample)

    if isinstance(shap_output, shap.Explanation):
        if shap_output.values.ndim == 3:
            shap_values = shap_output.values[:, :, 1]
        else:
            shap_values = shap_output.values
    else:
        shap_values = shap_output[1]

    shap_df = pd.DataFrame(shap_values, columns=feature_cols)

    importance = (
        np.abs(shap_df)
        .mean()
        .sort_values(ascending=False)
    )

    return importance, shap_values, sample