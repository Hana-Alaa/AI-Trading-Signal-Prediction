import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------
# Wrapper for calibrated model
# --------------------------------------------------
class CalibratedModelWrapper:
    def __init__(self, base_model, iso_model):
        self.base_model = base_model
        self.iso_model = iso_model

    def predict_proba(self, X):
        base = self.base_model.predict_proba(X)[:, 1]
        calibrated = self.iso_model.predict(base)
        return np.vstack([1 - calibrated, calibrated]).T

# --------------------------------------------------
# SHAP runner
# --------------------------------------------------
def run_shap(
    model_path: Path,
    test_csv: Path,
    model_name: str,
    reports_dir: Path,
    sample_size: int = 1000
):
    print(f"\n🔍 Running SHAP for: {model_name}")

    # Load data
    df = pd.read_csv(test_csv)

    # Drop non-feature columns if exist
    drop_cols = [c for c in ["id", "created_at"] if c in df.columns]
    X = df.drop(columns=drop_cols)

    # Load model
    import __main__
    __main__.CalibratedModelWrapper = CalibratedModelWrapper
    model = joblib.load(model_path)

    base_model = model.base_model if hasattr(model, "base_model") else model

    # SHAP explainer
    try:
        explainer = shap.TreeExplainer(base_model)
    except Exception:
        explainer = shap.Explainer(base_model, X)

    sample = X.sample(min(sample_size, len(X)), random_state=42)
    shap_exp = explainer(sample)

    # Handle binary classification
    if shap_exp.values.ndim == 3:
        shap_values = shap_exp.values[:, :, 1]
        shap_exp = shap.Explanation(
            values=shap_values,
            base_values=shap_exp.base_values[:, 1],
            data=sample,
            feature_names=sample.columns
        )

    # Mean importance
    shap_df = pd.DataFrame(
        np.abs(shap_exp.values),
        columns=sample.columns
    )

    importance = shap_df.mean().sort_values(ascending=False)

    # Save bar plot
    plt.figure(figsize=(10, 6))
    importance.head(10)[::-1].plot(kind="barh")
    plt.title(f"Top Features – {model_name}")
    plt.tight_layout()

    bar_path = reports_dir / f"shap_{model_name}_importance.png"
    plt.savefig(bar_path, dpi=300)
    plt.close()

    # Save summary plot
    shap.summary_plot(shap_exp, sample, show=False)
    summary_path = reports_dir / f"shap_{model_name}_summary.png"
    plt.savefig(summary_path, dpi=300)
    plt.close()

    print(f"✅ Saved SHAP plots for {model_name}")
    return importance