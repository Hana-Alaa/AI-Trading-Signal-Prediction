import pandas as pd
import numpy as np
import sys
import json
import joblib
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    precision_recall_curve,
)

# ────────────────────────────────────────────────
# Project setup
# ────────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from config import PROCESSED_DATA_DIR, MODELS_DIR  # noqa

LOGS_DIR = project_root / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# ────────────────────────────────────────────────
# Wrapper (for calibrated models)
# ────────────────────────────────────────────────
class CalibratedModelWrapper:
    def __init__(self, base_model, iso_model):
        self.base_model = base_model
        self.iso_model = iso_model

    def predict_proba(self, X):
        base_probs = self.base_model.predict_proba(X)[:, 1]
        calibrated_probs = self.iso_model.predict(base_probs)
        return np.vstack([1 - calibrated_probs, calibrated_probs]).T

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

# ────────────────────────────────────────────────
# Metrics helpers
# ────────────────────────────────────────────────
def precision_at_k(y_true, y_scores, k=0.1):
    cutoff = int(len(y_true) * k)
    if cutoff == 0:
        return 0.0
    idx = np.argsort(y_scores)[::-1][:cutoff]
    return precision_score(y_true[idx], np.ones(len(idx)), zero_division=0)

# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser("Final Test Evaluation")
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=["target_hit", "stop_hit"],
        help="Which target to evaluate",
    )
    args = parser.parse_args()
    target = args.target

    print(f"\nFINAL TEST EVALUATION | TARGET = {target.upper()}")
    print("=" * 55)

    # ────────────────────────────────────────────────
    # Paths (target-aware)
    # ────────────────────────────────────────────────
    split_dir = PROCESSED_DATA_DIR / f"splits_{target}"
    test_path = split_dir / "test.csv"
    model_path = MODELS_DIR / f"model_{target}_final_calibrated.pkl"
    metadata_path = MODELS_DIR / "metadata.json"

    if not test_path.exists():
        raise FileNotFoundError(f"❌ Missing test file: {test_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"❌ Missing model file: {model_path}")

    if not metadata_path.exists():
        raise FileNotFoundError("❌ metadata.json not found — REQUIRED for final test")

    # ────────────────────────────────────────────────
    # Load metadata
    # ────────────────────────────────────────────────
    with open(metadata_path, "r") as f:
        meta = json.load(f)

    meta_key = f"{target}_model"
    if meta_key not in meta:
        raise KeyError(f"❌ Metadata missing key: {meta_key}")

    target_meta = meta[meta_key]
    best_threshold = target_meta["metrics_val"]["best_threshold"]
    feature_names = target_meta["features"]

    print(f"🔹 Threshold (from VAL): {best_threshold}")
    print(f"🔹 #Features: {len(feature_names)}")

    # ────────────────────────────────────────────────
    # Load test data
    # ────────────────────────────────────────────────
    test_df = pd.read_csv(test_path, low_memory=False)
    print(f"✅ Loaded test set: {test_df.shape}")

    # Align features strictly
    missing = [f for f in feature_names if f not in test_df.columns]
    if missing:
        raise ValueError(f"❌ Missing required features in test set: {missing}")

    X_test = test_df[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0)
    y_test = test_df[target].astype(int)

    # ────────────────────────────────────────────────
    # Load model
    # ────────────────────────────────────────────────
    import __main__
    __main__.CalibratedModelWrapper = CalibratedModelWrapper

    model = joblib.load(model_path)
    print(f"✅ Loaded model: {model_path.name}")

    # ────────────────────────────────────────────────
    # Predictions
    # ────────────────────────────────────────────────
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= best_threshold).astype(int)

    # ────────────────────────────────────────────────
    # Core metrics
    # ────────────────────────────────────────────────
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_val = roc_auc_score(y_test, y_prob)

    print("\n📈 CORE METRICS (TEST)")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {auc_val:.4f}")

    # ────────────────────────────────────────────────
    # Trading-oriented analysis
    # ────────────────────────────────────────────────
    if target == "target_hit":
        print("\n💰 ENTRY SIGNAL ANALYSIS")
        p5 = precision_at_k(y_test.values, y_prob, 0.05)
        p10 = precision_at_k(y_test.values, y_prob, 0.10)
        p20 = precision_at_k(y_test.values, y_prob, 0.20)

        avg_win, avg_loss = 1.5, 1.0
        ev = (prec * avg_win) - ((1 - prec) * avg_loss)

        print(f"Precision@5% : {p5:.4f}")
        print(f"Precision@10%: {p10:.4f}")
        print(f"Precision@20%: {p20:.4f}")
        print(f"Expected Value / trade: {ev:.4f}")

    else:
        print("\n🛑 RISK FILTER ANALYSIS")
        stop_rate = y_pred.mean()
        print(f"Stop Signal Rate: {stop_rate:.4f}")
        print("⬇️ Lower is better (while keeping precision high)")

    # ────────────────────────────────────────────────
    # Confusion Matrix
    # ────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix | {target}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_path = LOGS_DIR / f"test_{target}_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    # ────────────────────────────────────────────────
    # ROC Curve
    # ────────────────────────────────────────────────
    plt.figure(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title(f"ROC Curve | {target}")
    roc_path = LOGS_DIR / f"test_{target}_roc_curve.png"
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()

    # ────────────────────────────────────────────────
    # Precision-Recall Curve
    # ────────────────────────────────────────────────
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, precisions[:-1], label="Precision")
    plt.plot(thresholds, recalls[:-1], label="Recall")
    plt.title(f"Precision-Recall vs Threshold | {target}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    pr_path = LOGS_DIR / f"test_{target}_precision_recall_curve.png"
    plt.tight_layout()
    plt.savefig(pr_path)
    plt.close()

    # ────────────────────────────────────────────────
    # Classification Report
    # ────────────────────────────────────────────────
    print("\n📋 CLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    # ────────────────────────────────────────────────
    # Threshold sensitivity (ANALYSIS ONLY)
    # ────────────────────────────────────────────────
    print("\n⚠️ Threshold Sensitivity (ANALYSIS ONLY — NO TUNING)")
    rows = []
    for t in np.linspace(0.1, 0.9, 17):
        yp = (y_prob >= t).astype(int)
        rows.append({
            "threshold": t,
            "precision": precision_score(y_test, yp, zero_division=0),
            "recall": recall_score(y_test, yp, zero_division=0),
            "f1": f1_score(y_test, yp, zero_division=0),
        })

    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # ────────────────────────────────────────────────
    # Save summary
    # ────────────────────────────────────────────────
    summary = {
        "target": target,
        "metrics_test": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc_val,
            "threshold": best_threshold,
        },
        "evaluated_at": str(pd.Timestamp.now()),
    }

    out_path = LOGS_DIR / f"test_{target}_evaluation_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=4)

    print("\n✅ FINAL TEST COMPLETED SUCCESSFULLY")
    print(f"📁 Logs saved to: {LOGS_DIR}")
    print("=" * 55)

if __name__ == "__main__":
    main()