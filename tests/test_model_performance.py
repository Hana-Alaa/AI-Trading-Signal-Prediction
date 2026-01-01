import pandas as pd
import numpy as np
import sys
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay

# Add project root to path
# Script is in tests/
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from config import PROCESSED_DATA_DIR, MODELS_DIR

# 🚨 IMPORTANT: Re-define the wrapper so joblib can deserialize the model
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

def main():
    print("🚀 Starting Final Test Evaluation...")
    
    # 0. Ensure logs folder exists
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)

    # 1. Load metadata for threshold and feature alignment
    metadata_path = MODELS_DIR / "metadata.json"
    best_threshold = 0.5
    feature_names = []
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
            if "target_hit_model" in meta:
                target_meta = meta["target_hit_model"]
                best_threshold = target_meta.get("metrics_val", {}).get("best_threshold", 0.5)
                feature_names = target_meta.get("features", [])
    
    print(f"🎯 Using optimal threshold from metadata: {best_threshold}")

    # 2. Load test data
    test_path = PROCESSED_DATA_DIR / "splits" / "test.csv"
    if not test_path.exists():
        print(f"❌ Test file not found at: {test_path}")
        return
    
    test_df = pd.read_csv(test_path, low_memory=False)
    print(f"✅ Loaded test data: {test_df.shape}")
    
    # 3. Prepare X_test, y_test
    if feature_names:
        print(f"🧬 Aligning X_test with {len(feature_names)} features from metadata...")
        # Check if all features exist in test_df
        missing = [f for f in feature_names if f not in test_df.columns]
        if missing:
            print(f"⚠️ Warning: Missing features in test set: {missing[:5]}...")
            # Fill missing with 0
            for f in missing:
                test_df[f] = 0
        X_test = test_df[feature_names].copy()
    else:
        # Fallback to numeric-only logic
        exclude_cols = ['id', 'status', 'target_hit', 'stop_hit', 'target_type', 'hit_first', 'created_at', 'coin', 'TP1']
        X_test = test_df.drop(columns=[c for c in exclude_cols if c in test_df.columns], errors="ignore")
        X_test = X_test.select_dtypes(include=np.number).copy()
    
    y_test = test_df['target_hit']
    
    print(f"🔹 Features in Test: {X_test.shape[1]}")

    # 4. Load Model
    model_path = MODELS_DIR / "model_target_hit_final_calibrated.pkl"
    if not model_path.exists():
        print(f"❌ Model file not found at: {model_path}")
        return
    
    import __main__
    __main__.CalibratedModelWrapper = CalibratedModelWrapper
    
    model = joblib.load(model_path)
    print(f"✅ Loaded model: {model_path}")

    # Clean data types before prediction
    X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0)
    # 5. Calculate Metrics
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= best_threshold).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_val = roc_auc_score(y_test, y_prob)
    
    print("\n" + "="*40)
    print("📈 FINAL TEST PERFORMANCE RESULTS:")
    print(f"🔹 Accuracy:  {acc:.4f}")
    print(f"🔹 Precision: {prec:.4f}")
    print(f"🔹 Recall:    {rec:.4f}")
    print(f"🔹 F1 Score:  {f1:.4f}")
    print(f"🔹 ROC AUC:   {auc_val:.4f}")
    print("="*40)
    
    # 6. Visualization - Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("\n📊 CONFUSION MATRIX:")
    print(f"   Predicted 0   Predicted 1")
    print(f"Actual 0: {tn:<12} {fp:<12} (TN, FP)")
    print(f"Actual 1: {fn:<12} {tp:<12} (FN, TP)")
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Test Confusion Matrix (threshold={best_threshold})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    cm_path = logs_dir / "test_confusion_matrix.png"
    plt.savefig(cm_path)
    print(f"\n�️  Confusion matrix image saved to: {cm_path}")
    plt.close()

    # 7. Visualization - ROC Curve
    plt.figure(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title("ROC Curve – Test Set")
    roc_path = logs_dir / "test_roc_curve.png"
    plt.savefig(roc_path)
    print(f"� ROC Curve saved to: {roc_path}")
    plt.close()
    
    # 8. Classification Report
    print("\n📋 Detailed Classification Report:")
    report = classification_report(y_test, y_pred, zero_division=0, digits=4)
    print(report)
    
    # 9. Save JSON results
    results = {
        "test_metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc_val,
            "best_threshold": best_threshold
        },
        "evaluated_at": str(pd.Timestamp.now())
    }
    results_path = logs_dir / "test_evaluation_summary.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"📖 Test results summary saved to: {results_path}")
    print(f"✅ Test F1: {f1:.4f} | AUC: {auc_val:.4f} | Threshold: {best_threshold}")

if __name__ == "__main__":
    main()
