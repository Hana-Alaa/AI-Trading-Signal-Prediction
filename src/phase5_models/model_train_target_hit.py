import pandas as pd
import numpy as np
import sys
import joblib
import json
import traceback
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
# Add project root to path
# Script is in src/phase5_models/
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from config import PROCESSED_DATA_DIR, MODELS_DIR, MODEL_VERSION, CREATED_BY

def load_splits():
    """Load train and validation splits."""
    splits_dir = PROCESSED_DATA_DIR / "splits"
    train_path = splits_dir / "train.csv"
    valid_path = splits_dir / "valid.csv"
    
    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError("Split files not found in data/processed/splits/. Run Phase 4 first.")
        
    # Using low_memory=False to handle mixed types warnings if any
    train = pd.read_csv(train_path, low_memory=False)
    valid = pd.read_csv(valid_path, low_memory=False)
    print(f"Loaded Splits - Train: {train.shape}, Valid: {valid.shape}")
    return train, valid

def prepare_data(df, target_col='target_hit', reference_features=None):
    """Separate features and target and align columns with training set."""
    exclude_cols = [
        'id', 'status', 'target_hit', 'stop_hit', 'target_type',
        'hit_first', 'created_at', 'coin', 'TP1'
    ]
    X = df.drop(columns=[c for c in exclude_cols if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=np.number).copy()
    y = df[target_col]

    if reference_features is not None:
        missing_cols = [c for c in reference_features if c not in X.columns]
        extra_cols   = [c for c in X.columns if c not in reference_features]

        for col in missing_cols:
            X[col] = 0

        if extra_cols:
            X = X.drop(columns=extra_cols)

        X = X[reference_features]

    return X, y
# ======================================
# Models
# ======================================
def train_logistic_regression(X_train, y_train):
    print("\n" + "="*40)
    print("⏳ Training Logistic Regression (v1.0)...")
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    print("✅ Logistic Regression Training Complete.")
    return model

def train_random_forest(X_train, y_train):
    """Train Random Forest with simple tuning."""
    print("\n" + "="*40)
    print("⏳ Training Random Forest (v1.2)...")

    rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,             
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print("✅ Random Forest Training Complete.")
    return rf

# def train_lightgbm(X_train, y_train):
#     """Train LightGBM model with basic tuning."""
#     print("\n" + "="*40)
#     print("⏳ Training LightGBM (v1.3)...")

#     # handle imbalance
#     count_neg = (y_train == 0).sum()
#     count_pos = (y_train == 1).sum()
#     scale_pos_weight = count_neg / (count_pos + 1e-9)

#     lgb_model = LGBMClassifier(
#         n_estimators=400,
#         learning_rate=0.05,
#         num_leaves=31,
#         subsample=0.9,
#         colsample_bytree=0.8,
#         scale_pos_weight=scale_pos_weight,
#         random_state=42
#     )
#     lgb_model.fit(X_train, y_train)
#     print("✅ LightGBM Training Complete.")
#     return lgb_model

def train_lightgbm(X_train, y_train):
    """Train LightGBM with hyperparameter tuning via RandomizedSearchCV."""
    print("\n" + "="*40)
    print("⏳ Training LightGBM with Random Search (v1.4)...")

    # Handle class imbalance
    count_neg = (y_train == 0).sum()
    count_pos = (y_train == 1).sum()
    scale_pos_weight = count_neg / (count_pos + 1e-9)

    base_model = LGBMClassifier(
        objective="binary",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        metric="binary_logloss"
    )

    # Define parameter space
    param_dist = {
        "n_estimators": [300, 500, 700],
        "learning_rate": [0.02, 0.03, 0.04, 0.05],
        "num_leaves": [31, 45, 60],
        "max_depth": [6, 8, 10],
        "min_child_samples": [15, 20, 25],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "reg_alpha": [0.0, 0.05, 0.1],
        "reg_lambda": [0.0, 0.05, 0.1]
    }

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=12,              
        scoring="f1",
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    print("\n✅ Best LightGBM Parameters found:")
    print(search.best_params_)
    best_model = search.best_estimator_
    print("✅ LightGBM Training Complete with optimized parameters.")
    return best_model

def train_xgboost(X_train, y_train):
    print("\n" + "="*40)
    print("⏳ Training XGBoost with Hyperparameter Tuning (v1.1)...")
    count_neg = (y_train == 0).sum()
    count_pos = (y_train == 1).sum()
    scale_weight = count_neg / (count_pos + 1e-9)

    base_model = XGBClassifier(
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=scale_weight
    )
    param_grid = {
        "n_estimators": [200, 400, 600],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_weight": [3, 5]
    }
    tuner = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=5,
        scoring="f1",
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    tuner.fit(X_train, y_train)
    print("\n✅ Best XGB Params:", tuner.best_params_)
    best_model = tuner.best_estimator_
    print("✅ XGBoost Training Complete.")
    return best_model

# ======================================
# Evaluation + Saving
# ======================================

def evaluate_model(model, X_val, y_val, model_label, X_train=None, y_train=None):
    """Evaluate model comprehensively on training and validation sets."""
    print(f"\n📊 Evaluation Results for {model_label}:")
    y_prob = model.predict_proba(X_val)[:, 1]

    best_f1 = -1
    best_metrics = {}
    best_thresh = 0.5
    y_pred_best = None

    # Tune threshold in fine increments
    thresholds = np.linspace(0.05, 0.95, 19)
    f1_scores = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        f1_scores.append(f1)
        if f1 > best_f1:
            best_f1, best_thresh, y_pred_best = f1, t, y_pred

    # Optional: plot F1 vs threshold
    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, f1_scores, marker='o', label='F1 Score')
    plt.axvline(best_thresh, color='red', linestyle='--', label=f'Best Threshold: {best_thresh:.2f}')
    plt.title(f"F1 vs Threshold – {model_label}")
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # === Validation metrics ===
    prec = precision_score(y_val, y_pred_best, zero_division=0)
    rec  = recall_score(y_val, y_pred_best, zero_division=0)
    f1   = f1_score(y_val, y_pred_best, zero_division=0)
    acc  = accuracy_score(y_val, y_pred_best)
    roc  = roc_auc_score(y_val, y_prob)

    print(f"\n✅ Best threshold selected: {best_thresh}")
    print(f"VAL → Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {roc:.4f}")

    # === TRAIN performance (optional) ===
    if X_train is not None and y_train is not None:
        y_train_prob = model.predict_proba(X_train)[:, 1]
        y_train_pred = (y_train_prob >= best_thresh).astype(int)
        acc_tr = accuracy_score(y_train, y_train_pred)
        f1_tr  = f1_score(y_train, y_train_pred, zero_division=0)
        print(f"TRAIN → Acc: {acc_tr:.4f}, F1: {f1_tr:.4f}")
        print(f"⚖️  Gap (Train − Val F1): {f1_tr - f1:.4f}\n")

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred_best)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"Confusion Matrix – {model_label} (thr={best_thresh})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # Classification Report
    print("\n📋 Classification Report Summary:")
    print(classification_report(y_val, y_pred_best, zero_division=0, digits=4))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc,
        "best_threshold": float(best_thresh)
    }

from sklearn.isotonic import IsotonicRegression
class CalibratedModelWrapper:
    def __init__(self, base_model, iso_model):
        self.base_model = base_model
        self.iso_model = iso_model

    def predict_proba(self, X):
        base_probs = self.base_model.predict_proba(X)[:, 1]
        calibrated_probs = self.iso_model.predict(base_probs)
        # return probabilities for both classes (n, 2)
        return np.vstack([1 - calibrated_probs, calibrated_probs]).T

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

def calibrate_best_model(model, X_val, y_val, model_label, method='isotonic'):
    """
    Manual calibration compatible with sklearn>=1.8.
    Fits an IsotonicRegression over the model's predicted probabilities
    and returns a pickle‑safe wrapped model.
    """
    print("\n" + "=" * 40)
    print(f"⏳ Calibrating {model_label} manually using {method} scaling...")

    # Step 1: get raw model probabilities
    y_prob = model.predict_proba(X_val)[:, 1]

    # Step 2: fit isotonic regression
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(y_prob, y_val)

    # Step 3: wrap and return
    calibrated_model = CalibratedModelWrapper(model, iso_reg)
    print(f"✅ Manual {method} calibration complete.")
    return calibrated_model
def plot_calibration_curve_comparison(models_dict, X_val, y_val):
    """
    Plot calibration curves for multiple models to compare.
    """
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    for label, model in models_dict.items():
        prob_pos = model.predict_proba(X_val)[:, 1]
        fraction_of_positives, mean_predicted_value = calibration_curve(y_val, prob_pos, n_bins=10)
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=label)
        
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted value")
    plt.title("Calibration curves (Reliability Diagram)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

def save_model_info(model, metrics, algorithm_name, version, feature_names):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_filename = f"model_target_hit_v{version}.pkl"
    model_path = MODELS_DIR / model_filename
    joblib.dump(model, model_path)
    print(f"💾 Model {version} saved to: {model_path}")

    metadata_path = MODELS_DIR / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}

    if 'target_hit_versions' not in metadata:
        metadata['target_hit_versions'] = []

    version_entry = {
        'version': version,
        'algorithm': algorithm_name,
        'trained_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'created_by': CREATED_BY,
        'metrics_val': metrics,
        'feature_count': len(feature_names),
        'features': list(feature_names),
        'model_path': str(model_filename)
    }
    metadata['target_hit_versions'] = [v for v in metadata['target_hit_versions'] if v['version'] != version]
    metadata['target_hit_versions'].append(version_entry)
    metadata['target_hit_model'] = version_entry

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"📖 Metadata updated for version {version}")

# ======================================
# MAIN
# ======================================
def main():
    try:
        train, valid = load_splits()
        X_train, y_train = prepare_data(train)
        feature_list = X_train.columns.tolist()
        X_val, y_val = prepare_data(valid, reference_features=feature_list)
        print(f"Feature count: {X_train.shape[1]}")

        # 1️. Logistic Regression
        lr = train_logistic_regression(X_train, y_train)

        # Train Performance
        y_pred_train = lr.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred_train)
        train_f1  = f1_score(y_train, y_pred_train, zero_division=0)
        print(f"\n📘 Logistic Regression Train Accuracy: {train_acc:.4f} | Train F1: {train_f1:.4f}")

        # Validation Performance
        lr_metrics = evaluate_model(lr, X_val, y_val, "Logistic Regression (v1.2)")

        save_model_info(lr, lr_metrics, "Logistic Regression", "1.2.2_balanced15", X_train.columns)

        # 2️. XGBoost
        xgb = train_xgboost(X_train, y_train)
        xgb_metrics = evaluate_model(xgb, X_val, y_val, "XGBoost (v1.1)", X_train, y_train)
        save_model_info(xgb, xgb_metrics, "XGBoost", "1.1.2_balanced15", X_train.columns)

        # Train Performance
        y_pred_train = xgb.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred_train)
        train_f1  = f1_score(y_train, y_pred_train, zero_division=0)
        print(f"\n📘 XGBoost Train Accuracy: {train_acc:.4f} | Train F1: {train_f1:.4f}")

        # Validation Performance
        xgb_metrics = evaluate_model(xgb, X_val, y_val, " XGBoost (v1.2)")

        save_model_info(xgb, xgb_metrics, " XGBoost", "1.2.2_balanced15", X_train.columns)

        # 3️. Random Forest
        rf = train_random_forest(X_train, y_train)

        # Train Performance
        y_pred_train = rf.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred_train)
        train_f1  = f1_score(y_train, y_pred_train, zero_division=0)
        print(f"\n📘 Random Forest Train Accuracy: {train_acc:.4f} | Train F1: {train_f1:.4f}")

        # Validation Performance
        rf_metrics = evaluate_model(rf, X_val, y_val, "Random Forest (v1.2)")

        save_model_info(rf, rf_metrics, "Random Forest", "1.2.2_balanced15", X_train.columns)

        # 4️. LightGBM (tuned)
        lgb = train_lightgbm(X_train, y_train)

        # Train Performance
        y_pred_train = lgb.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred_train)
        train_f1  = f1_score(y_train, y_pred_train, zero_division=0)
        print(f"\n📘 LightGBM Train Accuracy: {train_acc:.4f} | Train F1: {train_f1:.4f}")

        # Validation Performance
        lgb_metrics = evaluate_model(lgb, X_val, y_val, "LightGBM (v1.4)")

        save_model_info(lgb, lgb_metrics, "LightGBM", "1.4.2_balanced15", X_train.columns)

        print("\n" + "="*40)
        print("🏁 COMPARISON SUMMARY (Best F1):")
        print(f"🔹 Logistic Regression: {lr_metrics['f1_score']:.4f}")
        print(f"🔹 XGBoost:             {xgb_metrics['f1_score']:.4f}")
        print(f"🔹 Random Forest:       {rf_metrics['f1_score']:.4f}")
        print(f"🔹 LightGBM:            {lgb_metrics['f1_score']:.4f}")
        print("="*40)

        train_f1_lr  = f1_score(y_train, lr.predict(X_train), zero_division=0)
        train_f1_xgb = f1_score(y_train, xgb.predict(X_train), zero_division=0)
        train_f1_rf  = f1_score(y_train, rf.predict(X_train), zero_division=0)
        train_f1_lgb = f1_score(y_train, lgb.predict(X_train), zero_division=0)

        summary = pd.DataFrame([
            ["Logistic Regression", train_f1_lr,  lr_metrics["f1_score"]],
            ["XGBoost",             train_f1_xgb, xgb_metrics["f1_score"]],
            ["Random Forest",       train_f1_rf,  rf_metrics["f1_score"]],
            ["LightGBM",            train_f1_lgb, lgb_metrics["f1_score"]],
        ], columns=["Model", "Train_F1", "Val_F1"])

        summary["Gap"] = summary["Train_F1"] - summary["Val_F1"]

        print("\n📊 F1 Comparison Table:")
        print(summary.round(4))

        # 5️. Calibration (User Recommended)
        # Choosing the best model based on Val_F1 for calibration
        best_model_name = summary.loc[summary['Val_F1'].idxmax(), 'Model']
        print(f"\n🏆 Best model for calibration: {best_model_name}")
        
        best_models_map = {
            "Logistic Regression": lr,
            "XGBoost": xgb,
            "Random Forest": rf,
            "LightGBM": lgb
        }
        raw_best_model = best_models_map[best_model_name]
        
        # We use Isotonic as default (better if N > 1000)
        calibrated_model = calibrate_best_model(raw_best_model, X_val, y_val, best_model_name, method='isotonic')
        
        # Evaluate Calibrated Model
        calib_metrics = evaluate_model(calibrated_model, X_val, y_val, f"{best_model_name} (Calibrated v1.5)")
        
        # Save Calibrated Model
        save_model_info(calibrated_model, calib_metrics, f"{best_model_name} Calibrated", "1.5_calibrated", X_train.columns)
        
        # Save explicitly as the "Final" model for easy access
        final_calibrated_path = MODELS_DIR / "model_target_hit_final_calibrated.pkl"
        joblib.dump(calibrated_model, final_calibrated_path)
        print(f"🎯 FINAL CALIBRATED MODEL saved separately to: {final_calibrated_path}")

        print("\n" + "="*40)
        print("🎉 Calibration Phase Complete! Model is now ready for Test Evaluation.")
        print("="*40)

    except Exception as e:
        print(f"❌ Error in training pipeline: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()