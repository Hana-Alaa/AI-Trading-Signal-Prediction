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
    train_path = splits_dir / "train_stop_hit.csv"
    valid_path = splits_dir / "valid_stop_hit.csv"
    
    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError("Split files not found in data/processed/splits/. Run Phase 4 first.")
        
    # Using low_memory=False to handle mixed types warnings if any
    train = pd.read_csv(train_path, low_memory=False)
    valid = pd.read_csv(valid_path, low_memory=False)
    print(f"Loaded Splits - Train: {train.shape}, Valid: {valid.shape}")
    return train, valid

def prepare_data(df, target_col='stop_hit', reference_features=None):
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
    print("⏳ Training Logistic Regression (Basic v1.0)...")
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    print("✅ Logistic Regression Training Complete.")
    return model

def train_logistic_regression_tuned(X_train, y_train):
    print("\n" + "="*40)
    print("⏳ Training Logistic Regression with Tuning (v1.5)...")
    base_model = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
    search = RandomizedSearchCV(base_model, param_grid, n_iter=10, scoring='f1', cv=3, random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    print("\n✅ Best LR Params:", search.best_params_)
    return search.best_estimator_

def train_random_forest(X_train, y_train):
    """Train Random Forest with simple tuning."""
    print("\n" + "="*40)
    print("⏳ Training Random Forest (Basic v1.2)...")

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

def train_random_forest_tuned(X_train, y_train):
    print("\n" + "="*40)
    print("⏳ Training Random Forest with Random Search (v1.6)...")
    base_rf = RandomForestClassifier(class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    param_dist = {
        "n_estimators": [200, 400, 600],
        "max_depth": [6, 12, 18, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False]
    }
    search = RandomizedSearchCV(base_rf, param_dist, n_iter=10, scoring="f1", cv=3, random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    print("\n✅ Best RF Params:", search.best_params_)
    return search.best_estimator_

def train_lightgbm_basic(X_train, y_train):
    """Train LightGBM model with basic/default hyperparameters (no tuning)."""
    print("\n" + "="*40)
    print("⏳ Training LightGBM (Basic v1.3)...")

    count_neg = (y_train == 0).sum()
    count_pos = (y_train == 1).sum()
    scale_pos_weight = count_neg / (count_pos + 1e-9)

    lgb_model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
    lgb_model.fit(X_train, y_train)
    print("✅ LightGBM (Basic) Training Complete.")
    return lgb_model

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

def train_xgboost_basic(X_train, y_train):
    print("\n" + "="*40)
    print("⏳ Training XGBoost (Basic v1.7)...")
    count_neg = (y_train == 0).sum()
    count_pos = (y_train == 1).sum()
    scale_weight = count_neg / (count_pos + 1e-9)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=scale_weight
    )
    model.fit(X_train, y_train)
    print("✅ XGBoost Basic Training Complete.")
    return model

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
    print("✅ XGBoost Tuning Complete.")
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

    # Confusion Matrix Console Output
    cm = confusion_matrix(y_val, y_pred_best)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n🧩 Confusion Matrix (Threshold={best_thresh:.2f}):")
    print(f"   TN: {tn:<8} FP: {fp}")
    print(f"   FN: {fn:<8} TP: {tp}")

    # Plot Confusion Matrix
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"Confusion Matrix – {model_label} (thr={best_thresh:.2f})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # Classification Report
    print("\n📋 Classification Report Summary:")
    report = classification_report(y_val, y_pred_best, zero_division=0, digits=4, output_dict=True)
    print(classification_report(y_val, y_pred_best, zero_division=0, digits=4))

    # Calculate Train metrics more formally if provided
    train_metrics = {}
    if X_train is not None and y_train is not None:
        y_train_prob = model.predict_proba(X_train)[:, 1]
        y_train_pred = (y_train_prob >= best_thresh).astype(int)
        train_metrics = {
            "accuracy": accuracy_score(y_train, y_train_pred),
            "precision": precision_score(y_train, y_train_pred, zero_division=0),
            "recall": recall_score(y_train, y_train_pred, zero_division=0),
            "f1_score": f1_score(y_train, y_train_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_train, y_train_prob)
        }

    return {
        "val": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc,
            "best_threshold": float(best_thresh)
        },
        "train": train_metrics
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
    model_filename = f"model_stop_hit_v{version}.pkl"
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

    if 'stop_hit_versions' not in metadata:
        metadata['stop_hit_versions'] = []

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
    metadata['stop_hit_versions'] = [v for v in metadata['stop_hit_versions'] if v['version'] != version]
    metadata['stop_hit_versions'].append(version_entry)
    metadata['stop_hit_model'] = version_entry

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

        # 1. Logistic Regression
        lr = train_logistic_regression(X_train, y_train)
        lr_metrics = evaluate_model(lr, X_val, y_val, "LR Basic (v1.0)", X_train, y_train)
        save_model_info(lr, lr_metrics["val"], "Logistic Regression", "1.2.2_balanced15", feature_list)

        lr_tuned = train_logistic_regression_tuned(X_train, y_train)
        lr_tuned_metrics = evaluate_model(lr_tuned, X_val, y_val, "LR Tuned (v1.5)", X_train, y_train)
        save_model_info(lr_tuned, lr_tuned_metrics["val"], "Logistic Regression Tuned", "1.5.2_balanced15", feature_list)

        # 2. XGBoost
        xgb_basic = train_xgboost_basic(X_train, y_train)
        xgb_basic_metrics = evaluate_model(xgb_basic, X_val, y_val, "XGB Basic (v1.7)", X_train, y_train)
        save_model_info(xgb_basic, xgb_basic_metrics["val"], "XGBoost Basic", "1.7.2_balanced15", feature_list)

        xgb = train_xgboost(X_train, y_train)
        xgb_metrics = evaluate_model(xgb, X_val, y_val, "XGB Tuned (v1.1)", X_train, y_train)
        save_model_info(xgb, xgb_metrics["val"], "XGBoost", "1.2.2_balanced15", feature_list)

        # 3. Random Forest
        rf = train_random_forest(X_train, y_train)
        rf_metrics = evaluate_model(rf, X_val, y_val, "RF Basic (v1.2)", X_train, y_train)
        save_model_info(rf, rf_metrics["val"], "Random Forest", "1.2.2_balanced15", feature_list)

        rf_tuned = train_random_forest_tuned(X_train, y_train)
        rf_tuned_metrics = evaluate_model(rf_tuned, X_val, y_val, "RF Tuned (v1.6)", X_train, y_train)
        save_model_info(rf_tuned, rf_tuned_metrics["val"], "Random Forest Tuned", "1.6.2_balanced15", feature_list)

        # 4. LightGBM
        lgb_basic = train_lightgbm_basic(X_train, y_train)
        lgb_basic_metrics = evaluate_model(lgb_basic, X_val, y_val, "LGB Basic (v1.3)", X_train, y_train)
        save_model_info(lgb_basic, lgb_basic_metrics["val"], "LightGBM Basic", "1.3_balanced15", feature_list)

        lgb = train_lightgbm(X_train, y_train)
        lgb_metrics = evaluate_model(lgb, X_val, y_val, "LGB Tuned (v1.4)", X_train, y_train)
        save_model_info(lgb, lgb_metrics["val"], "LightGBM", "1.4.2_balanced15", feature_list)

        print("\n" + "="*40)
        print("🏁 COMPARISON SUMMARY:")
        print("="*40)
        
        results_list = []
        for name, m in [
            ("LR Basic", lr_metrics),
            ("LR Tuned", lr_tuned_metrics),
            ("XGB Basic", xgb_basic_metrics),
            ("XGB Tuned", xgb_metrics),
            ("RF Basic", rf_metrics),
            ("RF Tuned", rf_tuned_metrics),
            ("LGB Basic", lgb_basic_metrics),
            ("LGB Tuned", lgb_metrics)
        ]:
            row = {
                "Model": name,
                "Train_Acc": m["train"].get("accuracy", 0),
                "Train_Prec": m["train"].get("precision", 0),
                "Train_Rec": m["train"].get("recall", 0),
                "Train_F1": m["train"].get("f1_score", 0),
                "Val_Acc": m["val"]["accuracy"],
                "Val_Prec": m["val"]["precision"],
                "Val_Rec": m["val"]["recall"],
                "Val_F1": m["val"]["f1_score"],
                "Gap_F1": m["train"].get("f1_score", 0) - m["val"]["f1_score"]
            }
            results_list.append(row)

        summary = pd.DataFrame(results_list)
        print(summary.round(4).to_string(index=False))

        # Save detailed summary to file
        log_dir = project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        summary_path = log_dir / "stop_hit_model_training_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\n📊 Detailed training summary saved to: {summary_path}")

        # 5. Calibration (Probability Alignment)
        # Note: We calibrate on X_val. Testing the calibrated model on X_val and Gap again 
        # will yield optimistic results. Final judgment must be on the Test set.
        stable_models = summary[summary['Gap_F1'].abs() <= 0.05]
        if not stable_models.empty:
            best_model_name = stable_models.loc[stable_models['Val_F1'].idxmax(), 'Model']
        else:
            best_model_name = summary.loc[summary['Val_F1'].idxmax(), 'Model']

        print(f"\n🏆 Smart selection for calibration: {best_model_name}")
        
        best_models_map = {
            "LR Basic": lr,
            "LR Tuned": lr_tuned,
            "XGB Basic": xgb_basic,
            "XGB Tuned": xgb,
            "RF Basic": rf,
            "RF Tuned": rf_tuned,
            "LGB Basic": lgb_basic,
            "LGB Tuned": lgb
        }
        raw_best_model = best_models_map[best_model_name]
        
        calibrated_model = calibrate_best_model(raw_best_model, X_val, y_val, best_model_name, method='isotonic')
        
        # Evaluate Calibrated Model (INTERNAL REFERENCE ONLY)
        print("\nNOTICE: The following Validation metrics are OPTIMISTIC (calibrated on the same data).")
        print("Final unbiased performance must be measured using 'tests/test_model_performance.py' on the Test set.")
        calib_metrics = evaluate_model(calibrated_model, X_val, y_val, f"{best_model_name} (Calibrated v1.5)")
        
        # Save Calibrated Model
        save_model_info(calibrated_model, calib_metrics["val"], f"{best_model_name} Calibrated", "1.5_calibrated", feature_list)
        
        # Save explicitly as the "Final" model for easy access
        final_calibrated_path = MODELS_DIR / "model_stop_hit_final_calibrated.pkl"
        joblib.dump(calibrated_model, final_calibrated_path)
        print(f"FINAL CALIBRATED MODEL saved separately to: {final_calibrated_path}")

        print("\n" + "="*40)
        print("Calibration Phase Complete! Model is now ready for Test Evaluation.")
        print("="*40)

    except Exception as e:
        print(f"Error in training pipeline: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()