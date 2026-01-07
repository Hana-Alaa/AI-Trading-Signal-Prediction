import pandas as pd
import numpy as np
import sys
import joblib
import json
import traceback
import argparse
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping as lgb_early_stopping, log_evaluation as lgb_log_evaluation

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, average_precision_score
)
from sklearn.isotonic import IsotonicRegression

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from config import PROCESSED_DATA_DIR, MODELS_DIR, CREATED_BY 

RANDOM_STATE = 42

# ======================================
# Data loading
# ======================================
def load_splits(target: str):
    splits_dir = PROCESSED_DATA_DIR / f"splits_{target}"
    train_path = splits_dir / "train.csv"
    valid_path = splits_dir / "valid.csv"

    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError(
            f"Split files not found in {splits_dir}. Run Phase 4 target-aware preprocessing first."
        )

    train = pd.read_csv(train_path, low_memory=False)
    valid = pd.read_csv(valid_path, low_memory=False)
    print(f"✅ Loaded Splits [{target}] - Train: {train.shape}, Valid: {valid.shape}")
    return train, valid

def prepare_data(df, target_col: str, reference_features=None):
    exclude_cols = [
        "id", "status", "target_hit", "stop_hit", "target_type",
        "hit_first", "created_at", "coin", "TP1"
    ]
    X = df.drop(columns=[c for c in exclude_cols if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=np.number).copy()

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    y = df[target_col].astype(int)

    if reference_features is not None:
        missing_cols = [c for c in reference_features if c not in X.columns]
        extra_cols = [c for c in X.columns if c not in reference_features]

        for col in missing_cols:
            X[col] = 0
        if extra_cols:
            X = X.drop(columns=extra_cols)

        X = X[reference_features]

    return X, y

# ======================================
# Target-aware helpers
# ======================================
def _scale_pos_weight(y_train, target: str) -> float:
    """
    - stop_hit: positives rare => neg/pos helps
    - target_hit: positives are majority => don't downweight positives (use 1.0)
    """
    if target == "target_hit":
        return 1.0
    count_neg = (y_train == 0).sum()
    count_pos = (y_train == 1).sum()
    return float(count_neg / (count_pos + 1e-9))

def _default_rank_metric(target: str) -> str:
    # stop_hit: PR-AUC best for rare positives
    # target_hit: ROC-AUC more informative when positives are majority
    return "pr_auc" if target == "stop_hit" else "roc_auc"

# ======================================
# Models
# ======================================
def train_logistic_regression(X_train, y_train):
    print("\n" + "=" * 40)
    print("⏳ Training Logistic Regression (Basic) [Scaled Pipeline]...")
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced")),
        ]
    )
    model.fit(X_train, y_train)
    print("✅ Logistic Regression Training Complete.")
    return model

def train_logistic_regression_tuned(X_train, y_train, cv):
    print("\n" + "=" * 40)
    print("⏳ Training Logistic Regression with Tuning [Scaled Pipeline + TimeSeriesSplit]...")

    base_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE, class_weight="balanced")),
        ]
    )

    param_grid = {
        "lr__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "lr__penalty": ["l1", "l2"],
        "lr__solver": ["liblinear", "saga"],
    }

    search = RandomizedSearchCV(
        base_model,
        param_grid,
        n_iter=10,
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        error_score="raise",
        verbose=0,
    )
    search.fit(X_train, y_train)
    print("\n✅ Best LR Params:", search.best_params_)
    return search.best_estimator_

def train_random_forest(X_train, y_train):
    print("\n" + "=" * 40)
    print("⏳ Training Random Forest (Basic)...")
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    print("✅ Random Forest Training Complete.")
    return rf

def train_random_forest_tuned(X_train, y_train, cv):
    print("\n" + "=" * 40)
    print("⏳ Training Random Forest with Random Search [TimeSeriesSplit]...")
    base_rf = RandomForestClassifier(class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1)
    param_dist = {
        "n_estimators": [300, 500, 800],
        "max_depth": [6, 10, 14, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4, 6],
        "bootstrap": [True, False],
        "max_features": ["sqrt", "log2", None],
    }
    search = RandomizedSearchCV(
        base_rf,
        param_dist,
        n_iter=12,
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        error_score="raise",
        verbose=0,
    )
    search.fit(X_train, y_train)
    print("\n✅ Best RF Params:", search.best_params_)
    return search.best_estimator_

def train_lightgbm_basic(X_train, y_train, X_val, y_val, target: str):
    print("\n" + "=" * 40)
    print("⏳ Training LightGBM (Basic) with Early Stopping...")
    spw = _scale_pos_weight(y_train, target)
    model = LGBMClassifier(
        objective="binary",
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb_early_stopping(100), lgb_log_evaluation(0)],
    )
    print("✅ LightGBM (Basic) Training Complete.")
    return model

def train_lightgbm_tuned(X_train, y_train, X_val, y_val, cv, target: str):
    print("\n" + "=" * 40)
    print("⏳ Training LightGBM with Random Search [TimeSeriesSplit + Early Stopping]...")
    spw = _scale_pos_weight(y_train, target)
    base_model = LGBMClassifier(
        objective="binary",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=spw,
        n_estimators=3000,
    )
    param_dist = {
        "learning_rate": [0.02, 0.03, 0.05],
        "num_leaves": [31, 45, 63],
        "max_depth": [5, 7, 9, -1],
        "min_child_samples": [15, 25, 40],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "reg_alpha": [0.0, 0.05, 0.1],
        "reg_lambda": [0.0, 0.05, 0.1],
    }

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=12,
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        error_score="raise",
        verbose=0,
    )
    search.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb_early_stopping(100), lgb_log_evaluation(0)],
    )
    print("\n✅ Best LGB Params:", search.best_params_)
    return search.best_estimator_

# XGBoost (compat mode)
def train_xgboost_basic(X_train, y_train, target: str):
    print("\n" + "=" * 40)
    print("⏳ Training XGBoost (Basic) [NO early stopping - compatibility mode]...")
    spw = _scale_pos_weight(y_train, target)
    model = XGBClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        eval_metric="auc",
        scale_pos_weight=spw,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("✅ XGBoost (Basic) Training Complete.")
    return model

def train_xgboost_tuned(X_train, y_train, cv, target: str):
    print("\n" + "=" * 40)
    print("⏳ Training XGBoost with Random Search [TimeSeriesSplit, NO early stopping]...")
    spw = _scale_pos_weight(y_train, target)
    base_model = XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="auc",
        scale_pos_weight=spw,
        n_jobs=-1,
    )
    param_grid = {
        "n_estimators": [300, 600, 900],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.03, 0.05],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.2],
        "reg_lambda": [0.0, 0.5, 1.0],
    }
    tuner = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=10,
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        error_score="raise",
        verbose=0,
    )
    tuner.fit(X_train, y_train)
    print("\n✅ Best XGB Params:", tuner.best_params_)
    return tuner.best_estimator_

# ======================================
# Evaluation helpers
# ======================================
def _tune_threshold_for_f1(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    best = {"f1": -1.0, "threshold": 0.5}
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best["f1"]:
            best = {"f1": float(f1), "threshold": float(t)}
    return best

def _tune_threshold_for_balanced_accuracy(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    best = {"score": -1.0, "threshold": 0.5}
    y_true = np.asarray(y_true)

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))

        tpr = tp / (tp + fn + 1e-9)  # recall class 1
        tnr = tn / (tn + fp + 1e-9)  # recall class 0 (specificity)
        bal_acc = 0.5 * (tpr + tnr)

        if bal_acc > best["score"]:
            best = {"score": float(bal_acc), "threshold": float(t)}

    return best

def _tune_threshold_for_precision_target(y_true, y_prob, precision_target=0.35, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    best = {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if p >= precision_target and r > best["recall"]:
            best = {"threshold": float(t), "precision": float(p), "recall": float(r), "f1": float(f1)}

    return best

def evaluate_threshold_free(model, X, y):
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y, y_prob)),
        "pr_auc": float(average_precision_score(y, y_prob)),
    }

def _metrics_at_threshold(y_true, y_prob, thr: float):
    y_pred = (y_prob >= thr).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }, y_pred

def precision_at_k(y_true, y_scores, k_ratio=0.1):
    """
    Precision@TopK%
    - y_scores: predicted probabilities
    - k_ratio: percentage (0.05, 0.1, 0.2)
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    k = int(len(y_true) * k_ratio)
    if k <= 0:
        return 0.0

    top_idx = np.argsort(y_scores)[::-1][:k]
    y_top = y_true[top_idx]

    return float(np.mean(y_top))

def evaluate_full(model, X_train, y_train, X_val, y_val, model_label, target,
                  show_plots=False, precision_target=0.35):
    print(f"\n📊 Evaluation for {model_label}")

    prob_val = model.predict_proba(X_val)[:, 1]
    prob_train = model.predict_proba(X_train)[:, 1]

    tf_val = {
        "roc_auc": roc_auc_score(y_val, prob_val),
        "pr_auc": average_precision_score(y_val, prob_val),
    }
    tf_tr = {
        "roc_auc": roc_auc_score(y_train, prob_train),
        "pr_auc": average_precision_score(y_train, prob_train),
    }

    # ---------- Precision@TopK ----------
    p_at_5 = precision_at_k(y_val, prob_val, 0.05)
    p_at_10 = precision_at_k(y_val, prob_val, 0.10)
    p_at_20 = precision_at_k(y_val, prob_val, 0.20)

    print(
        f"🎯 Precision@K → "
        f"P@5%={p_at_5:.3f} | P@10%={p_at_10:.3f} | P@20%={p_at_20:.3f}"
    )

    # ---------- target-aware threshold tuning ----------
    if target == "stop_hit":
        tune = _tune_threshold_for_precision_target(
            y_val, prob_val, precision_target=precision_target
        )
        thr = tune["threshold"]
        print(
            f"🎯 Threshold by Precision≥{precision_target:.2f}: "
            f"thr={thr:.2f} | P={tune['precision']:.3f} "
            f"R={tune['recall']:.3f} F1={tune['f1']:.3f}"
        )
    else:
        tune = _tune_threshold_for_balanced_accuracy(y_val, prob_val)
        thr = tune["threshold"]
        print(
            f"⚖️ Threshold by Balanced Accuracy: "
            f"thr={thr:.2f} | bal_acc={tune['score']:.3f}"
        )

    m_val, pred_val = _metrics_at_threshold(y_val, prob_val, thr)
    m_tr, pred_tr = _metrics_at_threshold(y_train, prob_train, thr)

    print(
        f"VAL(threshold-free) → ROC-AUC: {tf_val['roc_auc']:.4f} "
        f"| PR-AUC: {tf_val['pr_auc']:.4f}"
    )
    print(
        f"TRAIN(threshold-free) → ROC-AUC: {tf_tr['roc_auc']:.4f} "
        f"| PR-AUC: {tf_tr['pr_auc']:.4f}"
    )
    print(
        f"VAL@thr → Acc: {m_val['accuracy']:.4f} "
        f"Prec: {m_val['precision']:.4f} "
        f"Rec: {m_val['recall']:.4f} "
        f"F1: {m_val['f1_score']:.4f}"
    )
    print(
        f"TRAIN@thr → Acc: {m_tr['accuracy']:.4f} "
        f"Prec: {m_tr['precision']:.4f} "
        f"Rec: {m_tr['recall']:.4f} "
        f"F1: {m_tr['f1_score']:.4f}"
    )
    print(f"⚖️ Gap F1 (Train−Val): {m_tr['f1_score'] - m_val['f1_score']:.4f}")

    cm = confusion_matrix(y_val, pred_val)
    tn, fp, fn, tp = cm.ravel()
    print(f"🧩 Confusion Matrix (VAL, thr={thr:.2f}): TN={tn} FP={fp} FN={fn} TP={tp}")
    print("\n📋 Classification Report (VAL):")
    print(classification_report(y_val, pred_val, zero_division=0, digits=4))

    if show_plots:
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix – {model_label} (VAL thr={thr:.2f})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()

    return {
        "best_threshold": float(thr),
        "train": {
            **{k: float(v) for k, v in m_tr.items()},
            "roc_auc": float(tf_tr["roc_auc"]),
            "pr_auc": float(tf_tr["pr_auc"]),
        },
        "val": {
            **{k: float(v) for k, v in m_val.items()},
            "roc_auc": float(tf_val["roc_auc"]),
            "pr_auc": float(tf_val["pr_auc"]),
            "precision_at_5": float(p_at_5),
            "precision_at_10": float(p_at_10),
            "precision_at_20": float(p_at_20),
            "best_threshold": float(thr),
        },
    }

# ======================================
# Calibration + Saving
# ======================================
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

def calibrate_best_model(model, X_val, y_val, model_label):
    print("\n" + "=" * 40)
    print(f"⏳ Calibrating {model_label} using Isotonic Regression (fit on VAL)...")
    y_prob = model.predict_proba(X_val)[:, 1]
    iso_reg = IsotonicRegression(out_of_bounds="clip")
    iso_reg.fit(y_prob, y_val)
    print("✅ Calibration complete.")
    return CalibratedModelWrapper(model, iso_reg)

def save_model_info(
    model,
    metrics_pack,
    algorithm_name,
    version,
    feature_names,
    target: str,
    set_as_current: bool = False,
):
    """Save model + update metadata.json. Set current model only if set_as_current=True."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_filename = f"model_{target}_v{version}.pkl"
    model_path = MODELS_DIR / model_filename
    joblib.dump(model, model_path)
    print(f"💾 Saved model: {model_path}")

    metadata_path = MODELS_DIR / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}

    versions_key = f"{target}_versions"
    current_key = f"{target}_model"

    if versions_key not in metadata:
        metadata[versions_key] = []

    version_entry = {
        "version": version,
        "algorithm": algorithm_name,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "created_by": CREATED_BY,
        "metrics_val": metrics_pack["val"],
        "metrics_train": metrics_pack.get("train", {}),
        "feature_count": int(len(feature_names)),
        "features": list(feature_names),
        "model_path": str(model_filename),
    }

    metadata[versions_key] = [v for v in metadata[versions_key] if v.get("version") != version]
    metadata[versions_key].append(version_entry)

    if set_as_current:
        metadata[current_key] = version_entry
        print(f"🏷️ Set CURRENT model for {target}: {algorithm_name} (v{version})")

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"📖 Metadata updated: {versions_key}" + (f", {current_key}" if set_as_current else ""))

# ======================================
# MAIN
# ======================================
def main():
    parser = argparse.ArgumentParser(description="Train models (target-aware, time-series safe)")
    parser.add_argument("--target", type=str, default="target_hit", choices=["target_hit", "stop_hit"])
    parser.add_argument("--calibrate", type=str, default="off", choices=["on", "off"])
    parser.add_argument("--plots", type=str, default="off", choices=["on", "off"])

    # rank-metric optional (None => auto)
    parser.add_argument("--rank-metric", type=str, default=None, choices=["pr_auc", "roc_auc", None])

    # used for stop_hit thresholding only
    parser.add_argument("--precision-target", type=float, default=0.35)

    # stability filter for choosing winner as "current model"
    parser.add_argument("--gap-limit", type=float, default=0.08)

    args = parser.parse_args()

    target = args.target
    do_calibrate = (args.calibrate == "on")
    show_plots = (args.plots == "on")

    rank_metric = _default_rank_metric(target) if args.rank_metric is None else args.rank_metric
    precision_target = float(args.precision_target)
    gap_limit = float(args.gap_limit)

    try:
        train_df, valid_df = load_splits(target=target)

        X_train, y_train = prepare_data(train_df, target_col=target)
        feature_list = X_train.columns.tolist()
        X_val, y_val = prepare_data(valid_df, target_col=target, reference_features=feature_list)

        print(f"🔎 Feature count: {X_train.shape[1]}")
        print(f"🏷️  rank_metric={rank_metric} | threshold_mode={'precision_target' if target=='stop_hit' else 'balanced_accuracy'}")

        tscv = TimeSeriesSplit(n_splits=3)

        models = {
            "LR Basic": train_logistic_regression(X_train, y_train),
            "LR Tuned": train_logistic_regression_tuned(X_train, y_train, cv=tscv),
            "XGB Basic": train_xgboost_basic(X_train, y_train, target=target),
            "XGB Tuned": train_xgboost_tuned(X_train, y_train, cv=tscv, target=target),
            "RF Basic": train_random_forest(X_train, y_train),
            "RF Tuned": train_random_forest_tuned(X_train, y_train, cv=tscv),
            "LGB Basic": train_lightgbm_basic(X_train, y_train, X_val, y_val, target=target),
            "LGB Tuned": train_lightgbm_tuned(X_train, y_train, X_val, y_val, cv=tscv, target=target),
        }

        rows = []
        eval_cache = {}

        for name, model in models.items():
            metrics_pack = evaluate_full(
                model, X_train, y_train, X_val, y_val,
                model_label=name,
                target=target,
                show_plots=show_plots,
                precision_target=precision_target,
            )
            eval_cache[name] = metrics_pack

            # Save each model as a version 
            save_model_info(
                model,
                metrics_pack,
                algorithm_name=name,
                version=name.lower().replace(" ", "_"),
                feature_names=feature_list,
                target=target,
                set_as_current=False,
            )

            rows.append({
                "Model": name,
                "Rank_Val_ROC_AUC": metrics_pack["val"]["roc_auc"],
                "Rank_Val_PR_AUC": metrics_pack["val"]["pr_auc"],

                "P@5%": metrics_pack["val"]["precision_at_5"],
                "P@10%": metrics_pack["val"]["precision_at_10"],
                "P@20%": metrics_pack["val"]["precision_at_20"],

                "Val_Acc@Thr": metrics_pack["val"]["accuracy"],
                "Val_Prec@Thr": metrics_pack["val"]["precision"],
                "Val_Rec@Thr": metrics_pack["val"]["recall"],
                "Val_F1@Thr": metrics_pack["val"]["f1_score"],

                "BestThr(VAL)": metrics_pack["val"]["best_threshold"],

                "Train_Acc@Thr": metrics_pack["train"]["accuracy"],
                "Train_Prec@Thr": metrics_pack["train"]["precision"],
                "Train_Rec@Thr": metrics_pack["train"]["recall"],
                "Train_F1@Thr": metrics_pack["train"]["f1_score"],

                "Gap_F1": metrics_pack["train"]["f1_score"] - metrics_pack["val"]["f1_score"],
            })

        summary_df = pd.DataFrame(rows)
        sort_col = "Rank_Val_PR_AUC" if rank_metric == "pr_auc" else "Rank_Val_ROC_AUC"
        summary_df = summary_df.sort_values(sort_col, ascending=False)

        print("\n" + "=" * 40)
        print(f"🏁 SUMMARY [{target}] (rank_metric={rank_metric})")
        print("=" * 40)
        print(summary_df.round(4).to_string(index=False))

        logs_dir = project_root / "logs"
        logs_dir.mkdir(exist_ok=True)
        summary_path = logs_dir / f"{target}_model_training_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n📊 Summary saved to: {summary_path}")

        # -------------------------
        # Pick WINNER with stability (gap filter), then set as current once
        # -------------------------
        stable_df = summary_df[summary_df["Gap_F1"].abs() <= gap_limit].copy()
        if not stable_df.empty:
            winner_row = stable_df.iloc[0]
            print(f"\n✅ Winner chosen from STABLE models (|Gap_F1| <= {gap_limit}): {winner_row['Model']}")
        else:
            winner_row = summary_df.iloc[0]
            print(f"\n⚠️ No stable model under gap limit {gap_limit}. Falling back to top ranked: {winner_row['Model']}")

        winner_name = winner_row["Model"]
        winner_model = models[winner_name]
        winner_version = winner_name.lower().replace(" ", "_")
        winner_metrics = eval_cache[winner_name]

        # Set CURRENT model in metadata once
        save_model_info(
            winner_model,
            winner_metrics,
            algorithm_name=winner_name,
            version=winner_version,
            feature_names=feature_list,
            target=target,
            set_as_current=True,
        )

        # Calibration only on winner (if enabled)
        if do_calibrate:
            best_model_name = winner_name
            raw_best_model = models[best_model_name]
            print(f"\n🏆 Calibrating winner model: {best_model_name}")

            calibrated_model = calibrate_best_model(raw_best_model, X_val, y_val, best_model_name)
            calib_metrics = evaluate_full(
                calibrated_model, X_train, y_train, X_val, y_val,
                model_label=f"{best_model_name} (Calibrated)",
                target=target,
                show_plots=show_plots,
                precision_target=precision_target,
            )

            save_model_info(
                calibrated_model,
                calib_metrics,
                algorithm_name=f"{best_model_name} Calibrated",
                version="calibrated",
                feature_names=feature_list,
                target=target,
                set_as_current=True,  # make calibrated model current
            )

            final_calibrated_path = MODELS_DIR / f"model_{target}_final_calibrated.pkl"
            joblib.dump(calibrated_model, final_calibrated_path)
            print(f"✅ FINAL CALIBRATED MODEL saved to: {final_calibrated_path}")
        else:
            print("\n⏭️ Calibration is OFF.")

        print("\n✅ Training pipeline complete.")

    except Exception as e:
        print(f"❌ Error in training pipeline: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()