import pandas as pd
import numpy as np
import sys
import joblib
import json
import traceback
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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

def prepare_data(df, target_col='target_hit'):
    """Separate features and target."""
    # Columns to exclude from training
    exclude_cols = [
        'id', 'status', 'target_hit', 'stop_hit', 'target_type', 
        'hit_first', 'created_at', 'coin', 'TP1'
    ]
    
    X = df.drop(columns=[c for c in exclude_cols if c in df.columns])
    X = X.select_dtypes(include=np.number)
    y = df[target_col]
    
    return X, y

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression as baseline."""
    print("\n" + "="*40)
    print("⏳ Training Logistic Regression (v1.0)...")
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    print("✅ Logistic Regression Training Complete.")
    return model

def train_xgboost(X_train, y_train):
    """Train XGBoost as advanced model."""
    print("\n" + "="*40)
    print("⏳ Training XGBoost (v1.1)...")
    
    # Calculate scale_pos_weight for imbalance
    count_neg = (y_train == 0).sum()
    count_pos = (y_train == 1).sum()
    scale_weight = count_neg / (count_pos + 1e-9)
    
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    print("✅ XGBoost Training Complete.")
    return model

def evaluate_model(model, X_val, y_val, model_label):
    """Evaluate model on validation set with multiple thresholds."""
    print(f"\n📊 Evaluation Results for {model_label}:")
    
    y_prob = model.predict_proba(X_val)[:, 1]
    
    best_f1 = -1
    best_metrics = {}
    
    for t in [0.3, 0.4, 0.5, 0.6]:
        y_pred = (y_prob >= t).astype(int)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        acc = accuracy_score(y_val, y_pred)
        
        print(f"  Threshold {t} -> Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_metrics = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1_score': f1,
                'roc_auc': roc_auc_score(y_val, y_prob)
            }
            
    return best_metrics

def save_model_info(model, metrics, algorithm_name, version, feature_names):
    """Save model file and update metadata.json."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_filename = f"model_target_hit_v{version}.pkl"
    model_path = MODELS_DIR / model_filename
    
    # Save Model
    joblib.dump(model, model_path)
    print(f"💾 Model {version} saved to: {model_path}")
    
    # Update Metadata
    metadata_path = MODELS_DIR / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except json.JSONDecodeError:
            metadata = {}
            
    # Ensure target_hit_versions entry exists
    if 'target_hit_versions' not in metadata:
        metadata['target_hit_versions'] = []
        
    # Add/Update version entry
    version_entry = {
        'version': version,
        'algorithm': algorithm_name,
        'trained_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'created_by': CREATED_BY,
        'metrics_val': metrics,
        'feature_count': len(feature_names),
        'model_path': str(model_filename)
    }
    
    # Remove existing entry if version matches to update
    metadata['target_hit_versions'] = [v for v in metadata['target_hit_versions'] if v['version'] != version]
    metadata['target_hit_versions'].append(version_entry)
    
    # Also update the primary 'target_hit_model' pointer to the latest
    metadata['target_hit_model'] = version_entry
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"📖 Metadata updated for version {version}")

def main():
    try:
        # 1. Load Data
        train, valid = load_splits()
        
        # 2. Prepare X, y
        X_train, y_train = prepare_data(train)
        X_val, y_val = prepare_data(valid)
        
        print(f"Feature count: {X_train.shape[1]}")
        
        # 3. Train Logistic Regression (Baseline v1.0)
        lr_model = train_logistic_regression(X_train, y_train)
        lr_metrics = evaluate_model(lr_model, X_val, y_val, "Logistic Regression (v1.0)")
        save_model_info(lr_model, lr_metrics, "Logistic Regression", "1.0", X_train.columns)
        
        # 4. Train XGBoost (Advanced v1.1)
        xgb_model = train_xgboost(X_train, y_train)
        xgb_metrics = evaluate_model(xgb_model, X_val, y_val, "XGBoost (v1.1)")
        save_model_info(xgb_model, xgb_metrics, "XGBoost", "1.1", X_train.columns)
        
        # Final Summary
        print("\n" + "="*40)
        print("🏁 COMPARISON SUMMARY (Optimal F1):")
        print(f"🔹 Logistic Regression: {lr_metrics['f1_score']:.4f}")
        print(f"🔹 XGBoost:             {xgb_metrics['f1_score']:.4f}")
        print("="*40)
        
    except Exception as e:
        print(f"❌ Error in training pipeline: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
