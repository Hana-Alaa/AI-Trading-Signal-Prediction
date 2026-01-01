"""
Model Training Script: Stop Hit
-------------------------------
This script will:
1. Load Train/Valid splits.
2. Define Features & Target (stop_hit).
3. Train Logistic Regression & XGBoost.
4. Save models to models/ and metrics to metadata.json.
"""
import pandas as pd
import numpy as np
import sys
import joblib
import json
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Add project root to path
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
        
    train = pd.read_csv(train_path)
    valid = pd.read_csv(valid_path)
    print(f"Loaded Splits - Train: {train.shape}, Valid: {valid.shape}")
    return train, valid

def prepare_data(df, target_col='stop_hit'):
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

def train_baseline(X_train, y_train):
    """Train Logistic Regression as baseline."""
    print("Training Logistic Regression Baseline...")
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    print("Baseline Training Complete.")
    return model

def evaluate_model(model, X_val, y_val):
    """Evaluate model on validation set."""
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1_score': f1_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_prob)
    }
    
    print("\nValidation Metrics (Baseline):")
    for k, v in metrics.items():
        print(f"  - {k.capitalize()}: {v:.4f}")
    
    return metrics

def save_model_and_metadata(model, metrics, feature_names):
    """Save model and update metadata.json."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_name = f"model_stop_hit_{MODEL_VERSION}.pkl"
    model_path = MODELS_DIR / model_name
    
    # Save Model
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Update Metadata
    metadata_path = MODELS_DIR / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
    metadata['stop_hit_model'] = {
        'version': MODEL_VERSION,
        'algorithm': 'Logistic Regression (Baseline)',
        'trained_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'created_by': CREATED_BY,
        'metrics_val': metrics,
        'feature_count': len(feature_names),
        'features': feature_names.tolist() if hasattr(feature_names, 'tolist') else list(feature_names)
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata updated: {metadata_path}")

import traceback

def main():
    try:
        # 1. Load Data
        train, valid = load_splits()
        
        # 2. Prepare X, y
        X_train, y_train = prepare_data(train)
        X_val, y_val = prepare_data(valid)
        
        # 3. Train Baseline
        model = train_baseline(X_train, y_train)
        
        # 4. Evaluate
        metrics = evaluate_model(model, X_val, y_val)
        
        # 5. Save
        save_model_and_metadata(model, metrics, X_train.columns)
        
        print("\nmodel_train_stop_hit.py (Baseline) executed successfully!")
        
    except Exception as e:
        print(f"Error in training pipeline: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
