import pandas as pd
import numpy as np
import sys
import joblib
from pathlib import Path
import warnings
from sklearn.preprocessing import StandardScaler

# Add project root to path
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    sys.path.append(str(project_root))

from config import PROCESSED_DATA_DIR, LOGS_DIR, MODELS_DIR

warnings.filterwarnings("ignore")

def load_data():
    """Load the engineered features dataset."""
    path = PROCESSED_DATA_DIR / "step3_features_engineered.csv"
    if not path.exists():
        raise FileNotFoundError(f"❌ File not found: {path} - Please run Phase 3 first.")
    df = pd.read_csv(path)
    print(f"✅ Loaded data: {df.shape}")
    return df

def clean_numerical_residues(df):
    """
    Step 4.1: Clean Numerical Residues
    - Replace Inf/-Inf with NaN
    - Fill NaN with Median
    """
    print("⏳ Cleaning numerical residues...")
    df = df.copy()
    
    # Replace Inf with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Fill NaN with Median
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            
    print("✅ Cleaned Inf/NaN values.")
    print(f"🔹 Shape after this step: {df.shape}")
    return df

def perform_time_based_split(df):
    """
    Step 4.4: Time-Based Split (70/15/15)
    - Sequential split (NO shuffling)
    """
    print("⏳ Performing Time-Based Split...")
    
    # Sort by time just in case, though it should be chronological
    if 'created_at' in df.columns:
        df = df.sort_values('created_at').reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)
    
    train = df.iloc[:train_end].copy()
    valid = df.iloc[train_end:valid_end].copy()
    test = df.iloc[valid_end:].copy()
    
    # Assertion to ensure no data loss
    assert len(train) + len(valid) + len(test) == len(df), "❌ Split size mismatch!"
    
    print(f"✅ Split Ratios: Train={len(train)/n:.1%}, Valid={len(valid)/n:.1%}, Test={len(test)/n:.1%}")
    return train, valid, test

def clip_outliers(train, valid, test):
    """
    Step 4.2: Outlier Clipping (Post-Split)
    - Calculate limits on TRAIN only.
    - Apply to Train, Valid, Test.
    """
    print("⏳ Clipping outliers (1st-99th percentile) [Fit on Train]...")
    
    # Exclude targets and non-numeric cols from clipping
    exclude_cols = ['id', 'status', 'target_hit', 'stop_hit', 'target_type', 'hit_first']
    numeric_cols = train.select_dtypes(include=[np.number]).columns
    cols_to_clip = [c for c in numeric_cols if c not in exclude_cols]
    
    for col in cols_to_clip:
        # Calculate limits on TRAIN
        lower = train[col].quantile(0.01)
        upper = train[col].quantile(0.99)
        
        # Apply to ALL splits
        train[col] = train[col].clip(lower, upper)
        valid[col] = valid[col].clip(lower, upper)
        test[col] = test[col].clip(lower, upper)
        
    print(f"✅ Clipped outliers for {len(cols_to_clip)} features.")
    return train, valid, test

def perform_feature_scaling(train, valid, test):
    """
    Step 4.3: Feature Scaling (StandardScaler)
    - Fit on TRAIN only
    - Transform Train, Valid, Test
    - Save Scaler
    """
    print("⏳ Scaling Features (StandardScaler)...")
    
    # Identify feature columns (numeric excluding targets/metadata)
    exclude_cols = ['id', 'status', 'target_hit', 'stop_hit', 'target_type', 'hit_first', 'created_at', 'coin']
    
    # Get numeric columns that are in the dataframe AND not excluded
    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    if not feature_cols:
        print("⚠️ No features to scale!")
        return train, valid, test

    # Initialize Scaler
    scaler = StandardScaler()
    
    # FIT on Train
    scaler.fit(train[feature_cols])
    
    # TRANSFORM all
    train[feature_cols] = scaler.transform(train[feature_cols])
    valid[feature_cols] = scaler.transform(valid[feature_cols])
    test[feature_cols] = scaler.transform(test[feature_cols])
    
    # Save Scaler
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    scaler_path = MODELS_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    
    print(f"✅ Scaled {len(feature_cols)} features.")
    print(f"💾 Scaler saved to: {scaler_path}")
    
    # Check for Infs after scaling (can happen if std is very small)
    for split_name, split_df in zip(["train", "valid", "test"], [train, valid, test]):
        if np.isinf(split_df.select_dtypes(include=[np.number]).values).any():
            print(f"⚠️ Found Inf values after scaling in {split_name}, replacing with 0.")
            split_df.replace([np.inf, -np.inf], 0, inplace=True)
    
    return train, valid, test

def check_data_drift(train, test):
    """
    Data Drift Check
    - Compare mean and std between Train and Test
    - Save report
    """
    print("⏳ Checking for Data Drift...")
    
    # Exclude non-features from drift check
    exclude_cols = ['id', 'status', 'target_hit', 'stop_hit', 'target_type', 'hit_first', 'created_at', 'coin', 'TP1']
    numeric_cols = train.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    drift_report = []
    
    for col in feature_cols:
        train_std = train[col].std()
        
        # Skip Low Variance columns (to avoid divide by zero or noise)
        if train_std < 1e-6:
            # print(f"⚠️ Skipping {col} (low variance needs check)") 
            continue
            
        train_mean = train[col].mean()
        test_mean = test[col].mean()
        test_std = test[col].std()
        
        # Simple Z-test like metric for drift magnitude
        mean_diff = abs(train_mean - test_mean)
        # Avoid division by zero
        std_denom = train_std if train_std != 0 else 1e-9
        drift_score = mean_diff / std_denom
        
        drift_report.append({
            'feature': col,
            'train_mean': train_mean,
            'test_mean': test_mean,
            'train_std': train_std,
            'test_std': test_std,
            'drift_score_std_devs': drift_score
        })
        
    if not drift_report:
        print("⚠️ No features valid for drift check (all excluded or low variance).")
        return

    drift_df = pd.DataFrame(drift_report)
    
    # Calculate percentage relative to STD (Variant of Z-Score) for realistic magnitude
    drift_df["drift_intensity_pct"] = (abs(drift_df["train_mean"] - drift_df["test_mean"]) / (drift_df["train_std"] + 1e-9)) * 100
    
    drift_df = drift_df.sort_values('drift_score_std_devs', ascending=False)
    
    # Save Report
    report_path = LOGS_DIR / "data_drift_report.csv"
    drift_df.to_csv(report_path, index=False)
    
    print("📊 Top 5 Features with highest drift (mean shift in std devs):")
    print(drift_df[['feature', 'drift_score_std_devs', 'drift_intensity_pct']].head(5))
    print(f"✅ Drift report saved to: {report_path}")

def main():
    """
    Phase 4: Data Preprocessing & Splitting
    -------------------------------------------------
    Steps:
      4.1  Clean numerical residues (Inf -> NaN -> Median)
      4.4  Time-based Split (Train 70 / Valid 15 / Test 15)
      4.2  Outlier clipping (Fit on Train, Apply to All)
      4.3  Feature scaling (Fit on Train, Apply to All)
      4.5  Data Drift Check -> logs/data_drift_report.csv
    """
    try:
        # 1. Load
        df = load_data()
        
        # 2. Clean
        df = clean_numerical_residues(df)
        
        # Save pre-split full data for reference
        full_path = PROCESSED_DATA_DIR / "step4_preprocessed_full.csv"
        df.to_csv(full_path, index=False)
        print(f"💾 Saved full preprocessed (cleaned) data to: {full_path}")
        
        # 3. Split (BEFORE Clipping/Scaling)
        train, valid, test = perform_time_based_split(df)
        
        # 4. Clip (New Position: After Split, Fit on Train)
        train, valid, test = clip_outliers(train, valid, test)
        
        # 5. Scale
        train, valid, test = perform_feature_scaling(train, valid, test)
        
        # 6. Drift Check
        check_data_drift(train, test)
        
        # 7. Save Splits (in 'splits' folder)
        splits_dir = PROCESSED_DATA_DIR / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)
        
        train.to_csv(splits_dir / "train.csv", index=False)
        valid.to_csv(splits_dir / "valid.csv", index=False)
        test.to_csv(splits_dir / "test.csv", index=False)
        
        print(f"\n💾 Saved scaled splits to: {splits_dir}")
        print("✅ Phase 4 Pipeline Complete!")
        
    except Exception as e:
        print(f"❌ Error in Preprocessing pipeline: {e}")

if __name__ == "__main__":
    main()

