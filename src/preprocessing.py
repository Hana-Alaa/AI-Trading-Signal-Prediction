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

def balance_dataset(df, target_col="target_hit", ratio=2.5):
    """
    Step 4.0: Balance dataset before splitting.
    - Keeps all minority samples (class 1)
    - Downsamples majority class (class 0) to maintain ~1:ratio proportion.
    """
    print("\n⚖️ Balancing dataset before time-based split...")

    if target_col not in df.columns:
        raise ValueError(f"❌ Target column '{target_col}' not found in dataframe!")

    # Separate majority and minority classes
    minority = df[df[target_col] == 1]
    majority = df[df[target_col] == 0]

    print(f"📊 Original distribution: 0={len(majority)}, 1={len(minority)}, ratio={(len(majority) / len(minority)):.2f}:1")

    # Downsample majority class to maintain target ratio
    desired_majority = int(len(minority) * ratio)
    majority_downsampled = majority.sample(
        n=min(desired_majority, len(majority)),  # avoids sampling beyond available rows
        random_state=42
    )

    balanced_df = pd.concat([minority, majority_downsampled]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"✅ Balanced dataset: 0={len(majority_downsampled)}, 1={len(minority)}, "
          f"ratio≈{(len(majority_downsampled)/len(minority)):.2f}:1")
    print(f"🔹 Shape after balancing: {balanced_df.shape}")
    print(df["target_hit"].value_counts(normalize=True).rename("percent").to_frame()*100)

    return balanced_df

def perform_time_based_split(df):
    """
    Step 4.4: Time-Based Split (70/15/15), preserves time order but softly rebalances validation/test.
    """
    print("⏳ Performing Time-Based Split...")

    # Sort by created_at (chronological order)
    if 'created_at' in df.columns:
        df = df.sort_values('created_at').reset_index(drop=True)

    n = len(df)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)

    train = df.iloc[:train_end].copy()
    valid = df.iloc[train_end:valid_end].copy()
    test  = df.iloc[valid_end:].copy()

    # Sanity check
    def has_both_classes(sub):
        return sub['target_hit'].nunique() == 2

    # Ensure at least both classes exist
    if not has_both_classes(valid):
        print("⚠️ VALID had only one class — injecting 50 positive samples from Train.")
        extra = train[train['target_hit'] == 1].tail(50)
        valid = pd.concat([valid, extra]).sort_values('created_at').reset_index(drop=True)

    if not has_both_classes(test):
        print("⚠️ TEST had only one class — injecting 50 positive samples from Train.")
        extra = train[train['target_hit'] == 1].tail(50)
        test = pd.concat([test, extra]).sort_values('created_at').reset_index(drop=True)

    # 🟩 Optionally rebalance valid/test to have ~10% positives (without shuffling)
    target_col = "target_hit"
    desired_ratio = 0.15  # ~15% positives

    for name, split_df in [("Valid", valid), ("Test", test)]:
        pos_count = split_df[split_df[target_col] == 1].shape[0]
        total_count = len(split_df)
        current_ratio = pos_count / total_count
        if current_ratio < desired_ratio:
            need = int(desired_ratio * total_count - pos_count)
            extra = train[train[target_col] == 1].tail(need)
            split_df = pd.concat([split_df, extra]).sort_values('created_at').reset_index(drop=True)
            print(f"🩵 Added {len(extra)} positives to {name} to reach ≈{desired_ratio*100:.0f}% class 1.")
        if name == "Valid":
            valid = split_df
        else:
            test = split_df

    # Summary
    print(f"✅ Split Ratios (approx): Train={len(train)/n:.1%}, Valid={len(valid)/n:.1%}, Test={len(test)/n:.1%}")
    print("📊 Target distribution (after balancing):")
    for name, d in zip(["Train", "Valid", "Test"], [train, valid, test]):
        counts = d['target_hit'].value_counts(normalize=True).mul(100).round(2).to_dict()
        print(f"   {name:<5}: {counts}")

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

def drop_irrelevant_and_leakage_cols(df, target_col='target_hit'):
    """
    Drop columns that shouldn't be used as input features:
    - IDs, metadata
    - opposite targets (leakage)
    - time-based leakage columns
    - purely descriptive identifiers
    """
    print(f"🧹 Dropping irrelevant & leakage columns for target: {target_col}")

    # columns we usually never need
    to_drop = [
        'id', 'coin', 'status',
        'target_type', 'hit_first',
        'time_to_event',
        'TP1','TP5','TP7','TP9','TP10','TP12','TP14','TP16','TP18','TP20','TP25','TP45',
        '1h','1day','3day'
    ]

    # Add the "opposite target" to drop
    if target_col == 'target_hit' and 'stop_hit' in df.columns:
        to_drop.append('stop_hit')
    elif target_col == 'stop_hit' and 'target_hit' in df.columns:
        to_drop.append('target_hit')

    # drop columns if they exist
    drop_existing = [c for c in to_drop if c in df.columns]
    df = df.drop(columns=drop_existing, errors='ignore')

    print(f"✅ Dropped {len(drop_existing)} columns: {drop_existing[:8]}{'...' if len(drop_existing)>8 else ''}")
    print(f"Remaining columns: {len(df.columns)}")
    return df

def enforce_numeric_columns(df):
    """
    Force all numeric-like columns (even if read as object) to stay numeric.
    Prevents features from disappearing after saving/loading.
    """
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="ignore")
    return df

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

        # 2. Drop leakage & unrelated columns (auto detects based on target)
        df = drop_irrelevant_and_leakage_cols(df, target_col='target_hit')

        # 3. Clean numerical residues
        df = clean_numerical_residues(df)


        # 4. enforce_numeric_columns
        df = enforce_numeric_columns(df)
        
        # 5. Balance data BEFORE splitting
        df = balance_dataset(df, target_col="target_hit", ratio=2.5)
        
        # Save pre-split full data for reference
        full_path = PROCESSED_DATA_DIR / "step4_preprocessed_full.csv"
        df.to_csv(full_path, index=False)
        print(f"💾 Saved full preprocessed (cleaned & balanced) data to: {full_path}")
        
        # 6. Time-based split
        train, valid, test = perform_time_based_split(df)
        
        for split_name, split_df in zip(["train", "valid", "test"], [train, valid, test]):
            if "created_at" in split_df.columns:
                split_df.drop(columns=["created_at"], inplace=True)
                print(f"🗑️ Dropped 'created_at' from {split_name} split.")

        # 7. Clip (New Position: After Split, Fit on Train)
        train, valid, test = clip_outliers(train, valid, test)
        
        # 8. Scale
        train, valid, test = perform_feature_scaling(train, valid, test)
        
        # 9. Drift Check
        check_data_drift(train, test)
        
        # 10. Save Splits (in 'splits' folder)
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

