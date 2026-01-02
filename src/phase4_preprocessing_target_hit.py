import pandas as pd
import numpy as np
import sys
import joblib
from pathlib import Path
import warnings
from sklearn.preprocessing import StandardScaler

# ────────────────────────────────────────────────
# Setup project paths
# ────────────────────────────────────────────────
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    sys.path.append(str(project_root))

from config import PROCESSED_DATA_DIR, LOGS_DIR, MODELS_DIR

warnings.filterwarnings("ignore")


# ────────────────────────────────────────────────
# 1️⃣ Load Data
# ────────────────────────────────────────────────
def load_data():
    """Load engineered feature dataset from Phase 3 output."""
    path = PROCESSED_DATA_DIR / "step3_features_engineered.csv"
    if not path.exists():
        raise FileNotFoundError(f"❌ Missing file: {path} — Run Phase 3 first!")
    df = pd.read_csv(path)
    print(f"✅ Loaded data with shape {df.shape}")
    return df


# ────────────────────────────────────────────────
# 2️⃣ Clean numerical residues
# ────────────────────────────────────────────────
def clean_numerical_residues(df):
    """
    Replaces Inf/-Inf with NaN, fills NaN by median.
    """
    print("⏳ Cleaning numerical residues (Inf/NaN)...")
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    print("✅ Cleaned all Inf/NaN values.")
    return df


# ────────────────────────────────────────────────
# 3️⃣ Drop leakage & irrelevant cols
# ────────────────────────────────────────────────
def drop_irrelevant_and_leakage_cols(df, target_col="target_hit"):
    """
    Drop columns that leak future information or are irrelevant for model training.
    """
    print(f"🧹 Dropping irrelevant & leakage columns for target: {target_col}")

    to_drop = [
        "id", "coin", "status", "target_type", "hit_first",
        "time_to_event",
        "TP1","TP5","TP7","TP9","TP10","TP12","TP14","TP16","TP18","TP20","TP25","TP45",
        "1h","1day","3day"
    ]

    # prevent leaking opposite target
    if target_col == "target_hit" and "stop_hit" in df.columns:
        to_drop.append("stop_hit")
    elif target_col == "stop_hit" and "target_hit" in df.columns:
        to_drop.append("target_hit")

    existing = [c for c in to_drop if c in df.columns]
    df = df.drop(columns=existing, errors="ignore")

    print(f"✅ Dropped {len(existing)} leakage columns.")
    print(f"Remaining columns: {len(df.columns)}")
    return df


# ────────────────────────────────────────────────
# 4️⃣ Time-Based Stratified Split
# ────────────────────────────────────────────────
def perform_time_based_split(df, target_col="target_hit", train_size=0.7, valid_size=0.15):
    """
    Splits the dataset based on chronological order into Train/Valid/Test
    while maintaining class ratio over time slices.
    """
    print("⏳ Performing time‑aware stratified split...")
    df = df.sort_values("created_at").reset_index(drop=True)
    n = len(df)
    slices = 20
    df["time_bin"] = pd.qcut(np.arange(n), q=slices, labels=False)

    train_parts, valid_parts, test_parts = [], [], []
    test_size = 1 - (train_size + valid_size)

    for _, chunk in df.groupby("time_bin"):
        if chunk[target_col].nunique() < 2:
            continue

        chunk_ones = chunk[chunk[target_col] == 1]
        chunk_zeros = chunk[chunk[target_col] == 0]

        # sample by ratio to preserve class proportion per time bin
        train_1 = chunk_ones.sample(frac=train_size, random_state=42)
        valid_1 = chunk_ones.drop(train_1.index).sample(
            frac=valid_size / (valid_size + test_size), random_state=42
        )
        test_1 = chunk_ones.drop(train_1.index).drop(valid_1.index)

        train_0 = chunk_zeros.sample(frac=train_size, random_state=42)
        valid_0 = chunk_zeros.drop(train_0.index).sample(
            frac=valid_size / (valid_size + test_size), random_state=42
        )
        test_0 = chunk_zeros.drop(train_0.index).drop(valid_0.index)

        train_parts.append(pd.concat([train_1, train_0]))
        valid_parts.append(pd.concat([valid_1, valid_0]))
        test_parts.append(pd.concat([test_1, test_0]))

    train, valid, test = (
        pd.concat(train_parts).sort_index(),
        pd.concat(valid_parts).sort_index(),
        pd.concat(test_parts).sort_index(),
    )

    for name, d in zip(["Train", "Valid", "Test"], [train, valid, test]):
        dist = d[target_col].value_counts(normalize=True).mul(100).round(2).to_dict()
        print(f"   {name:<6}: {dist}")

    for split_df in [train, valid, test]:
        split_df.drop(columns=["time_bin"], inplace=True, errors="ignore")
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

# ────────────────────────────────────────────────
# 6️⃣ Feature Scaling
# ────────────────────────────────────────────────
def perform_feature_scaling(train, valid, test):
    """
    Fit StandardScaler on TRAIN features only.
    Scaling matches production feature schema.
    """
    print("⏳ Scaling features (StandardScaler, production‑aligned)...")
    PRODUCTION_READY_FEATURES = [
        "close", "volume", "RSI", "rsi_1d", "rsi_3d", "atr_1h",
        "candle_body", "upper_wick", "candle_range", "wick_ratio",
        "ratio_high_low", "ratio_close_high"
    ]
    feature_cols = [c for c in PRODUCTION_READY_FEATURES if c in train.columns]
    if not feature_cols:
        print("⚠️ No production features found to scale.")
        return train, valid, test

    scaler = StandardScaler()
    scaler.fit(train[feature_cols])

    train[feature_cols] = scaler.transform(train[feature_cols])
    valid[feature_cols] = scaler.transform(valid[feature_cols])
    test[feature_cols]  = scaler.transform(test[feature_cols])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "feature_order": feature_cols}, MODELS_DIR / "scaler.pkl")
    print(f"✅ Scaled {len(feature_cols)} features — Scaler saved to models/scaler.pkl")

    for split_name, split_df in zip(["train", "valid", "test"], [train, valid, test]):
        if np.isinf(split_df[feature_cols].values).any():
            print(f"⚠️ Found Inf values in {split_name}, replacing with 0.")
            split_df[feature_cols].replace([np.inf, -np.inf], 0, inplace=True)
    return train, valid, test


# ────────────────────────────────────────────────
# 7️⃣ Data Drift Check
# ────────────────────────────────────────────────
def check_data_drift(train, test):
    """
    Compare train vs test feature statistics (mean/std)
    and export a drift report for diagnostics.
    """
    print("⏳ Checking for data drift between Train & Test...")
    exclude = [
        "id","status","target_hit","stop_hit","target_type",
        "hit_first","created_at","coin","TP1"
    ]
    numeric_cols = train.select_dtypes(include=[np.number]).columns
    features = [c for c in numeric_cols if c not in exclude]

    drift_records = []
    for col in features:
        s_train = train[col]
        s_test  = test[col]
        if s_train.std() < 1e-6:
            continue
        drift_score = abs(s_train.mean() - s_test.mean()) / (s_train.std() + 1e-9)
        drift_records.append({
            "feature": col,
            "train_mean": s_train.mean(),
            "test_mean": s_test.mean(),
            "train_std": s_train.std(),
            "test_std": s_test.std(),
            "drift_stddevs": drift_score
        })

    if drift_records:
        drift_df = pd.DataFrame(drift_records).sort_values("drift_stddevs", ascending=False)
        report_path = LOGS_DIR / "data_drift_report.csv"
        drift_df.to_csv(report_path, index=False)
        print("📊 Top Drifted Features:")
        print(drift_df.head(5))
        print(f"✅ Drift report saved to: {report_path}")
    else:
        print("⚠️ No drift report generated (no valid features).")


# ────────────────────────────────────────────────
# 8️⃣ Helper: enforce numeric
# ────────────────────────────────────────────────
def enforce_numeric_columns(df):
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


# ────────────────────────────────────────────────
# 🚀 Main Pipeline
# ────────────────────────────────────────────────
def main():
    """
    Phase 4: Data Preprocessing & Splitting
    ------------------------------------------------------------
    Steps:
      1. Load data
      2. Drop leakage / irrelevant cols
      3. Clean missing and Inf values
      4. Enforce numeric types
      5. Chronological Split (Train / Valid / Test)
      6. Clip Outliers (based on Train)
      7. Scale Features
      8. Data Drift Check
      9. Save final splits
    """
    try:
        # Step 1–4
        df = load_data()
        df = drop_irrelevant_and_leakage_cols(df, target_col="target_hit")
        df = clean_numerical_residues(df)
        df = enforce_numeric_columns(df)

        # Save cleaned full copy (for traceability)
        pre_path = PROCESSED_DATA_DIR / "step4_preprocessed_full.csv"
        df.to_csv(pre_path, index=False)
        print(f"💾 Preprocessed (uncut) data saved to: {pre_path}")

        # Step 5 – time-based split
        train, valid, test = perform_time_based_split(df)

        for name, split in zip(["train","valid","test"], [train,valid,test]):
            if "created_at" in split.columns:
                split.drop(columns=["created_at"], inplace=True, errors="ignore")
                print(f"🗑️ Dropped 'created_at' from {name} split.")

        # Step 6 – clip outliers
        train, valid, test = clip_outliers(train, valid, test)

        # Step 7 – scaling
        train, valid, test = perform_feature_scaling(train, valid, test)

        # Step 8 – drift check
        check_data_drift(train, test)

        # Step 9 – save splits
        split_dir = PROCESSED_DATA_DIR / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)
        train.to_csv(split_dir / "train.csv", index=False)
        valid.to_csv(split_dir / "valid.csv", index=False)
        test.to_csv(split_dir / "test.csv", index=False)

        print(f"\n✅ Phase 4 complete. Splits & scaler stored in {split_dir}")
        print("👉 Next: Train models using class_weight='balanced' to handle imbalance.")
    except Exception as e:
        print(f"❌ Error in preprocessing pipeline: {e}")

if __name__ == "__main__":
    main()