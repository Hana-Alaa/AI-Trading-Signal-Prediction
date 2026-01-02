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

# ===============================================================
#   STEP 4.1 — Load & Clean Data
# ===============================================================
def load_data():
    """Load the engineered features dataset."""
    path = PROCESSED_DATA_DIR / "step3_features_engineered.csv"
    if not path.exists():
        raise FileNotFoundError(f"❌ File not found: {path} - Please run Phase 3 first.")
    df = pd.read_csv(path)
    print(f"✅ Loaded data: {df.shape}")
    return df

def clean_numerical_residues(df):
    """Replace Inf/NaN with median values."""
    print("⏳ Cleaning numerical residues...")
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    print("✅ Cleaned Inf/NaN values.")
    return df

# ===============================================================
#   STEP 4.2 — Drop Leakage / Irrelevant Columns
# ===============================================================
def drop_irrelevant_and_leakage_cols(df, target_col='target_hit'):
    """Remove columns that cause leakage or are redundant."""
    print(f"🧹 Dropping irrelevant/leakage columns for target: {target_col}")

    to_drop = [
        'id', 'coin', 'status', 'target_type', 'hit_first',
        'time_to_event',
        'TP1','TP5','TP7','TP9','TP10','TP12','TP14','TP16','TP18',
        'TP20','TP25','TP45','1h','1day','3day'
    ]

    # Drop opposite target to prevent data leakage
    if target_col == 'target_hit' and 'stop_hit' in df.columns:
        to_drop.append('stop_hit')
    elif target_col == 'stop_hit' and 'target_hit' in df.columns:
        to_drop.append('target_hit')

    existing = [c for c in to_drop if c in df.columns]
    df.drop(columns=existing, inplace=True, errors="ignore")
    print(f"✅ Dropped {len(existing)} columns. Remaining columns: {len(df.columns)}")
    return df

# ===============================================================
#   STEP 4.3 — Time-based Split
# ===============================================================
def perform_time_based_split(df, target_col="target_hit", train_size=0.7, valid_size=0.15):
    """Perform a chronological stratified split (Train/Valid/Test)."""
    print("⏳ Performing Time‑Aware Stratified Split...")

    df = df.sort_values("created_at").reset_index(drop=True)
    test_size = 1 - (train_size + valid_size)
    slices = 20
    df["time_bin"] = pd.qcut(np.arange(len(df)), q=slices, labels=False)

    train_parts, valid_parts, test_parts = [], [], []

    for _, chunk in df.groupby("time_bin"):
        if chunk[target_col].nunique() < 2:
            continue

        chunk_ones = chunk[chunk[target_col] == 1]
        chunk_zeros = chunk[chunk[target_col] == 0]

        train_1 = chunk_ones.sample(frac=train_size, random_state=42)
        valid_1 = chunk_ones.drop(train_1.index).sample(frac=valid_size/(valid_size+test_size), random_state=42)
        test_1  = chunk_ones.drop(train_1.index).drop(valid_1.index)

        train_0 = chunk_zeros.sample(frac=train_size, random_state=42)
        valid_0 = chunk_zeros.drop(train_0.index).sample(frac=valid_size/(valid_size+test_size), random_state=42)
        test_0  = chunk_zeros.drop(train_0.index).drop(valid_0.index)

        train_parts.append(pd.concat([train_1, train_0]))
        valid_parts.append(pd.concat([valid_1, valid_0]))
        test_parts.append(pd.concat([test_1, test_0]))

    train = pd.concat(train_parts).sort_index()
    valid = pd.concat(valid_parts).sort_index()
    test  = pd.concat(test_parts).sort_index()

    for name, d in zip(["Train", "Valid", "Test"], [train, valid, test]):
        distribution = d[target_col].value_counts(normalize=True).mul(100).round(2).to_dict()
        print(f"   {name:<5}: {distribution}")

    df.drop(columns=["time_bin"], inplace=True, errors="ignore")
    return train, valid, test

# ===============================================================
#   STEP 4.4 — Outlier Clipping
# ===============================================================
def clip_outliers(train, valid, test):
    """Clip extreme values (1st–99th percentiles) based on TRAIN only."""
    print("⏳ Clipping outliers (1st–99th percentiles)...")
    exclude_cols = ['id', 'status', 'target_hit', 'stop_hit', 'target_type', 'hit_first']
    numeric_cols = [c for c in train.select_dtypes(include=[np.number]).columns if c not in exclude_cols]

    for col in numeric_cols:
        lower = train[col].quantile(0.01)
        upper = train[col].quantile(0.99)
        train[col] = train[col].clip(lower, upper)
        valid[col] = valid[col].clip(lower, upper)
        test[col]  = test[col].clip(lower, upper)

    print(f"✅ Outliers clipped for {len(numeric_cols)} features.")
    return train, valid, test

# ===============================================================
#   STEP 4.5 — Feature Scaling
# ===============================================================
def perform_feature_scaling(train, valid, test):
    """Standardize only the numeric features that production API will use."""
    print("⏳ Scaling Features (StandardScaler, production-aligned)...")

    PRODUCTION_READY_FEATURES = [
        'close', 'volume', 'RSI', 'rsi_1d', 'rsi_3d', 'atr_1h',
        'candle_body', 'upper_wick', 'candle_range', 'wick_ratio',
        'ratio_high_low', 'ratio_close_high'
    ]

    feature_cols = [f for f in PRODUCTION_READY_FEATURES if f in train.columns]
    if not feature_cols:
        print("⚠️ No matching production features found; skipping scaling.")
        return train, valid, test

    scaler = StandardScaler()
    scaler.fit(train[feature_cols])

    train[feature_cols] = scaler.transform(train[feature_cols])
    valid[feature_cols] = scaler.transform(valid[feature_cols])
    test[feature_cols]  = scaler.transform(test[feature_cols])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    scaler_path = MODELS_DIR / "scaler.pkl"
    joblib.dump({'scaler': scaler, 'feature_order': feature_cols}, scaler_path)
    print(f"✅ Scaler fitted and saved to: {scaler_path}")
    return train, valid, test

# ===============================================================
#   STEP 4.6 — Data Drift Check
# ===============================================================
def check_data_drift(train, test):
    """Report feature drift between train and test sets."""
    print("⏳ Checking Data Drift...")
    exclude_cols = ['id', 'status', 'target_hit', 'stop_hit', 'target_type', 'hit_first', 'created_at', 'coin']
    numeric_cols = train.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    drift_report = []

    for col in feature_cols:
        tr_std = train[col].std()
        if tr_std < 1e-6:
            continue
        diff = abs(train[col].mean() - test[col].mean()) / (tr_std + 1e-9)
        drift_report.append({'feature': col, 'drift_stddevs': diff})

    drift_df = pd.DataFrame(drift_report).sort_values('drift_stddevs', ascending=False)
    report_path = LOGS_DIR / "data_drift_report.csv"
    drift_df.to_csv(report_path, index=False)
    print("📊 Top Drifted Features:")
    print(drift_df.head(5))
    print(f"✅ Drift report saved to: {report_path}")

# ===============================================================
#   MAIN PIPELINE
# ===============================================================
def main():
    """
    Phase 4: Preprocessing & Splitting Pipeline (final no‑balancing version)
    -------------------------------------------------
    Steps:
      4.1  Load & clean numerical residues
      4.2  Drop leakage columns
      4.3  Time-based split (Train 70 / Valid 15 / Test 15)
      4.4  Clip outliers
      4.5  Standardize features
      4.6  Check data drift
    """
    try:
        df = load_data()
        df = drop_irrelevant_and_leakage_cols(df, target_col='target_hit')
        df = clean_numerical_residues(df)
        df = enforce_numeric_columns(df)

        # 🔹 Save pre-split dataset snapshot
        full_path = PROCESSED_DATA_DIR / "step4_preprocessed_full.csv"
        df.to_csv(full_path, index=False)
        print(f"💾 Saved cleaned dataset to: {full_path}")

        # 🔹 Time-based split only (no balancing)
        train, valid, test = perform_time_based_split(df)

        # 🔹 Drop timestamp columns
        for name, split in zip(["train", "valid", "test"], [train, valid, test]):
            if "created_at" in split.columns:
                split.drop(columns=["created_at"], inplace=True)
                print(f"🗑️ Dropped 'created_at' from {name} split.")

        # 🔹 Clip & Scale
        train, valid, test = clip_outliers(train, valid, test)
        train, valid, test = perform_feature_scaling(train, valid, test)

        # 🔹 Drift check
        check_data_drift(train, test)

        # 🔹 Save outputs
        splits_dir = PROCESSED_DATA_DIR / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)
        train.to_csv(splits_dir / "train.csv", index=False)
        valid.to_csv(splits_dir / "valid.csv", index=False)
        test.to_csv(splits_dir / "test.csv", index=False)

        print(f"\n💾 Saved Train/Valid/Test splits → {splits_dir}")
        print("✅ Phase 4 Preprocessing Pipeline Complete!")

    except Exception as e:
        print(f"❌ Error in preprocessing pipeline: {e}")

# ===============================================================
#   Helper
# ===============================================================
def enforce_numeric_columns(df):
    """Force numeric types for consistency across splits."""
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="ignore")
    return df

if __name__ == "__main__":
    main()