import pandas as pd
import numpy as np
import sys
import argparse
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

from config import PROCESSED_DATA_DIR, LOGS_DIR, MODELS_DIR  # noqa: E402
warnings.filterwarnings("ignore")
TARGETS = ("target_hit", "stop_hit")

PRODUCTION_FEATURES_BY_TARGET = {
  "target_hit": [
    "close","volume",
    "candle_body","upper_wick","candle_range","wick_ratio",
    "ratio_high_low","ratio_close_high"
  ],
  "stop_hit": [
    "close","volume",
    "candle_body","upper_wick","candle_range","wick_ratio",
    "ratio_high_low","ratio_close_high"
  ],
}

# ────────────────────────────────────────────────
# 1) Load Data (target-aware)
# ────────────────────────────────────────────────
def load_data(target: str) -> pd.DataFrame:
    """Load engineered feature dataset from Phase 3 output (target-aware)."""
    path = PROCESSED_DATA_DIR / f"step3_features_engineered_{target}.csv"
    if not path.exists():
        raise FileNotFoundError(f"❌ Missing file: {path} — Run Phase 3 feature engineering for '{target}' first!")
    df = pd.read_csv(path, low_memory=False)
    print(f"✅ Loaded {target} data: {path.name} with shape {df.shape}")
    return df

def apply_production_filter(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = df.copy()

    keep = PRODUCTION_FEATURES_BY_TARGET[target_col].copy()
    extra = [target_col]
    if "created_at" in df.columns:
        extra.append("created_at")

    missing = [c for c in keep + extra if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required production columns for {target_col}: {missing}")

    return df[keep + extra]

# ────────────────────────────────────────────────
# 2) Clean numerical residues
# ────────────────────────────────────────────────
def clean_numerical_residues(df: pd.DataFrame) -> pd.DataFrame:
    """Replace Inf/-Inf with NaN, fill NaN in numeric cols with median (robust)."""
    print("⏳ Cleaning numerical residues (Inf/NaN)...")
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    print("✅ Cleaned all Inf/NaN values.")
    return df

# ────────────────────────────────────────────────
# 4) Time-Based Stratified Split
# ────────────────────────────────────────────────
def perform_time_based_split(
    df: pd.DataFrame,
    target_col: str,
    train_size: float = 0.70,
    valid_size: float = 0.15,
    slices: int = 20,
    seed: int = 42,
):
    """
    Splits the dataset based on chronological order into Train/Valid/Test
    while maintaining class ratio over time slices (approx).
    """
    print("⏳ Performing time‑aware stratified split...")
    if "created_at" not in df.columns:
        raise ValueError("❌ 'created_at' is required for time-aware splitting but was not found.")

    df = df.sort_values("created_at").reset_index(drop=True).copy()

    if target_col not in df.columns:
        raise ValueError(f"❌ Target column '{target_col}' not found in dataframe columns.")

    test_size = 1 - (train_size + valid_size)
    if test_size <= 0:
        raise ValueError("❌ train_size + valid_size must be < 1.0")

    n = len(df)
    df["time_bin"] = pd.qcut(np.arange(n), q=slices, labels=False, duplicates="drop")

    train_parts, valid_parts, test_parts = [], [], []

    for _, chunk in df.groupby("time_bin"):
        # must have both classes inside bin to stratify safely
        if chunk[target_col].nunique() < 2:
            continue

        ones = chunk[chunk[target_col] == 1]
        zeros = chunk[chunk[target_col] == 0]

        # Split each class inside the bin (preserve ratio)
        train_1 = ones.sample(frac=train_size, random_state=seed)
        valid_1 = ones.drop(train_1.index).sample(
            frac=valid_size / (valid_size + test_size), random_state=seed
        )
        test_1 = ones.drop(train_1.index).drop(valid_1.index)

        train_0 = zeros.sample(frac=train_size, random_state=seed)
        valid_0 = zeros.drop(train_0.index).sample(
            frac=valid_size / (valid_size + test_size), random_state=seed
        )
        test_0 = zeros.drop(train_0.index).drop(valid_0.index)

        train_parts.append(pd.concat([train_1, train_0]))
        valid_parts.append(pd.concat([valid_1, valid_0]))
        test_parts.append(pd.concat([test_1, test_0]))

    if not train_parts or not valid_parts or not test_parts:
        raise ValueError("❌ Split failed: not enough bins with both classes. Consider reducing 'slices'.")

    train = pd.concat(train_parts).sort_index()
    valid = pd.concat(valid_parts).sort_index()
    test = pd.concat(test_parts).sort_index()

    for name, d in [("Train", train), ("Valid", valid), ("Test", test)]:
        dist = d[target_col].value_counts(normalize=True).mul(100).round(2).to_dict()
        print(f"   {name:<6}: {dist} | n={len(d)}")

    for split_df in (train, valid, test):
        split_df.drop(columns=["time_bin"], inplace=True, errors="ignore")

    return train, valid, test

# ────────────────────────────────────────────────
# 5) Outlier Clipping (fit on Train only)
# ────────────────────────────────────────────────
def clip_outliers(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, target_col: str):
    """Clip numeric features to [1%, 99%] quantiles computed on TRAIN only."""
    print("⏳ Clipping outliers (1st-99th percentile) [Fit on Train]...")

    exclude_cols = {"id", "status", "target_hit", "stop_hit", "target_type", "hit_first", "created_at"}
    numeric_cols = train.select_dtypes(include=[np.number]).columns
    cols_to_clip = [c for c in numeric_cols if c not in exclude_cols and c != target_col]

    for col in cols_to_clip:
        lower = train[col].quantile(0.01)
        upper = train[col].quantile(0.99)
        train[col] = train[col].clip(lower, upper)
        valid[col] = valid[col].clip(lower, upper)
        test[col] = test[col].clip(lower, upper)

    print(f"✅ Clipped outliers for {len(cols_to_clip)} numeric features.")
    return train, valid, test

# ────────────────────────────────────────────────
# 6) Feature Scaling
# ────────────────────────────────────────────────
def _select_scale_columns(train: pd.DataFrame, mode: str, target_col: str):
    non_features = {"id", "status", "target_hit", "stop_hit", "target_type", "hit_first", "created_at"}

    if mode == "production":
        cols = [
            c for c in PRODUCTION_FEATURES_BY_TARGET[target_col]
            if c in train.columns and c not in non_features and c != target_col
        ]
        return cols

    if mode == "all":
        numeric_cols = train.select_dtypes(include=[np.number]).columns
        cols = [c for c in numeric_cols if c not in non_features and c != target_col]
        return cols

    raise ValueError("scale_features must be one of: production, all")

def perform_feature_scaling(train, valid, test, target_col: str, scale_features: str, scaler_name: str):
    """Fit StandardScaler on TRAIN only, apply to all splits, and save the scaler."""
    cols = _select_scale_columns(train, mode=scale_features, target_col=target_col)
    if not cols:
        print("⚠️ No columns selected for scaling. Skipping scaling.")
        return train, valid, test

    print(f"⏳ Scaling features using StandardScaler (cols={len(cols)})...")

    scaler = StandardScaler()
    scaler.fit(train[cols])

    train[cols] = scaler.transform(train[cols])
    valid[cols] = scaler.transform(valid[cols])
    test[cols] = scaler.transform(test[cols])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "feature_order": cols}, MODELS_DIR / scaler_name)
    print(f"✅ Scaler saved to models/{scaler_name}")

    # Safety: replace inf if any
    for split_name, split_df in [("train", train), ("valid", valid), ("test", test)]:
        if np.isinf(split_df[cols].to_numpy()).any():
            print(f"⚠️ Found Inf values in {split_name} after scaling; replacing with 0.")
            split_df[cols] = split_df[cols].replace([np.inf, -np.inf], 0)

    return train, valid, test

# ────────────────────────────────────────────────
# 7) Data Drift Check (target-aware report file)
# ────────────────────────────────────────────────
def check_data_drift(train: pd.DataFrame, test: pd.DataFrame, target: str):
    """Compare train vs test feature stats and export a drift report."""
    print("⏳ Checking for data drift between Train & Test...")

    exclude = {"id", "status", "target_hit", "stop_hit", "target_type", "hit_first", "created_at", "coin", "TP1"}
    numeric_cols = train.select_dtypes(include=[np.number]).columns
    features = [c for c in numeric_cols if c not in exclude]

    drift_records = []
    for col in features:
        s_train = train[col]
        s_test = test[col]
        if s_train.std() < 1e-6:
            continue
        drift_score = abs(s_train.mean() - s_test.mean()) / (s_train.std() + 1e-9)
        drift_records.append(
            {
                "feature": col,
                "train_mean": float(s_train.mean()),
                "test_mean": float(s_test.mean()),
                "train_std": float(s_train.std()),
                "test_std": float(s_test.std()),
                "drift_stddevs": float(drift_score),
            }
        )

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = LOGS_DIR / f"data_drift_report_{target}.csv"

    if drift_records:
        drift_df = pd.DataFrame(drift_records).sort_values("drift_stddevs", ascending=False)
        drift_df.to_csv(report_path, index=False)
        print("📊 Top Drifted Features:")
        print(drift_df.head(5).to_string(index=False))
        print(f"✅ Drift report saved to: {report_path}")
    else:
        # still create an empty report for traceability
        pd.DataFrame(columns=["feature", "train_mean", "test_mean", "train_std", "test_std", "drift_stddevs"]).to_csv(
            report_path, index=False
        )
        print(f"⚠️ No drift records; empty drift report saved to: {report_path}")

# ────────────────────────────────────────────────
# 8) Helper: enforce numeric (light-touch)
# ────────────────────────────────────────────────
def enforce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Try converting object columns to numeric where possible (keeps non-numeric untouched)."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase 4: Target-aware preprocessing & splitting")
    parser.add_argument("--target", type=str, default="target_hit", choices=list(TARGETS))
    parser.add_argument(
        "--scaling",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="auto: on for linear models, off for tree models; on/off forces behavior",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="tree",
        choices=["tree", "linear"],
        help="Used only when --scaling auto",
    )
    parser.add_argument(
        "--scale-features",
        type=str,
        default="production",
        choices=["production", "all"],
        help="Which columns to scale if scaling enabled",
    )
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--valid-size", type=float, default=0.15)
    parser.add_argument("--slices", type=int, default=20)
    args = parser.parse_args()

    target = args.target

    # Determine scaling behavior
    if args.scaling == "on":
        do_scaling = True
    elif args.scaling == "off":
        do_scaling = False
    else:
        # auto
        do_scaling = (args.model_type == "linear")

    try:
        # Step 1–4
        df = load_data(target=target)
        df = apply_production_filter(df, target_col=target)
        df = enforce_numeric_columns(df)
        df = clean_numerical_residues(df)

        # Save cleaned full copy (target-aware)
        pre_path = PROCESSED_DATA_DIR / f"step4_preprocessed_full_{target}.csv"
        df.to_csv(pre_path, index=False)
        print(f"💾 Preprocessed (uncut) data saved to: {pre_path}")

        # Step 5 – time-based split (target-aware)
        train, valid, test = perform_time_based_split(
            df,
            target_col=target,
            train_size=args.train_size,
            valid_size=args.valid_size,
            slices=args.slices,
        )

        # Drop created_at AFTER split (keeps chronology for splitting only)
        for name, split in [("train", train), ("valid", valid), ("test", test)]:
            if "created_at" in split.columns:
                split.drop(columns=["created_at"], inplace=True, errors="ignore")
                print(f"🗑️ Dropped 'created_at' from {name} split.")

        # Step 6 – clip outliers
        train, valid, test = clip_outliers(train, valid, test, target_col=target)

        # Step 7 – scaling (optional, target-aware save)
        if do_scaling:
            scaler_name = f"scaler_{target}.pkl"
            train, valid, test = perform_feature_scaling(
                train,
                valid,
                test,
                target_col=target,
                scale_features=args.scale_features,
                scaler_name=scaler_name,
            )
        else:
            print("⏭️ Scaling is OFF (recommended for tree models). Scaler will not be saved.")

        # Step 8 – drift check (target-aware report)
        check_data_drift(train, test, target=target)

        # Step 9 – save splits (target-aware folder)
        split_dir = PROCESSED_DATA_DIR / f"splits_{target}"
        split_dir.mkdir(parents=True, exist_ok=True)

        train.to_csv(split_dir / "train.csv", index=False)
        valid.to_csv(split_dir / "valid.csv", index=False)
        test.to_csv(split_dir / "test.csv", index=False)

        print(f"\n✅ Phase 4 complete for '{target}'. Splits stored in: {split_dir}")

        if do_scaling:
            print(f"✅ Scaler stored in: {MODELS_DIR / f'scaler_{target}.pkl'}")

        print("👉 Next: Train models using the matching splits folder (and match the same target).")

    except Exception as e:
        print(f"❌ Error in preprocessing pipeline: {e}")
        raise

if __name__ == "__main__":
    main()