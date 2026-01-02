import pandas as pd
import numpy as np
import sys
from pathlib import Path
import warnings

# Add project root to path if running directly
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    sys.path.append(str(project_root))

from config import PROCESSED_DATA_DIR

warnings.filterwarnings("ignore")

def calculate_price_action_features(df):
    """
    Calculates Price Action features: Candle Body, Wicks, and Price Ratios.
    """
    print("⏳ Calculating Price Action Features...")
    df = df.copy()
    
    # 1. Candle Properties
    df['candle_body'] = df['close'] - df['open']
    df['candle_body_pct'] = (df['close'] - df['open']) / df['open']
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['candle_range'] = df['high'] - df['low']
    
    # Handle division by zero for wick ratio
    df['wick_ratio'] = df['upper_wick'] / (df['lower_wick'] + 1e-9)

    # 2. Price Ratios
    df['ratio_close_open'] = df['close'] / df['open']
    df['ratio_high_low'] = df['high'] / df['low']
    df['ratio_close_high'] = df['close'] / df['high']
    df['ratio_close_low'] = df['close'] / df['low']
    
    # 3. Entry Price Context (if available)
    if 'entry_price' in df.columns:
        df['price_move_ratio'] = (df['close'] - df['entry_price']) / df['entry_price']
        
    print("✅ Price Action Features calculated.")
    return df

def calculate_volatility_features(df, windows=[7, 14, 50]):
    """
    Calculates Volatility features: Rolling Stats, Bollinger Bands, ATR.
    """
    print("⏳ Calculating Volatility Features...")
    df = df.copy()
    
    # 1. Rolling Statistics
    for w in windows:
        df[f'rolling_mean_{w}'] = df['close'].rolling(window=w).mean()
        df[f'rolling_std_{w}'] = df['close'].rolling(window=w).std()
        df[f'z_score_{w}'] = (df['close'] - df[f'rolling_mean_{w}']) / (df[f'rolling_std_{w}'] + 1e-9)

    # 2. Bollinger Bands (20, 2)
    bb_window = 20
    bb_std = 2
    df['bb_middle'] = df['close'].rolling(window=bb_window).mean()
    df['bb_upper'] = df['bb_middle'] + (df['close'].rolling(window=bb_window).std() * bb_std)
    df['bb_lower'] = df['bb_middle'] - (df['close'].rolling(window=bb_window).std() * bb_std)
    
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_pct_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-9)

    # 3. ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(window=14).mean()
    df['atr_pct'] = df['atr_14'] / df['close']
    
    # Cleanup initial NaNs from rolling windows (using largest window)
    max_window = max(windows + [bb_window])
    initial_shape = df.shape
    df.dropna(subset=[f'rolling_std_{max_window}', 'bb_upper', 'atr_14'], inplace=True)
    
    print(f"✅ Volatility Features calculated. Dropped {initial_shape[0] - df.shape[0]} initial rows (NaNs).")
    return df

def calculate_momentum_features(df):
    """
    Calculates Momentum features: RSI, MACD, ROC, Interaction features.
    """
    print("⏳ Calculating Momentum Features...")
    df = df.copy()
    
    # Helper: RSI Calculation
    def calculate_rsi(series, window=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    # 1. RSI
    if 'RSI' not in df.columns:
        df['RSI'] = calculate_rsi(df['close'])
    
    df['rsi_slope'] = df['RSI'].diff()
    df['rsi_slope_3'] = df['RSI'].diff(3)

    # 2. MACD (12, 26, 9)
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # 3. ROC
    for n in [7, 14]:
        df[f'roc_{n}'] = df['close'].pct_change(periods=n) * 100

    # 4. Interaction Features
    df['rsi_x_vol'] = df['RSI'] * df['bb_width']
    df['macd_x_vol'] = df['macd_hist'] * df['atr_pct']
    
    # RSI momentum between current RSI and 3-day RSI
    if 'rsi_3d' in df.columns:
        df['rsi_momentum'] = df['RSI'] - df['rsi_3d']

    # Drop NaNs created by diff/pct_change
    df.dropna(subset=['rsi_slope', 'macd_hist', 'roc_14'], inplace=True)
    
    print("✅ Momentum Features calculated.")
    return df

def perform_feature_selection(df, target_col='target_hit', threshold=0.95):
    """
    Drops highly correlated features (> threshold), keeping the one more correlated with target.
    """
    print(f"⏳ Performing Feature Selection (Threshold: {threshold})...")
    df = df.copy()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Exclude non-features from correlation check
    exclude_cols = ['target_hit', 'stop_hit', 'target_type', 'hit_first']
    if 'Unnamed: 0' in df.columns:
        exclude_cols.append('Unnamed: 0')
        
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    # Correlation Matrix
    corr_matrix = df[feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find candidates to drop
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    final_drop_list = []
    
    if target_col in df.columns:
        target_corr = df[feature_cols].corrwith(df[target_col]).abs()
        
        for col in to_drop:
            # All features correlated with 'col' > threshold
            correlated_with = upper.index[upper[col] > threshold].tolist()
            
            for other_col in correlated_with:
                score_current = target_corr.get(col, 0)
                score_other = target_corr.get(other_col, 0)
                
                # Drop the one with lower correlation to target
                if score_current < score_other:
                    final_drop_list.append(col)
                else:
                    final_drop_list.append(other_col)
    else:
        final_drop_list = to_drop
    
    # Dedup
    final_drop_list = list(set(final_drop_list))
    
    df_reduced = df.drop(columns=final_drop_list)
    print(f"🔥 Dropped {len(final_drop_list)} redundant features.")
    return df_reduced

# 1. Add this list to the beginning of the file (after imports)
PRODUCTION_READY_FEATURES = [
    'close', 'open', 'high', 'low', 'volume', 
    'RSI', 'rsi_1d', 'rsi_3d', 'atr_1h', 'time_bin',
    'candle_body', 'upper_wick', 'candle_range', 'wick_ratio',
    'ratio_close_open', 'ratio_high_low', 'ratio_close_high', 'price_move_ratio'
]

# 2. Replace the main() function with this version:
def main():
    """
    Main execution pipeline with Production Filter.
    """
    try:
        # 1. Load Data
        input_path = PROCESSED_DATA_DIR / "step1_quality_checked.csv"
        if not input_path.exists():
             raise FileNotFoundError(f"Input file not found: {input_path}")
             
        print(f"Loading data from {input_path}...")
        df = pd.read_csv(input_path)
        
        # 2. Feature Engineering Pipeline (Calculate everything first)
        df = calculate_price_action_features(df)
        df = calculate_volatility_features(df)
        df = calculate_momentum_features(df)
        
        # 3. Feature Selection (Filtering based on correlation)
        df = perform_feature_selection(df)
        
        # ---------------------------------------------------------
        # New Step: Production Filter
        # ---------------------------------------------------------
        print("Applying Production Filter (keeping only API-compatible features)...")
        
        # Identify features that actually exist in the production list + targets
        keep_cols = [c for c in df.columns if c in PRODUCTION_READY_FEATURES or c == 'created_at']
        
        # Add target columns if they exist to prevent training failure
        targets = ['target_hit', 'stop_hit', 'target_type', 'hit_first']
        keep_cols += [t for t in targets if t in df.columns]
        
        # Filter the DataFrame
        df = df[keep_cols]
        # ---------------------------------------------------------

        # 4. Summary Stats
        print("\nFinal Production Feature Summary:")
        print(f"Total Columns kept: {len(df.columns)}")
        print(f"Features list: {list(df.columns)}")
        
        # 5. Save
        output_path = PROCESSED_DATA_DIR / "step3_features_engineered.csv"
        df.to_csv(output_path, index=False)
        print(f"\nSaved production-ready data to: {output_path}")
        print("Engineered Columns:", list(df.columns))
        
    except Exception as e:
        print(f"❌ Error in feature engineering pipeline: {e}")


# def main():
#     """
#     Main execution pipeline.
#     """
#     try:
#         # 1. Load Data
#         input_path = PROCESSED_DATA_DIR / "step1_quality_checked.csv"
#         if not input_path.exists():
#              raise FileNotFoundError(f"Input file not found: {input_path}")
             
#         print(f"📂 Loading data from {input_path}...")
#         df = pd.read_csv(input_path)
        
#         # 2. Feature Engineering Pipeline
#         df = calculate_price_action_features(df)
#         df = calculate_volatility_features(df)
#         df = calculate_momentum_features(df)
        
#         # 3. Feature Selection
#         df = perform_feature_selection(df)
        
#         # 4. Summary Stats
#         print("\n📊 Final Feature Summary:")
#         print(f"Price Action:  {len([c for c in df.columns if 'candle' in c or 'ratio_' in c or 'wick' in c])}")
#         print(f"Volatility:    {len([c for c in df.columns if 'rolling_' in c or 'bb_' in c or 'atr' in c or 'z_score' in c])}")
#         print(f"Momentum:      {len([c for c in df.columns if 'rsi' in c or 'macd' in c or 'roc_' in c])}")
#         print(f"Total Columns: {len(df.columns)}")
        
#         # 5. Save
#         output_path = PROCESSED_DATA_DIR / "step3_features_engineered.csv"
#         df.to_csv(output_path, index=False)
#         print(f"\n💾 Saved engineered data to: {output_path}")
        
#     except Exception as e:
#         print(f"❌ Error in feature engineering pipeline: {e}")
#         # raise e # Optional: Raise if you want stack trace

if __name__ == "__main__":
    main()
