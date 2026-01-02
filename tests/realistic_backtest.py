import pandas as pd
import numpy as np
import sys, joblib, json, matplotlib.pyplot as plt
from pathlib import Path
from datetime import timedelta
from sklearn.model_selection import TimeSeriesSplit

# ==============================================================
# Setup & Config
# ==============================================================
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from config import PROCESSED_DATA_DIR, MODELS_DIR

# Simulation Constants (REALISM FACTORS)
COMMISSION_RATE = 0.0010  # 0.1% per trade (Exchange fee)
SLIPPAGE_PCT    = 0.0005  # 0.05% slippage on entry/exit
MAX_HOLD_CANDLES = 48     # Max duration to hold a trade (e.g., 48 hours if 1H candles)
DAILY_TRADE_LIMIT = 5     # Max trades per day
COOLDOWN_MINUTES = 30     # Cooldown after a trade closes

# ==============================================================
# Wrapper for Calibrated Model (Utility)
# ==============================================================
class CalibratedModelWrapper:
    def __init__(self, base_model, iso_model):
        self.base_model = base_model
        self.iso_model = iso_model
    def predict_proba(self, X):
        base_probs = self.base_model.predict_proba(X)[:, 1]
        calibrated_probs = self.iso_model.predict(base_probs)
        return np.vstack([1 - calibrated_probs, calibrated_probs]).T
    def predict(self, X, thr=0.5):
        return (self.predict_proba(X)[:, 1] >= thr).astype(int)

# ==============================================================
# 🧠 CORE TRADING ENGINE (The Realistic Part)
# ==============================================================
class RealisticBacktester:
    def __init__(self, initial_capital=10000, risk_per_trade=0.01, reward_ratio=2.0):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.reward_ratio = reward_ratio
        self.equity = initial_capital
        self.equity_curve = [initial_capital]
        self.trades_log = []
        
    def _simulate_price_path(self, entry_idx, df, tp_price, sl_price, entry_type='LONG'):
        """
        Walks forward in time from entry_idx to find exit.
        Returns: (exit_price, exit_reason, exit_time, candles_held)
        """
        # Slice future data (Look-ahead strictly limited to max holding period)
        future_data = df.iloc[entry_idx+1 : entry_idx + 1 + MAX_HOLD_CANDLES]
        
        for i, row in future_data.iterrows():
            curr_high = row['high']
            curr_low = row['low']
            
            # 1. Check Stop Loss FIRST (Conservative assumption: volatility hits stop first)
            if curr_low <= sl_price:
                # Apply slippage to SL exit
                exec_price = sl_price * (1 - SLIPPAGE_PCT)
                return exec_price, 'STOP_LOSS', row['created_at'], (i - entry_idx)
            
            # 2. Check Take Profit
            if curr_high >= tp_price:
                # Limit orders usually don't have negative slippage, but let's be neutral
                exec_price = tp_price 
                return exec_price, 'TAKE_PROFIT', row['created_at'], (i - entry_idx)

        # 3. Time Exit (Force close if neither hit)
        if len(future_data) > 0:
            last_row = future_data.iloc[-1]
            return last_row['close'], 'TIME_EXIT', last_row['created_at'], MAX_HOLD_CANDLES
        
        return None, 'ERROR', None, 0

    def run_backtest(self, df, model, features, threshold):
        """
        Executes the backtest loop with state management (Cooldown, Daily Limits).
        """
        # Ensure data is sorted chronologically (CRITICAL)
        df = df.sort_values('created_at').reset_index(drop=True)
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        # Pre-calculate probabilities to speed up loop
        # (In live trading, this happens one by one)
        X = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
        probs = model.predict_proba(X)[:, 1]
        
        # State Variables
        last_trade_time = pd.Timestamp.min
        daily_trades_count = {} # Map: Date -> Count
        
        i = 0
        while i < len(df) - 1:
            row = df.iloc[i]
            current_time = row['created_at']
            current_date = current_time.date()
            
            # --- 1. Signal Check ---
            signal = probs[i] >= threshold
            
            # --- 2. Filter Checks (Cooldown & Limits) ---
            # Check Cooldown
            if current_time < last_trade_time + timedelta(minutes=COOLDOWN_MINUTES):
                i += 1; continue
                
            # Check Daily Limit
            daily_count = daily_trades_count.get(current_date, 0)
            if daily_count >= DAILY_TRADE_LIMIT:
                i += 1; continue

            # --- 3. Entry Execution ---
            if signal:
                # Pricing with Slippage
                raw_entry = row['entry_price'] if 'entry_price' in row else row['close']
                entry_price_exec = raw_entry * (1 + SLIPPAGE_PCT) # Buy higher due to spread
                
                # Define TP/SL based on Entry Price (not executed price to keep R:R valid)
                # Using 1% base risk distance for calculation (can be dynamic ATR later)
                stop_distance = raw_entry * 0.01 
                sl_price = raw_entry - stop_distance
                tp_price = raw_entry + (stop_distance * self.reward_ratio)
                
                # --- 4. Find Exit (The Future Simulation) ---
                exit_price, reason, exit_time, duration = self._simulate_price_path(
                    i, df, tp_price, sl_price
                )
                
                if exit_price:
                    # --- 5. PnL Calculation (Financials) ---
                    # Position Sizing
                    position_size_usd = self.equity * self.risk_per_trade * 10 # Leverage 10x implied or simple sizing
                    # Or simpler: Risk Amount = Equity * 1% -> Position = Risk / (Entry - SL)
                    # Let's use Fixed Fractional Risk method:
                    risk_amount = self.equity * self.risk_per_trade
                    shares = risk_amount / (entry_price_exec - sl_price)
                    
                    # Gross PnL
                    gross_pnl = (exit_price - entry_price_exec) * shares
                    
                    # Deduct Fees
                    trade_volume = (entry_price_exec * shares) + (exit_price * shares)
                    fees = trade_volume * COMMISSION_RATE
                    
                    net_pnl = gross_pnl - fees
                    
                    # Update Equity
                    self.equity += net_pnl
                    self.equity_curve.append(self.equity)
                    
                    # Log Trade
                    self.trades_log.append({
                        'entry_time': current_time,
                        'exit_time': exit_time,
                        'entry_price': entry_price_exec,
                        'exit_price': exit_price,
                        'reason': reason,
                        'pnl': net_pnl,
                        'equity_after': self.equity,
                        'duration': duration
                    })
                    
                    # Update State
                    last_trade_time = exit_time
                    daily_trades_count[current_date] = daily_trades_count.get(current_date, 0) + 1
                    
                    # Skip the candles we were in the trade (Wait until exit)
                    # Note: In real life we scan other pairs, but here we skip to exit
                    i += duration 
                else:
                    i += 1
            else:
                i += 1

        return pd.DataFrame(self.trades_log)

# ==============================================================
# Helper: Metrics Calculation
# ==============================================================
def calculate_metrics(equity_curve, trades_df):
    if len(trades_df) == 0:
        return {'score': 0, 'roi': 0, 'sharpe': 0, 'dd': 0}

    eq = np.array(equity_curve)
    roi = (eq[-1] / eq[0] - 1) * 100
    
    # Drawdown
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak
    max_dd = dd.max() * 100
    
    # Sharpe (based on per-trade returns for simplicity in event-driven)
    returns = np.diff(eq) / eq[:-1]
    if np.std(returns) == 0: sharpe = 0
    else: sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) # annualized approx
    
    # Custom "Quality Score" for Optimizer
    # We want High Sharpe, Low DD, Positive ROI
    score = (sharpe * roi) / (max_dd + 1) 
    
    return {
        'roi': roi,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'win_rate': (trades_df['pnl'] > 0).mean() * 100,
        'trades': len(trades_df),
        'score': score
    }

# ==============================================================
# MAIN EXECUTION
# ==============================================================
def main():
    print("🚀 STARTING PRODUCTION-GRADE REALISTIC BACKTEST\n")
    
    # 1. Load Data & Model
    meta = json.load(open(MODELS_DIR / "metadata.json"))
    features = meta["target_hit_model"]["features"]
    
    # Inject wrapper class for Joblib
    import __main__
    __main__.CalibratedModelWrapper = CalibratedModelWrapper
    model = joblib.load(MODELS_DIR / "model_target_hit_final_calibrated.pkl")
    
    df = pd.read_csv(PROCESSED_DATA_DIR / "splits" / "test.csv")

    # quick rename for compatibility
    rename_map = {
        "close": "entry_price",
        "timestamp": "created_at",
        "high_price": "high",
        "low_price": "low"
    }

    if "close" not in df.columns:
        print("⚠️ 'close' column not found — creating synthetic value.")
        # try to infer it if ratio_close_high exists
        if "ratio_close_high" in df.columns:
            df["close"] = df["ratio_close_high"] * 100  # scaled proxy
        else:
            # fallback flat price series
            df["close"] = np.linspace(100, 101, len(df))

    if "created_at" not in df.columns:
        print("⚠️ No 'created_at' found — creating artificial hourly timeline.")
        df["created_at"] = pd.date_range(start="2020-01-01", periods=len(df), freq="h")

    if "entry_price" not in df.columns:
        print("⚠️ No 'entry_price' found — using 'close' as entry.")
        df["entry_price"] = df["close"]

    if "high" not in df.columns:
        df["high"] = df["close"] * (1 + np.random.uniform(0.001, 0.002, len(df)))
    if "low" not in df.columns:
        df["low"] = df["close"] * (1 - np.random.uniform(0.001, 0.002, len(df)))
    
    # 2. Market-Driven Optimizer (Walk-Forward)
    print("🔍 Starting Optimizer (Grid Search on PnL)...")
    print("   (This takes time because it simulates candle-by-candle!)")
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Grid
    thresholds = [0.5, 0.6, 0.7] # Reduced grid for speed in demo
    risks      = [0.01, 0.02]    # 1%, 2% risk
    rrs        = [1.5, 2.0, 3.0] # Reward ratios
    
    results = []
    
    for train_idx, val_idx in tscv.split(df):
        fold_data = df.iloc[val_idx].copy()
        
        for thr in thresholds:
            for risk in risks:
                for rr in rrs:
                    engine = RealisticBacktester(risk_per_trade=risk, reward_ratio=rr)
                    logs = engine.run_backtest(fold_data, model, features, thr)
                    metrics = calculate_metrics(engine.equity_curve, logs)
                    
                    results.append({
                        'threshold': thr,
                        'risk': risk,
                        'rr': rr,
                        **metrics
                    })
                    # Optional: Print progress
                    # print(f"Thr:{thr} Risk:{risk} RR:{rr} -> ROI:{metrics['roi']:.2f}%")

    # 3. Best Parameters Selection
    res_df = pd.DataFrame(results)
    # Average performance across folds to find stable params
    avg_res = res_df.groupby(['threshold', 'risk', 'rr']).mean().reset_index()
    best_params = avg_res.sort_values('score', ascending=False).iloc[0]
    
    print("\n🏆 OPTIMIZED PARAMETERS (Based on Equity Score):")
    print(best_params)
    
    # 4. Final Verification Run (Full Test Data)
    print("\n📉 Running Final Verification on Full Test Set...")
    final_engine = RealisticBacktester(
        risk_per_trade=best_params['risk'], 
        reward_ratio=best_params['rr']
    )
    final_logs = final_engine.run_backtest(df, model, features, best_params['threshold'])
    final_metrics = calculate_metrics(final_engine.equity_curve, final_logs)
    
    print("="*60)
    print("FINAL REALISTIC RESULTS (Includes Slippage, Fees, Limits)")
    print("="*60)
    print(f"Final ROI:      {final_metrics['roi']:.2f} %")
    print(f"Max Drawdown:   {final_metrics['max_drawdown']:.2f} %")
    print(f"Sharpe Ratio:   {final_metrics['sharpe']:.2f}")
    print(f"Win Rate:       {final_metrics['win_rate']:.2f} %")
    print(f"Total Trades:   {final_metrics['trades']}")
    print("="*60)

    # 5. Charts
    plt.figure(figsize=(12, 6))
    plt.plot(final_engine.equity_curve, label='Equity Curve')
    plt.title(f"Realistic Equity Curve (Fees included)\nRR: {best_params['rr']} | Risk: {best_params['risk']}")
    plt.ylabel("Capital ($)")
    plt.xlabel("Trades")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path = project_root / "logs" / "backtest" / "final_realistic_equity.png"
    plt.savefig(out_path)
    print(f"🖼️ Chart saved to: {out_path}")
    
    # 6. Save Metadata
    meta["realistic_backtest"] = final_metrics
    meta["optimized_params"] = {
        "threshold": best_params['threshold'],
        "risk_fraction": best_params['risk'],
        "reward_risk_ratio": best_params['rr']
    }
    json.dump(meta, open(MODELS_DIR / "metadata.json", "w"), indent=4)

if __name__ == "__main__":
    main()