import pandas as pd
import numpy as np
import sys, joblib, json, matplotlib.pyplot as plt
from pathlib import Path

# ==============================================================
# ✅ Setup
# ==============================================================
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from config import PROCESSED_DATA_DIR, MODELS_DIR

# ==============================================================
# ✅ Wrapper for Calibrated Model
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
# ✅ Phase 7.4 – Institutional Backtest (Real Daily Sharpe + Cooldown)
# ==============================================================
def main():
    print("🏁 STARTING PHASE 7.4 : TRUE DAILY‑SHARPE BACKTEST\n")

    # Paths
    logs_dir = project_root / "logs"
    back_dir = logs_dir / "backtest"
    back_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # Load metadata & config
    # ----------------------------------------------------------
    meta = json.load(open(MODELS_DIR / "metadata.json"))
    best_thr = meta["target_hit_model"]["metrics_val"]["best_threshold"]
    features = meta["target_hit_model"]["features"]

    print(f"⚙️ CONFIGURATION")
    print(f"   • Threshold = {best_thr}")
    print(f"   • Initial Capital = 10 000 $")
    print(f"   • Risk per Trade = 0.2 % (compounded)")
    print(f"   • Daily limit = 5 trades + 15‑min cool‑down")
    print(f"   • Sharpe based on true daily returns")
    print("-"*65)

    # ----------------------------------------------------------
    # Load data & model
    # ----------------------------------------------------------
    df = pd.read_csv(PROCESSED_DATA_DIR / "splits" / "test.csv")
    if "target_hit" not in df.columns:
        raise ValueError("Missing target_hit column")

    X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    import __main__; __main__.CalibratedModelWrapper = CalibratedModelWrapper
    model = joblib.load(MODELS_DIR / "model_target_hit_final_calibrated.pkl")

    # ----------------------------------------------------------
    # Generate signals
    # ----------------------------------------------------------
    df["signal"] = (model.predict_proba(X)[:, 1] >= best_thr).astype(int)
    trades = df[df["signal"] == 1].copy()
    if trades.empty:
        print("❌ No signals generated.")
        return

    # ----------------------------------------------------------
    # Exposure Control: Daily limit + 15 min cool‑down
    # ----------------------------------------------------------
    if "created_at" in trades.columns:
        trades["created_at"] = pd.to_datetime(trades["created_at"])
        trades.sort_values("created_at", inplace=True)
        limit = pd.Timedelta("15min")
        filtered = []
        last_time = pd.Timestamp.min
        for _, row in trades.iterrows():
            if row["created_at"] - last_time >= limit:
                filtered.append(row)
                last_time = row["created_at"]
        trades = pd.DataFrame(filtered)
        trades["trade_date"] = trades["created_at"].dt.date
        trades = trades.groupby("trade_date").head(5)
        trades.reset_index(drop=True, inplace=True)

    # ----------------------------------------------------------
    # Simulation Parameters
    # ----------------------------------------------------------
    init_cap = 10_000
    risk_frac = 0.002
    reward_risk = 2.0
    fees = 0.001
    max_gain, max_loss = 0.05, -0.05

    equity = init_cap
    pnl_usd, trade_returns, equity_curve = [], [], []

    for _, r in trades.iterrows():
        risk_amt = equity * risk_frac
        pnl = (risk_amt * reward_risk) - (risk_amt * fees) if r["target_hit"] == 1 else (-risk_amt) - (risk_amt * fees)
        ret = np.clip(pnl / equity, max_loss, max_gain)
        equity *= (1 + ret)
        pnl_usd.append(pnl)
        trade_returns.append(ret)
        equity_curve.append(equity)

    trades["pnl_usd"] = pnl_usd
    trades["return_pct"] = np.array(trade_returns) * 100
    trades["equity"] = equity_curve

    # ----------------------------------------------------------
    # 📊 Metrics
    # ----------------------------------------------------------
    total_trades = len(trades)
    win_rate = (trades["pnl_usd"] > 0).mean() * 100
    net_profit = sum(pnl_usd)
    final_equity = equity_curve[-1]
    roi = (final_equity / init_cap - 1) * 100

    eq = pd.Series(equity_curve)
    running_max = eq.cummax()
    drawdown = (eq - running_max) / running_max
    max_dd = drawdown.min() * 100

    # CAGR from time span
    if "created_at" in trades.columns:
        days = (trades["created_at"].iloc[-1] - trades["created_at"].iloc[0]).days
        years = max(days / 365, 1/365)
    else:
        years = total_trades / 252
    cagr = ((final_equity / init_cap) ** (1 / years) - 1) * 100

    # --- True Daily Returns for Sharpe/Sortino
    if "created_at" in trades.columns:
        daily_pnl = trades.set_index("created_at")["pnl_usd"].resample("1D").sum()
        daily_returns = daily_pnl / daily_pnl.shift(1).fillna(init_cap)
    else:
        daily_returns = pd.Series(trade_returns).groupby(np.arange(total_trades)//5).mean()

    avg_ret = daily_returns.mean()
    std_ret = max(daily_returns.std(), 1e-8)
    sharpe = (avg_ret / std_ret) * np.sqrt(252)
    downside = daily_returns[daily_returns < 0].std()
    downside = downside if downside and not np.isnan(downside) and downside > 1e-8 else std_ret
    sortino = min((avg_ret / downside) * np.sqrt(252), 10)   # Cap Sortino ≤ 10
    rr_ratio = trades.loc[trades["pnl_usd"]>0,"pnl_usd"].mean() / abs(trades.loc[trades["pnl_usd"]<=0,"pnl_usd"].mean())

    # ----------------------------------------------------------
    # 💵 Summary Report
    # ----------------------------------------------------------
    print("="*65)
    print("🏦  BACKTEST PERFORMANCE REPORT (Phase 7.4)")
    print("="*65)
    print(f"🔹 Initial Capital:   ${init_cap:,.2f}")
    print(f"🔹 Final Equity:      ${final_equity:,.2f}")
    print(f"🔹 Net Profit:        ${net_profit:,.2f}")
    print(f"🔹 ROI:               {roi:.2f}%")
    print(f"🔹 CAGR:              {cagr:.2f}%")
    print("-"*35)
    print(f"🔹 Trades Executed:   {total_trades}")
    print(f"🔹 Win Rate:          {win_rate:.2f}%")
    print(f"🔹 Reward/Risk:       {rr_ratio:.2f}")
    print("-"*35)
    print(f"🔹 Max Drawdown:      {max_dd:.2f}%")
    print(f"🔹 Sharpe Ratio:      {sharpe:.2f}")
    print(f"🔹 Sortino Ratio:     {sortino:.2f}")
    print("="*65)

    # ----------------------------------------------------------
    # Charts
    # ----------------------------------------------------------
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(eq, color="green")
    plt.title("Equity Curve (Compounded 0.2%)")
    plt.xlabel("Trade #"); plt.ylabel("Equity ($)")
    plt.grid(True, alpha=0.3)
    plt.subplot(1,2,2)
    plt.fill_between(range(len(drawdown)), drawdown*100, color="red", alpha=0.3)
    plt.title(f"Drawdown (%) | Max {max_dd:.2f}%")
    plt.tight_layout()
    plt.savefig(back_dir/"financial_backtest_phase7.4.png"); plt.close()

    plt.hist(daily_returns*100, bins=40, color="blue", alpha=0.7)
    plt.title("Daily Returns Distribution (%)")
    plt.xlabel("Return %"); plt.ylabel("Freq")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(back_dir/"return_distribution_phase7.4.png"); plt.close()

    # ----------------------------------------------------------
    # Save metrics
    # ----------------------------------------------------------
    results = dict(
        initial_capital=init_cap, final_equity=final_equity,
        net_profit=net_profit, roi_pct=roi, cagr_pct=cagr,
        win_rate=win_rate, reward_risk_ratio=rr_ratio,
        max_drawdown_pct=max_dd, sharpe_ratio=sharpe,
        sortino_ratio=sortino, total_trades=total_trades,
        evaluated_at=str(pd.Timestamp.now())
    )
    json.dump(results, open(back_dir/"financial_metrics_phase7.4.json","w"), indent=4)

    # ----------------------------------------------------------
    # Verdict
    # ----------------------------------------------------------
    print("\n🧐 FINAL VERDICT:")        
    if net_profit <= 0:
        print("❌ Strategy Losing.")    
    elif abs(max_dd) > 20:
        print("⚠️ Profitable but High Risk.")
    else:
        print("✅ Strategy Stable & Ready for Paper Trading Verification.")
    print("="*65)


if __name__ == "__main__":
    main()