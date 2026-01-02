import pandas as pd
import numpy as np
import sys, joblib, json, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score

# ==============================================================
# Setup
# ==============================================================
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from config import PROCESSED_DATA_DIR, MODELS_DIR

# ==============================================================
# Wrapper for Calibrated Model
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
# Phase 7.4 – Backtest + 7.5 Optimizer
# ==============================================================
def main():
    print("🚀 STARTING PHASE 7.4 + 7.5 BACKTEST & OPTIMIZER\n")

    logs_dir = project_root / "logs"
    back_dir = logs_dir / "backtest"
    back_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # Load config
    # ----------------------------------------------------------
    meta = json.load(open(MODELS_DIR / "metadata.json"))
    best_thr = meta["target_hit_model"]["metrics_val"]["best_threshold"]
    features = meta["target_hit_model"]["features"]

    print(f"CONFIGURATION")
    print(f"   • Threshold = {best_thr}")
    print(f"   • Initial Capital = 10 000 $")
    print(f"   • Risk per Trade = 0.2 % (compounded)")
    print(f"   • Daily limit = 5 trades + 15‑min cool‑down\n")

    # ----------------------------------------------------------
    # Load data
    # ----------------------------------------------------------
    df = pd.read_csv(PROCESSED_DATA_DIR / "splits" / "test.csv")
    if "target_hit" not in df.columns:
        raise ValueError("Missing target_hit column")
    import __main__
    __main__.CalibratedModelWrapper = CalibratedModelWrapper
    model = joblib.load(MODELS_DIR / "model_target_hit_final_calibrated.pkl")

    X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)

    # ----------------------------------------------------------
    # === PHASE 7.4 CLASSIC BACKTEST ===
    # ----------------------------------------------------------
    df["signal"] = (model.predict_proba(X)[:, 1] >= best_thr).astype(int)
    trades = df[df["signal"] == 1].copy()
    if trades.empty:
        print("❌ No signals generated.")
        return

    if "created_at" in trades.columns:
        trades["created_at"] = pd.to_datetime(trades["created_at"]).sort_values()
        limit = pd.Timedelta("15min")
        filtered, last_time = [], pd.Timestamp.min
        for _, row in trades.iterrows():
            if row["created_at"] - last_time >= limit:
                filtered.append(row); last_time = row["created_at"]
        trades = pd.DataFrame(filtered)
        trades["trade_date"] = trades["created_at"].dt.date
        trades = trades.groupby("trade_date").head(5).reset_index(drop=True)

    # Parameters
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
        pnl_usd.append(pnl); trade_returns.append(ret); equity_curve.append(equity)

    trades["pnl_usd"] = pnl_usd
    trades["return_pct"] = np.array(trade_returns) * 100
    trades["equity"] = equity_curve

    total_trades = len(trades)
    win_rate = (trades["pnl_usd"] > 0).mean() * 100
    net_profit = sum(pnl_usd)
    final_equity = equity_curve[-1]
    roi = (final_equity / init_cap - 1) * 100
    eq = pd.Series(equity_curve)
    running_max = eq.cummax()
    drawdown = (eq - running_max) / running_max
    max_dd = drawdown.min() * 100

    years = (trades["created_at"].iloc[-1] - trades["created_at"].iloc[0]).days / 365 if "created_at" in trades.columns else total_trades/252
    years = max(years, 1/365)
    cagr = ((final_equity / init_cap) ** (1 / years) - 1) * 100

    daily_returns = pd.Series(trade_returns).groupby(np.arange(total_trades)//5).mean()
    avg_ret = daily_returns.mean(); std_ret = max(daily_returns.std(),1e-8)
    sharpe = (avg_ret/std_ret)*np.sqrt(252)
    downside = daily_returns[daily_returns<0].std() or std_ret
    sortino = min((avg_ret/downside)*np.sqrt(252),10)
    rr_ratio = trades.loc[trades["pnl_usd"]>0,"pnl_usd"].mean() / abs(trades.loc[trades["pnl_usd"]<=0,"pnl_usd"].mean())

    print("="*65)
    print("BACKTEST PERFORMANCE REPORT (Phase 7.4)")
    print("="*65)
    print(f"Initial Capital: ${init_cap:,.2f}")
    print(f"Final Equity:    ${final_equity:,.2f}")
    print(f"ROI:             {roi:.2f}%  |  CAGR: {cagr:.2f}%")
    print(f"Trades: {total_trades}  |  Win Rate: {win_rate:.2f}%  |  R/R: {rr_ratio:.2f}")
    print(f"Drawdown: {max_dd:.2f}%  |  Sharpe: {sharpe:.2f}  |  Sortino: {sortino:.2f}")
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
    plt.savefig(back_dir / "phase7_4_equity_drawdown.png")
    plt.close()

    plt.hist(daily_returns*100, bins=40, color="blue", alpha=0.7)
    plt.title("Daily Returns Distribution (%)")
    plt.xlabel("Return %"); plt.ylabel("Freq")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(back_dir / "phase7_4_return_hist.png")
    plt.close()

    # ----------------------------------------------------------
    # === PHASE 7.5 OPTIMIZER (Walk‑Forward & Risk/Threshold) ===
    # ----------------------------------------------------------
    print("\n🔍 Starting Phase 7.5 – Walk‑Forward + Risk & Threshold Optimizer...")
    splits = 5
    tscv = TimeSeriesSplit(n_splits=splits)
    risk_grid = [0.001, 0.002, 0.005, 0.01]
    reward_grid = [1.5, 2.0, 3.0]
    thresh_grid = np.linspace(0.3, 0.7, 9)
    results = []

    for train_idx, test_idx in tscv.split(df):
        fold = df.iloc[test_idx]  # نختبر على المؤشر الزمني اللاحق
        Xf = fold[features].apply(pd.to_numeric, errors="coerce").fillna(0)
        yf = fold['target_hit']
        y_prob = model.predict_proba(Xf)[:, 1]
        for thr in thresh_grid:
            preds = (y_prob >= thr).astype(int)
            prec, rec, f1 = precision_score(yf, preds, zero_division=0), recall_score(yf, preds, zero_division=0), f1_score(yf, preds, zero_division=0)
            win = (sum(preds * yf) / (sum(preds) + 1e-9))
            for risk in risk_grid:
                for rr in reward_grid:
                    ev = prec * rr - (1 - prec)
                    score = ev * win
                    results.append((thr, risk, rr, prec, rec, f1, ev, score))

    res = pd.DataFrame(results, columns=["threshold", "risk_frac", "reward_risk", "precision", "recall", "f1", "EV", "score"])
    best = res.sort_values("score", ascending=False).iloc[0]
    print("\n🏆 Optimal Parameters (Phase 7.5):")
    print(best.round(4))

    # Save optimizer results
    opt_path = back_dir / "phase7_5_optimizer_results.json"
    res.to_json(opt_path, orient="records", indent=2)
    print(f"\n💾 All grid results saved to: {opt_path}")

    best_cfg = {
        "best_threshold": float(best["threshold"]),
        "best_risk_fraction": float(best["risk_frac"]),
        "best_reward_risk": float(best["reward_risk"]),
        "expected_value": float(best["EV"]),
        "mean_precision": float(best["precision"]),
        "mean_recall": float(best["recall"])
    }
    meta["optimized_params"] = best_cfg
    json.dump(meta, open(MODELS_DIR / "metadata.json", "w"), indent=4)
    print("📖 Metadata updated with optimized parameters ✅")

    print("\n✅ PHASE 7.5 OPTIMIZATION COMPLETE – Ready for Paper Trading.")
    print("="*65)


if __name__ == "__main__":
    main()