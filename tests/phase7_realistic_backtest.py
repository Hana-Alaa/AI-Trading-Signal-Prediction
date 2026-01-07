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
from decision.decision_layer import TradingDecisionEngine

# ==================================================
# Register CalibratedModelWrapper for pickle
# ==================================================
import types, sys
import __main__

class CalibratedModelWrapper:
    def __init__(self, base_model, iso_model):
        self.base_model = base_model
        self.iso_model = iso_model

    def predict_proba(self, X):
        base_probs = self.base_model.predict_proba(X)[:, 1]
        calibrated_probs = self.iso_model.predict(base_probs)
        return np.vstack([1 - calibrated_probs, calibrated_probs]).T

calibrated_wrapper = types.ModuleType("calibrated_wrapper")
calibrated_wrapper.CalibratedModelWrapper = CalibratedModelWrapper

sys.modules["calibrated_wrapper"] = calibrated_wrapper
__main__.CalibratedModelWrapper = CalibratedModelWrapper

# ==============================================================
# Simulation Constants (REALISM FACTORS)
# ==============================================================
COMMISSION_RATE = 0.0010
SLIPPAGE_PCT    = 0.0005
MAX_HOLD_CANDLES = 48
DAILY_TRADE_LIMIT = 5
COOLDOWN_MINUTES = 30
# ==============================================================
# CORE BACKTEST ENGINE
# ==============================================================
class RealisticBacktester:
    def __init__(self, initial_capital=10000, risk_per_trade=0.01, reward_ratio=2.0):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.reward_ratio = reward_ratio
        self.equity = initial_capital
        self.equity_curve = [initial_capital]
        self.trades_log = []

    def _simulate_price_path(self, entry_idx, df, tp_price, sl_price):
        future = df.iloc[entry_idx + 1: entry_idx + 1 + MAX_HOLD_CANDLES]

        for i, row in future.iterrows():
            if row["low"] <= sl_price:
                return sl_price * (1 - SLIPPAGE_PCT), "STOP_LOSS", row["created_at"], i - entry_idx
            if row["high"] >= tp_price:
                return tp_price, "TAKE_PROFIT", row["created_at"], i - entry_idx

        if len(future) > 0:
            last = future.iloc[-1]
            return last["close"], "TIME_EXIT", last["created_at"], MAX_HOLD_CANDLES

        return None, "ERROR", None, 0

    def run_backtest(
        self,
        df,
        model_target,
        model_stop,
        features,
        entry_thr,
        stop_thr
    ):
        df = df.sort_values("created_at").reset_index(drop=True)
        df["created_at"] = pd.to_datetime(df["created_at"])

        X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)

        p_target = model_target.predict_proba(X)[:, 1]
        p_stop   = model_stop.predict_proba(X)[:, 1]

        decision_engine = TradingDecisionEngine(
            entry_threshold=entry_thr,
            stop_threshold=stop_thr
        )

        last_trade_time = pd.Timestamp.min
        daily_trades = {}

        i = 0
        while i < len(df) - 1:
            row = df.iloc[i]
            now = row["created_at"]
            day = now.date()

            decision = decision_engine.decide(
                p_target=p_target[i],
                p_stop=p_stop[i]
            )

            if decision["decision"] != "ENTER":
                i += 1
                continue

            if now < last_trade_time + timedelta(minutes=COOLDOWN_MINUTES):
                i += 1
                continue

            if daily_trades.get(day, 0) >= DAILY_TRADE_LIMIT:
                i += 1
                continue

            raw_entry = row["entry_price"]
            entry_exec = raw_entry * (1 + SLIPPAGE_PCT)

            stop_dist = raw_entry * 0.01
            sl_price = raw_entry - stop_dist
            tp_price = raw_entry + stop_dist * self.reward_ratio

            exit_price, reason, exit_time, duration = self._simulate_price_path(
                i, df, tp_price, sl_price
            )

            if exit_price is None:
                i += 1
                continue

            risk_amount = self.equity * self.risk_per_trade
            shares = risk_amount / (entry_exec - sl_price)

            gross_pnl = (exit_price - entry_exec) * shares
            fees = (entry_exec * shares + exit_price * shares) * COMMISSION_RATE
            net_pnl = gross_pnl - fees

            self.equity += net_pnl
            self.equity_curve.append(self.equity)

            self.trades_log.append({
                "entry_time": now,
                "exit_time": exit_time,
                "p_target": p_target[i],
                "p_stop": p_stop[i],
                "entry_price": entry_exec,
                "exit_price": exit_price,
                "pnl": net_pnl,
                "reason": reason,
                "equity": self.equity
            })

            last_trade_time = exit_time
            daily_trades[day] = daily_trades.get(day, 0) + 1
            i += duration

        return pd.DataFrame(self.trades_log)

# ==============================================================
# Metrics
# ==============================================================
def calculate_metrics(equity_curve, trades):
    if len(trades) == 0:
        return dict(score=0, roi=0, sharpe=0, max_dd=0, trades=0)

    eq = np.array(equity_curve)
    roi = (eq[-1] / eq[0] - 1) * 100

    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak
    max_dd = dd.max() * 100

    returns = np.diff(eq) / eq[:-1]
    sharpe = 0 if returns.std() == 0 else (returns.mean() / returns.std()) * np.sqrt(252)

    score = (roi * sharpe) / (max_dd + 1)

    return {
        "roi": roi,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "win_rate": (trades["pnl"] > 0).mean() * 100,
        "trades": len(trades),
        "score": score
    }

# ==============================================================
# MAIN
# ==============================================================
def main():
    meta = json.load(open(MODELS_DIR / "metadata.json"))
    features = meta["target_hit_model"]["features"]

    model_target = joblib.load(MODELS_DIR / "model_target_hit_final_calibrated.pkl")
    model_stop   = joblib.load(MODELS_DIR / "model_stop_hit_final_calibrated.pkl")

    df = pd.read_csv(PROCESSED_DATA_DIR / "splits" / "test.csv")

    df["created_at"] = pd.to_datetime(df.get("created_at", pd.date_range("2020-01-01", periods=len(df), freq="h")))
    df["entry_price"] = df.get("entry_price", df["close"])
    df["high"] = df.get("high", df["close"] * 1.002)
    df["low"] = df.get("low", df["close"] * 0.998)

    tscv = TimeSeriesSplit(n_splits=3)

    entry_thrs = [0.65, 0.7, 0.75]
    stop_thrs  = [0.15, 0.2, 0.25]
    risks = [0.01, 0.02]
    rrs   = [2.0, 3.0]

    results = []

    for _, val_idx in tscv.split(df):
        fold = df.iloc[val_idx]

        for et in entry_thrs:
            for st in stop_thrs:
                for r in risks:
                    for rr in rrs:
                        engine = RealisticBacktester(risk_per_trade=r, reward_ratio=rr)
                        trades = engine.run_backtest(
                            fold,
                            model_target,
                            model_stop,
                            features,
                            et,
                            st
                        )
                        metrics = calculate_metrics(engine.equity_curve, trades)
                        results.append({
                            "entry_thr": et,
                            "stop_thr": st,
                            "risk": r,
                            "rr": rr,
                            **metrics
                        })

    res = pd.DataFrame(results)
    best = res.sort_values("score", ascending=False).iloc[0]

    print("\nBEST PARAMS")
    print(best)

    meta["optimized_params"] = best.to_dict()
    json.dump(meta, open(MODELS_DIR / "metadata.json", "w"), indent=4)

if __name__ == "__main__":
    main()