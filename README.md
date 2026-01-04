# AI Trading Signal Prediction System

> **Probabilistic machine-learning pipeline for generating calibrated trading signals, with realistic backtesting and a production-ready FastAPI service.**

---

## Overview
**AI Trading Signal Prediction** is an end‑to‑end machine learning project that predicts *target‑hit* events (successful trades) using market‑based technical features.  
The goal is to produce **probabilistic trading signals** that remain robust under live, time‑dependent conditions, supported by full calibration, monitoring, and API deployment.

---

## Pipeline Steps

1. **Data Cleaning & Leakage Control** – strict chronological handling, future‑feature removal.  
2. **Feature Engineering** – momentum (RSI), volatility (ATR), candle structure ratios.  
3. **Time‑Aware Splitting** – chronological train / validation / test windows.  
4. **Model Training & Calibration** – XGBoost ensemble + isotonic mapping for probability calibration.  
5. **Backtesting** – realistic ROI under slippage, fees & risk fraction (not optimistic compounding).  
6. **Paper Trading API** – REST interface via FastAPI for real‑time inference.  
7. **Dockerized Deployment** – reproducible build from `Dockerfile`.

---

## Key Highlights

- Probability‑calibrated model (XGB Tuned v1.5)  
- Handles nonlinear market patterns and feature interactions  
- Trading‑aware evaluation (Sharpe, Drawdown, ROI)  
- Stable generalization across temporal splits ( Δ F1 ≈ 0.007 )  
- Production API + Docker deployment ready

---

## Model Performance Summary

| Metric | Validation | Test | Comment |
|---------|-------------|------|----------|
| **F1 Score** | 0.878 | 0.871 | Stable across time |
| **AUC** | 0.83 | 0.81 | Good discrimination |
| **ROI (Backtest)** | ≈ 68 000 % (statistical) | Relative metric only |
| **Top Features** | `rsi_3d`, `RSI`, `volume`, `atr_1h`, `ratio_high_low` | Momentum + Volatility structure |

---

## Model Choice Justification

After benchmarking several algorithms — Logistic Regression, Random Forest, LightGBM, and XGBoost — the tuned **XGBoost** model achieved the most stable balance between accuracy and generalization.

**Why XGBoost?**

- Captures nonlinear market effects and feature interactions  
- Robust to outliers and unscaled data  
- Simple probability calibration via isotonic mapping  
- Generalizes well over time‑based splits (Δ F1 ≈ 0.007)

---

## Explainability (Feature Importance via SHAP)

| Feature | Economic meaning | Rank |
|----------|------------------|------|
| rsi_3d | Short‑term momentum (3 days) | 1 |
| RSI | Overall momentum oscillator | 2 |
| volume | Market participation | 3 |
| atr_1h | 1‑hour volatility (ATR) | 4 |
| ratio_high_low | Intra‑candle compression | 5 |

Reports: [`reports/shap_summary_beeswarm.png`](reports/shap_summary_beeswarm.png)

---

## Risks & Limitations

| Category | Description | Mitigation |
|-----------|--------------|-------------|
| **Data Bias & Coverage** | Limited historical window, single asset | Retrain periodically & validate multi‑symbol |
| **Class Imbalance** | ~30 % positive samples (“trade hit”) | Use `class_weight='balanced'` |
| **Temporal Leakage Risk** | Removed time_bin & future features | Live monitoring for integrity |
| **Market Regime Shift** | Breakdown under new volatility regimes | Rolling retraining + stress testing |
| **Backtest Over‑optimism** | 68 K % ROI is theoretical | Treat as relative metric only |
| **Limited Realtime Explainability** | SHAP offline only for speed | Keep API lightweight for latency |

---

## Deployment ( FastAPI + Docker )

**FastAPI Launch (local)**  
```bash
uvicorn api.phase8_paper_trading_api:app --reload
```
Once running:

* Health check → http://localhost:8000/
* Swagger Docs → http://localhost:8000/docs
* POST Endpoint → /predict


## Project Structure
```bash
AI-Trading-Signal-Prediction/
│
├─ data/                                   # Datasets
│   ├─ processed/                          # Clean and feature-ready data
│   └─ raw/                                # Raw market data (unprocessed)
│
├─ api/                                    # REST API modules
│   ├─ phase8_paper_trading_api.py         # FastAPI service (main production endpoint)
│   └─ api_signal_predictor.py             # Auxiliary or experimental API version
│
├─ models/                                 # Trained models & metadata
│   ├─ model_target_hit_final_calibrated.pkl
│   └─ metadata.json
│
├─ notebooks/                              # Research & analysis notebooks
│   ├─ 01_data_quality_and_leakage_control.ipynb
│   ├─ 02_EDA_and_visual_analysis.ipynb
│   └─ shap_explanation.ipynb
│
├─ src/                       # Feature engineering & modeling scripts
│   ├─ phase3_feature_engineering.py
│   ├─ phase4_preprocessing_stop_hit.py
│   ├─ phase4_preprocessing_target_hit.py
│   └─ phase5_models/
│       ├─ model_train_stop_hit.py
│       └─ model_train_target_hit.png
│
├─ reports/                                # Visual explainability reports & plots
│   ├─ shap_summary_beeswarm.png
│   └─ shap_feature_importance_bar.png
│
├─ config.py                               # Global paths and constants
├─ requirements.txt                        # Python dependencies
├─ Dockerfile                              # API containerization setup
└─ README.md                               # Project documentation

```
### Summary Statement
Final Target_Hit Model (XGB Tuned v1.5 – Calibrated) achieved:

* Stable out‑of‑sample performance (F1 ≈ 0.87, AUC ≈ 0.81)
* Intuitive feature drivers (market momentum + volatility)
* Ready for production deployment under controlled risk.
