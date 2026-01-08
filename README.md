# AI Trading Signal Prediction System

> **Dual-model probabilistic machine-learning system for candle-based trading decisions, combining profit targeting and risk control with calibrated outputs and production-ready deployment.**

---

## Overview
**AI Trading Signal Prediction** is an end-to-end machine learning system designed around a **dual-target architecture**, explicitly separating **profit realization** from **risk control**.

Instead of treating trading as a single binary classification problem, the system models:
- **Target-Hit probability** – likelihood of a trade reaching its profit objective.
- **Stop-Hit probability** – likelihood of a trade failing via stop-loss.

This design reflects real trading behavior, where conditions that generate profitable trades differ structurally from those that generate losses.

The system relies exclusively on **price-action and volume-derived candle structure features**, deliberately avoiding lag-heavy or overly complex indicators to ensure interpretability and robustness under live, time-dependent market conditions.

Model outputs are **fully probabilistic and calibrated**, enabling threshold-based decision logic rather than naive nter/NOT signals.

---

## Pipeline Steps

1. **Data Cleaning & Leakage Control**  
   Strict chronological handling, removal of future-dependent information, and validation of candle alignment.

2. **Feature Engineering**  
   Construction of candle-structure representations capturing market momentum, volatility, and price positioning within individual candles.

3. **Time-Aware Splitting**  
   Chronological train / validation / test splits to simulate real trading conditions and prevent temporal leakage.

4. **Model Training & Calibration**  
   Gradient-boosted tree models trained separately for each target, followed by probability calibration to ensure reliable confidence estimates.

5. **Backtesting**  
   Realistic evaluation under transaction costs, slippage, and controlled risk exposure (non-optimistic assumptions).

6. **Paper Trading API**  
   RESTful inference service built with FastAPI for real-time signal evaluation.

7. **Dockerized Deployment**  
   Reproducible, environment-independent deployment using Docker.
   
---

## Model Evaluation Philosophy

Traditional accuracy-based metrics were intentionally de-emphasized due to class imbalance and asymmetric trading costs.

Instead, evaluation focused on:
- **Precision & Precision@K** for entry quality
- **False-positive minimization** for risk filtering
- **Train-validation stability** across time splits
- **Probability calibration reliability**

This ensures that model performance translates into **economic value**, not just statistical scores.

---

## Key Highlights

- Probability‑calibrated model (XGB Tuned v1.5)  
- Handles nonlinear market patterns and feature interactions  
- Trading‑aware evaluation (Sharpe, Drawdown, ROI)  
- Stable generalization across temporal splits ( Δ F1 ≈ 0.007 )  
- Production API + Docker deployment ready

---

## Risks & Limitations

Despite strong empirical performance, several limitations are acknowledged:

- Sensitivity to market regime shifts  
- Limited historical coverage  
- Class imbalance inherent to trading outcomes  
- Theoretical nature of backtest returns  

Mitigation strategies include periodic retraining, forward-only validation, and conservative deployment thresholds.

---

## Deployment ( FastAPI + Docker )

**Local API Launch**  
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
├─ data/                # Raw and processed datasets
├─ api/                 # FastAPI inference service
├─ models/              # Trained models & metadata
├─ src/                 # Feature engineering & training logic
├─ decision/            # Risk-aware decision layer
├─ tests/               # Model validation & backtesting scripts
├─ analysis/            # Explainability & SHAP analysis
├─ reports/             # Technical documentation & plots
├─ config.py
├─ requirements.txt
├─ Dockerfile
└─ README.md

```

### Summary Statement
This project demonstrates a trading-first machine learning approach, where model decisions are justified by economic impact rather than raw metrics.
The combination of:
* High-precision entry filtering
* Conservative risk blocking
* Interpretable model behavior

results in a system suitable for decision support and semi-automated trading, under controlled risk management.
