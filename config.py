from pathlib import Path

# Project Metadata
MODEL_VERSION = "v1.0"
CREATED_BY = "Hana Alaa"
CREATED_DATE = "2025-12-31"

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "trades.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
ARTIFACTS_DIR = MODELS_DIR  # or separate if needed

# Create directories if they don't exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Feature Configuration
FEATURES_PRICE = [
    "entry_price", "open", "high", "low", "close", "volume", 
    "candle_body", "candle_wick"
]
FEATURES_TECHNICAL = ["rsi", "rsi_3d"]
FEATURES_ENGINEERED = [
    "price_move_ratio", "candle_range", "volatility_pct", "rsi_momentum"
]

# Modelling Config
TARGET_THRESHOLD = 0.5  # Initial Default
STOP_THRESHOLD = 0.5    # Initial Default
TEST_SIZE = 0.15
VALID_SIZE = 0.15
RANDOM_STATE = 42

# Drifts / Alerts
DRIFT_REPORT_PATH = LOGS_DIR / "data_drift_report.csv"
DECISION_LOG_PATH = LOGS_DIR / "decision_log.csv"
