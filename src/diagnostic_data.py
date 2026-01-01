import pandas as pd
from pathlib import Path

train_path = Path("data/processed/splits/train.csv")
try:
    df = pd.read_csv(train_path, nrows=5)
    print("Train slice:")
    print(df.head())
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nMissing values (first 1000 rows):")
    df_full = pd.read_csv(train_path, nrows=1000)
    print(df_full.isnull().sum().sum())
except Exception as e:
    print(f"Error reading train.csv: {e}")
