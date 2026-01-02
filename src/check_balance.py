import pandas as pd
from pathlib import Path

splits_dir = Path("data/processed/splits")
for split in ["train_stop_hit", "valid_stop_hit","test_stop_hit"]:
    path = splits_dir / f"{split}.csv"
    if path.exists():
        df = pd.read_csv(path)
        print(f"\nTarget distribution for {split}:")
        print(df['stop_hit'].value_counts(normalize=True))
        print(df['stop_hit'].value_counts())
    else:
        print(f"File not found: {path}")
