import os
import csv
from typing import List, Dict
import pandas as pd

def append_csv_row(path: str, fieldnames: List[str], rows: List[Dict[str, str]]):
    exists = os.path.exists(path)
    # Use newline='' when writing CSV files to avoid inserting extra blank lines
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)

def read_csv_index(path: str) -> List[Dict[str, str]]:
    """Read a CSV and return a list of dicts (records). Returns empty list if file missing."""
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    return df.to_dict(orient="records")

def save_dataset_in_batches(path: str, data: pd.DataFrame, batch_size=64):
    """Save `data` into multiple batch files named '<path>_batch_<i>.csv'.
    Each batch file is overwritten to ensure clean data with headers.
    """
    n = (len(data) + batch_size - 1) // batch_size
    for i in range(n):
        start = batch_size * i
        end = batch_size * (i + 1)
        batch = data.iloc[start:end]
        batch_path = f"{path}_batch_{i}.csv"
        
        # Usar mode='w' para sobrescribir y siempre incluir header
        batch.to_csv(batch_path, mode="w", header=True, index=False)