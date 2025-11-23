import os, csv, pandas as pd
from typing import List, Dict


def append_csv_row(path: str, fieldnames: List[str], rows: List[Dict[str, str]]):
    exists = os.path.exists(path)
    with open(path, "a", newline="\n", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def read_csv_index(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    return pd.DataFrame(pd.read_csv(path))


def save_dataset_in_batches(path: str, data: pd.DataFrame, batch_size=64):
    for i in range((len(data) + batch_size - 1) // batch_size):
        data[batch_size * i : batch_size * (i + 1)].to_csv(
            f"{path}_batch_{i}.csv",
            mode="a",
            header=not os.path.exists(path),
            index=False,
            chunksize=65,
        )