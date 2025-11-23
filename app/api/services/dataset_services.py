from pathlib import Path
import csv
from config.manager import DATASETS, DATASETS_META, BATCH_SIZE


def save_batches(dataset_id: str, rows: list[list[str]]) -> int:
    batches = 0
    for i in range(0, len(rows), BATCH_SIZE):
        batch_rows = rows[i : i + BATCH_SIZE]
        batch_file = DATASETS / f"{dataset_id}_batch_{batches}.csv"
        with batch_file.open("w", newline="") as f:
            csv.writer(f).writerows(batch_rows)
        batches += 1
    return batches


def get_batch_file(dataset_id: str, batch: int) -> Path:
    return DATASETS / f"{dataset_id}_batch_{batch}.csv"


def update_meta(dataset_id: str, batches: int):
    with DATASETS_META.open("r") as f:
        rows = list(csv.reader(f))
    found = False
    for row in rows:
        if row[0] == dataset_id:
            row[1] = str(batches)
            found = True
            break
    if not found:
        rows.append([dataset_id, str(batches)])
    with DATASETS_META.open("w", newline="") as f:
        csv.writer(f).writerows(rows)


def list_meta():
    with DATASETS_META.open("r") as f:
        reader = csv.DictReader(f)
        return [row["dataset_id"] for row in reader]
