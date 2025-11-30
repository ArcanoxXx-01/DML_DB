from pathlib import Path
import csv
from config.manager import DATASETS, DATASETS_META, BATCH_SIZE


def save_batches(dataset_id: str, rows: list[list[str]]) -> int:
    if not rows:
        return 0
    
    headers = rows[0]
    data_rows = rows[1:]
    batches = 0
    
    for i in range(0, len(data_rows), BATCH_SIZE):
        batch_rows = data_rows[i : i + BATCH_SIZE]
        batch_file = DATASETS / f"{dataset_id}_batch_{batches}.csv"
        with batch_file.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(batch_rows)
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

def append_to_csv_file(row: list[str], file_path: Path = DATASETS_META):
    """
    Appends a row to the CSV file if the ID (first column) does not already exist.
    Used primarily for syncing metadata from other peers.
    """
    dataset_id = row[0]
    
    # 1. Read existing to prevent duplicates
    if file_path.exists():
        with file_path.open("r") as f:
            reader = csv.reader(f)
            for existing_row in reader:
                # If dataset_id matches, we already have this record.
                if existing_row and existing_row[0] == dataset_id:
                    return 

    # 2. Append if strictly new
    with file_path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)