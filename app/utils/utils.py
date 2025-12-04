import time, uuid
from typing import List, Optional
from datetime import datetime, timedelta
import csv
import config.manager as manager
from config.manager import (
    _DATA_PATH_,
    DATASETS,
    DATASETS_META,
    TRAININGS_META,
    MODELS,
    MODELS_META,
    HEADERS,
    PREDICTIONS
)


def now_ts() -> float:
    """Returns the current UNIX timestamp, adjusted by TIME_OFFSET_SECONDS."""
    return time.time() + manager.TIME_OFFSET_SECONDS


def now_dt() -> datetime:
    """Returns the current UTC datetime, adjusted by TIME_OFFSET_SECONDS."""
    return datetime.utcnow() + timedelta(seconds=manager.TIME_OFFSET_SECONDS)


def now_iso() -> str:
    """Returns the current adjusted UTC datetime as an ISO-8601 string."""
    return now_dt().isoformat()


def gen_id(prefix: Optional[str] = None) -> str:
    uid = uuid.uuid4().hex
    return f"{prefix}-{uid}" if prefix else uid


def row(headers: List[str]):
    return "\n" + ",".join(headers)

def ensure_paths_exists():
    _DATA_PATH_.mkdir(exist_ok=True)
    DATASETS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    PREDICTIONS.mkdir(exist_ok=True)
    files = [DATASETS_META, TRAININGS_META, MODELS_META]
    for f in files:
        header = HEADERS.get(f, "")
        # If file doesn't exist, create and write header using csv to ensure newline correctness
        if not f.exists():
            with f.open("w", newline="", encoding="utf-8") as w:
                if header:
                    writer = csv.writer(w)
                    writer.writerow(header.split(","))
        else:
            # If file exists but header and first row are concatenated without a newline,
            # fix it by inserting a newline after the header string.
            try:
                with f.open("r", encoding="utf-8") as r:
                    content = r.read()
                if header and content.startswith(header) and len(content) > len(header):
                    next_char = content[len(header):len(header)+1]
                    if next_char not in ("\n", "\r"):
                        fixed = content[:len(header)] + "\n" + content[len(header):]
                        with f.open("w", encoding="utf-8", newline="") as w:
                            w.write(fixed)
            except Exception:
                # If any error occurs while repairing, skip to avoid breaking startup
                pass

# def get_dataset_id(training_id: str):
#     with open(TRAININGS_META, 'r')as r:
#         for row in r.readlines():
#             if(row["training_id"]==training_id):
#                 return row["dataset_id"]
#         return None
