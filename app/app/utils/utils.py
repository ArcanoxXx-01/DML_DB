import time, uuid
from typing import List, Optional
from datetime import datetime
from app.config.manager import (
    _DATA_PATH_,
    DATASETS,
    DATASETS_META,
    TRAININGS_META,
    MODELS,
    MODELS_META,
    HEADERS,
)


def now_ts() -> float:
    return time.time()


def now_iso() -> str:
    return datetime.utcnow().isoformat()


def gen_id(prefix: Optional[str] = None) -> str:
    uid = uuid.uuid4().hex
    return f"{prefix}-{uid}" if prefix else uid


def row(headers: List[str]):
    return "\n" + ",".join(headers)

def ensure_paths_exists():
    _DATA_PATH_.mkdir(exist_ok=True)
    DATASETS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    files = [DATASETS_META, TRAININGS_META, MODELS_META]
    for f in files:
        if not f.exists():
            with open(f, "a") as w:
                w.write(HEADERS[f])
                w.close()


