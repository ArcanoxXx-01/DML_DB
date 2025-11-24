from pathlib import Path

# ===== CONSTATNTS =====

BATCH_SIZE = 64
METRICS = 10

MEMBERSHIPS_TIME_REFRESH = 10


# ===== PATHS =====

_DATA_PATH_ = Path("./data")
DATASETS = _DATA_PATH_ / "datasets"
DATASETS_META = _DATA_PATH_ / "datasets.csv"
MODELS = _DATA_PATH_ / "models"
MODELS_META = _DATA_PATH_ / "models.csv"
TRAININGS_META = _DATA_PATH_ / "trainings.csv"

# ===== HEADERS =====

HEADERS = {
    DATASETS_META: "dataset_id,batches",
    MODELS_META: "model_id,training_id,model_name,task,training_type,status,health,"
    + ",".join([f"metric_{i}" for i in range(METRICS)]),
    TRAININGS_META: "training_id,dataset_id,training_type,task,status,created_at",
}


# ===== URLS =====
IP = "127.0.0.0"
PORT = "8000"
API = "/api/v1"
