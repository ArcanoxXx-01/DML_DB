from pathlib import Path
import csv
import json
from typing import Optional, Any
from config.manager import MODELS_META, METRICS, MODELS
from datetime import datetime, timedelta
from utils.utils import gen_id
from schemas.models import GetModelResponse
from models.status import Status

def create_models_for_training(
    training_id: str, dataset_id: str, model_names: list[str], training_type: str, task: str
):
    created = []
    with open(MODELS_META, "a") as w:
        for name in model_names:
            model_id = gen_id("model")
            r = [
                model_id,
                training_id,
                dataset_id,
                name,
                training_type,
                task,
                "pending",
                datetime.utcnow().isoformat(),
            ]
            r.extend(["0" for _ in range(METRICS)])
            w.write("\n" + ",".join(r))
            created.append(model_id)
    return created


def list_models_by_training_id(training_id: str):
    ids = []
    try:
        with MODELS_META.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["training_id"] == training_id:
                    ids.append(row["model_id"])
    except FileNotFoundError:
        return []
    return ids


def update_health(model_id: str):
    rows = []
    found = False
    try:
        with MODELS_META.open("r") as f:
            reader = csv.reader(f)
            rows = list(reader)
    except FileNotFoundError:
        return False

    header = rows[0]
    health_index = header.index("health")

    for i in range(1, len(rows)):
        if rows[i][0] == model_id:
            rows[i][health_index] = datetime.utcnow().isoformat()
            found = True
            break

    if not found:
        return False

    with MODELS_META.open("w", newline="") as f:
        csv.writer(f).writerows(rows)
    return True


def find_model_to_run():
    limit = datetime.utcnow() - timedelta(seconds=20)
    try:
        with MODELS_META.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                status = row.get("status", "")
                if status == Status.COMPLETED.value:
                    continue
                flag = False
                try:
                    h = datetime.fromisoformat(row["health"])
                except Exception:
                    flag = True
                if flag or h < limit:
                    dataset_id = row.get("dataset_id", "")
                    return {
                        "model_id": row["model_id"],
                        "dataset_id": dataset_id,
                        "running_type": row["task"],
                    }
    except FileNotFoundError:
        return None
    return None


def get_model_metrics(model_id: str):
    try:
        with MODELS_META.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["model_id"] == model_id:
                    metrics = [row[k] for k in row.keys() if k.startswith("metric_")]
                    return {"model_data": metrics}
    except FileNotFoundError:
        return None
    return None


def get_model_info(model_id: str):
    """Return basic model info from MODELS_META CSV for a given model_id.

    Returns a dict with keys: model_id, model_name, model_type, training_id, dataset_id
    or None if the file or model is not found.
    """
    try:
        with MODELS_META.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("model_id") == model_id:
                    return {
                        "model_id": row.get("model_id", ""),
                        "model_name": row.get("model_name", ""),
                        # mapping `model_type` to the CSV `training_type`
                        "model_type": row.get("training_type", ""),
                        "training_id": row.get("training_id", ""),
                        "dataset_id": row.get("dataset_id", ""),
                    }
    except FileNotFoundError:
        return None
    return None

def load_model(model_id: str) -> Optional[GetModelResponse]:
    model_file = MODELS / f"{model_id}.json"
    try:
        if model_file.exists():
            with model_file.open("r") as f:
                data = json.load(f)
            return {"model_data": data}
    except Exception:
        return None

def save_model_file(model_id: str, update: bool, model_data: Any) -> bool:
    """Save the given model_data (JSON-serializable) into the models folder as {model_id}.json."""
    try:
        MODELS.mkdir(parents=True, exist_ok=True)
        model_file = MODELS / f"{model_id}.json"
        with model_file.open("w") as f:
            json.dump(model_data, f, indent=2)
        return True
    except Exception:
        return False


def update_model_metrics(model_id: str, results: dict[str, float]) -> bool:
    """Update model metric columns by metric name.

    `results` should be a mapping of metric_name -> value. For each
    metric present in `results`, if a column with that metric name
    exists in `MODELS_META` it will be updated; otherwise it is skipped.
    Returns True on success, False if the model or file was not found.
    """
    try:
        with MODELS_META.open("r") as f:
            rows = list(csv.reader(f))
    except FileNotFoundError:
        return False

    if not rows:
        return False

    header = rows[0]
    # map header name -> column index for quick lookup
    header_index = {h: i for i, h in enumerate(header)}
    found = False

    # find column index for status (fall back to known position if header missing)
    status_index = header_index.get("status")

    for i in range(1, len(rows)):
        if rows[i][0] == model_id and rows[i][header_index['task']]=='training':
            for metric_name, val in results.items():
                if metric_name in header_index:
                    # results values are floats; convert to string for CSV
                    rows[i][header_index[metric_name]] = str(val)

            # mark model as completed after metrics update
            if status_index is not None:
                rows[i][status_index] = Status.COMPLETED.value

            found = True
            break

    if not found:
        return False

    with MODELS_META.open("w", newline="") as f:
        csv.writer(f).writerows(rows)
    return True
