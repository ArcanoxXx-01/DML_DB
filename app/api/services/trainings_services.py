from datetime import datetime
from config.manager import TRAININGS_META
from api.services.models_services import (
    create_models_for_training,
    list_models_by_training_id,
)
from api.services.models_services import update_model_metrics, update_health
from typing import List
import csv


def create_training(
    training_id: str, dataset_id: str, training_type: str, models_names: List[str]
):
    task = "training"
    created_at = datetime.now().isoformat()
    status = "pending"
    # Use Path.open with newline="" and csv.writer to avoid extra blank lines
    try:
        exists = TRAININGS_META.exists()
    except Exception:
        exists = False

    with TRAININGS_META.open("a", newline="", encoding="utf-8") as w:
        writer = csv.writer(w)
        if not exists:
            # write header if the file didn't exist
            writer.writerow(
                ["training_id", "dataset_id", "training_type", "task", "status", "created_at"]
            )
        writer.writerow([training_id, dataset_id, training_type, task, status, created_at])
    created_models = create_models_for_training(
        training_id=training_id,
        dataset_id=dataset_id,
        model_names=models_names,
        training_type=training_type,
        task=task,
    )
    return {"training_id": training_id, "models_created": created_models}


def get_training_by_id(training_id: str):
    with TRAININGS_META.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["training_id"] == training_id:
                models = list_models_by_training_id(training_id)
                return {**row, "models_id": models}
    return None


def save_results(training_id: str, model_id: str, results: dict):
    """Save training results (metrics) for a given model.

    The `results` dict should map metric_name -> value. Each provided
    metric will be written to the corresponding column in `MODELS_META`
    if that column exists; metrics without matching columns are skipped.
    If the model is not found, returns a dict with saved=False.
    """
    try:
        updated = update_model_metrics(model_id, results)
        if not updated:
            return {"saved": False, "reason": "model not found"}
        # update last-seen/health timestamp
        update_health(model_id)
        return {"saved": True, "training_id": training_id, "model_id": model_id, "results": results}
    except Exception as e:
        return {"saved": False, "reason": str(e)}
