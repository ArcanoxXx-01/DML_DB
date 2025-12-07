from pathlib import Path
import csv
import json
from typing import Optional, Any, List
from config.manager import MODELS_META, METRICS, MODELS, HEADERS
from datetime import datetime, timedelta
from utils.utils import gen_id, now_dt, now_iso
from schemas.models import GetModelResponse
from models.status import Status

def create_models_for_training(
    training_id: str, dataset_id: str, model_names: list[str], training_type: str, task: str
):
    created = []
    exists = MODELS_META.exists()
    # Use newline="" and csv.writer to avoid extra blank lines on Windows/Unix
    with MODELS_META.open("a", newline="", encoding="utf-8") as w:
        writer = csv.writer(w)
        # If file didn't exist, write header first
        if not exists:
            header = HEADERS.get(MODELS_META, "")
            if header:
                writer.writerow(header.split(","))

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
                now_iso(),
            ]
            r.extend(["0" for _ in range(METRICS)])
            writer.writerow(r)
            created.append(model_id)
    return created


def list_models_by_training_id(training_id: str) -> List[str]:
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


def check_model_trained(model_id: str) -> dict:
    """Check if a model has completed training.
    
    Returns a dict with:
    - model_id: the model ID
    - trained: True if training is completed, False otherwise
    - status: current status of the training task
    """
    try:
        with MODELS_META.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("model_id") == model_id and row.get("task") == "training":
                    status = row.get("status", "")
                    return {
                        "model_id": model_id,
                        "trained": status == Status.COMPLETED.value,
                        "status": status
                    }
    except FileNotFoundError:
        pass
    return {"model_id": model_id, "trained": False, "status": "not_found"}


def check_model_dataset_completed(model_id: str, dataset_id: str) -> dict:
    """Check if a model + dataset combination has completed (training or prediction).
    
    Returns a dict with:
    - model_id: the model ID
    - dataset_id: the dataset ID
    - completed: True if the task is completed, False otherwise
    - task: the task type (training or prediction)
    - status: current status
    """
    try:
        with MODELS_META.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("model_id") == model_id and row.get("dataset_id") == dataset_id:
                    status = row.get("status", "")
                    task = row.get("task", "")
                    return {
                        "model_id": model_id,
                        "dataset_id": dataset_id,
                        "completed": status == Status.COMPLETED.value,
                        "task": task,
                        "status": status
                    }
    except FileNotFoundError:
        pass
    return {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "completed": False,
        "task": "not_found",
        "status": "not_found"
    }


def update_health(model_id: str, dataset_id: Optional[str] = None) -> bool:
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
        if rows[i][0] == model_id and ((dataset_id is None and rows[i][5] == "training") or rows[i][2] == dataset_id):
            rows[i][health_index] = now_iso()
            found = True
            break

    if not found:
        return False

    with MODELS_META.open("w", newline="") as f:
        csv.writer(f).writerows(rows)
    return True


def _is_model_training_completed(model_id: str, all_rows: list[dict]) -> bool:
    """Check if a model has a completed training row (training_id is not null/empty)."""
    for row in all_rows:
        if (row.get("model_id") == model_id 
            and row.get("training_id") 
            and row.get("task") == "training"):
            return row.get("status") == Status.COMPLETED.value
    return False


def find_model_to_run():
    limit = now_dt() - timedelta(seconds=20)
    try:
        with MODELS_META.open("r") as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)
        
        # Filter out rows where model_id doesn't start with "model"
        valid_rows = []
        invalid_found = False
        for row in all_rows:
            model_id = row.get("model_id", "")
            if model_id.startswith("model"):
                valid_rows.append(row)
            else:
                invalid_found = True
        
        # If invalid rows were found, rewrite the file with only valid rows
        if invalid_found:
            with MODELS_META.open("r") as f:
                reader = csv.reader(f)
                header = next(reader)
            with MODELS_META.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for row in valid_rows:
                    writer.writerow([row.get(h, "") for h in header])
            all_rows = valid_rows
        
        for row in all_rows:
            status = row.get("status", "")
            if status == Status.COMPLETED.value:
                continue
            
            task_type = row.get("task", "")
            
            # If task is prediction, check if the model has completed training first
            if task_type == "prediction":
                model_id = row.get("model_id", "")
                if not _is_model_training_completed(model_id, all_rows):
                    # Skip this prediction row, training not yet completed
                    continue
            
            flag = False
            try:
                h = datetime.fromisoformat(row["health"])
            except Exception:
                flag = True
            if flag or h < limit:
                dataset_id = row.get("dataset_id", "")
                # mark the model as claimed by updating its health timestamp
                try:
                    update_health(row["model_id"], dataset_id)
                except Exception:
                    # if updating health fails, still return the model info
                    pass
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


def get_training_metrics(model_id: str) -> Optional[dict]:
    """Get training metrics for a specific model.
    
    Returns a dict with all metric columns (accuracy, f1_score, etc.) for the model's training task.
    Returns None if the model is not found.
    """
    metric_columns = ["accuracy", "f1_score", "precision", "recall", "roc_auc", "log_loss", "rmse", "mae", "mse", "r2"]
    try:
        with MODELS_META.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("model_id") == model_id and row.get("task") == "training":
                    metrics = {}
                    for col in metric_columns:
                        if col in row:
                            try:
                                metrics[col] = float(row[col]) if row[col] else 0.0
                            except ValueError:
                                metrics[col] = 0.0
                    return {
                        "model_id": model_id,
                        "training_id": row.get("training_id", ""),
                        "dataset_id": row.get("dataset_id", ""),
                        "status": row.get("status", ""),
                        "metrics": metrics
                    }
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


def _is_incoming_model_better(existing_data: dict, incoming_data: dict) -> dict:
    """
    Compare incoming model with existing model to determine which is better.
    
    Returns a dict with:
    - 'use_incoming_training': True if incoming has better training state
    - 'better_prediction_datasets': list of dataset_ids where incoming has better predictions
    """
    existing_meta = existing_data.get("metadata", {})
    incoming_meta = incoming_data.get("metadata", {})
    
    # Training comparison
    existing_training_completed = existing_meta.get("training_completed", False)
    incoming_training_completed = incoming_meta.get("training_completed", False)
    existing_last_batch = existing_meta.get("last_trained_batch")
    incoming_last_batch = incoming_meta.get("last_trained_batch")
    
    use_incoming_training = False
    if incoming_training_completed and not existing_training_completed:
        use_incoming_training = True
    elif not existing_training_completed and not incoming_training_completed:
        existing_b = existing_last_batch if existing_last_batch is not None else -1
        incoming_b = incoming_last_batch if incoming_last_batch is not None else -1
        if incoming_b > existing_b:
            use_incoming_training = True
    
    # Prediction comparison
    existing_predictions = existing_meta.get("last_predicted_batch_by_dataset", {}) or {}
    incoming_predictions = incoming_meta.get("last_predicted_batch_by_dataset", {}) or {}
    
    better_prediction_datasets = []
    for dataset_id, incoming_batch in incoming_predictions.items():
        if dataset_id not in existing_predictions:
            better_prediction_datasets.append(dataset_id)
        else:
            existing_b = existing_predictions.get(dataset_id, -1) or -1
            incoming_b = incoming_batch if incoming_batch is not None else -1
            if incoming_b > existing_b:
                better_prediction_datasets.append(dataset_id)
    
    return {
        "use_incoming_training": use_incoming_training,
        "better_prediction_datasets": better_prediction_datasets
    }


def _merge_models(existing_data: dict, incoming_data: dict, comparison: dict) -> dict:
    """
    Merge existing and incoming model data based on comparison results.
    Keeps the better parts from each version.
    """
    # If incoming training is better, use incoming as base
    if comparison["use_incoming_training"]:
        merged = incoming_data.copy()
        merged_meta = merged.get("metadata", {})
        existing_meta = existing_data.get("metadata", {})
        
        # Preserve existing predictions that are better
        existing_predictions_by_ds = existing_meta.get("predictions_by_dataset", {}) or {}
        existing_last_predicted = existing_meta.get("last_predicted_batch_by_dataset", {}) or {}
        incoming_last_predicted = merged_meta.get("last_predicted_batch_by_dataset", {}) or {}
        
        if "predictions_by_dataset" not in merged_meta:
            merged_meta["predictions_by_dataset"] = {}
        if "last_predicted_batch_by_dataset" not in merged_meta:
            merged_meta["last_predicted_batch_by_dataset"] = {}
        
        # Keep existing predictions where they are better
        for dataset_id, existing_batch in existing_last_predicted.items():
            incoming_batch = incoming_last_predicted.get(dataset_id, -1) or -1
            existing_b = existing_batch if existing_batch is not None else -1
            if existing_b > incoming_batch:
                merged_meta["last_predicted_batch_by_dataset"][dataset_id] = existing_batch
                if dataset_id in existing_predictions_by_ds:
                    merged_meta["predictions_by_dataset"][dataset_id] = existing_predictions_by_ds[dataset_id]
        
        merged["metadata"] = merged_meta
    else:
        # Keep existing as base
        merged = existing_data.copy()
        merged_meta = merged.get("metadata", {})
        incoming_meta = incoming_data.get("metadata", {})
        
        # Update predictions where incoming is better
        incoming_predictions_by_ds = incoming_meta.get("predictions_by_dataset", {}) or {}
        incoming_last_predicted = incoming_meta.get("last_predicted_batch_by_dataset", {}) or {}
        
        if "predictions_by_dataset" not in merged_meta:
            merged_meta["predictions_by_dataset"] = {}
        if "last_predicted_batch_by_dataset" not in merged_meta:
            merged_meta["last_predicted_batch_by_dataset"] = {}
        
        for dataset_id in comparison["better_prediction_datasets"]:
            if dataset_id in incoming_last_predicted:
                merged_meta["last_predicted_batch_by_dataset"][dataset_id] = incoming_last_predicted[dataset_id]
            if dataset_id in incoming_predictions_by_ds:
                merged_meta["predictions_by_dataset"][dataset_id] = incoming_predictions_by_ds[dataset_id]
        
        merged["metadata"] = merged_meta
    
    return merged


def save_model_file(model_id: str, update: bool, model_data: Any) -> bool:
    """
    Save the given model_data (JSON-serializable) into the models folder as {model_id}.json.
    
    If a model file already exists, compares the incoming model with the existing one
    and keeps/merges the most updated version based on:
    - Training state (training_completed, last_trained_batch)
    - Prediction state per dataset (last_predicted_batch_by_dataset)
    """
    try:
        MODELS.mkdir(parents=True, exist_ok=True)
        model_file = MODELS / f"{model_id}.json"
        
        final_data = model_data
        
        # If file exists, compare and merge with existing data
        if model_file.exists():
            try:
                with model_file.open("r") as f:
                    existing_data = json.load(f)
                
                comparison = _is_incoming_model_better(existing_data, model_data)
                
                # Only merge if there's something better in either version
                if comparison["use_incoming_training"] or comparison["better_prediction_datasets"]:
                    final_data = _merge_models(existing_data, model_data, comparison)
                elif not comparison["use_incoming_training"] and not comparison["better_prediction_datasets"]:
                    # Existing is better or equal in all aspects, keep existing
                    # But still check if incoming has any new predictions we don't have
                    incoming_meta = model_data.get("metadata", {})
                    existing_meta = existing_data.get("metadata", {})
                    incoming_pred = incoming_meta.get("last_predicted_batch_by_dataset", {}) or {}
                    existing_pred = existing_meta.get("last_predicted_batch_by_dataset", {}) or {}
                    
                    new_datasets = [ds for ds in incoming_pred.keys() if ds not in existing_pred]
                    if new_datasets:
                        comparison["better_prediction_datasets"] = new_datasets
                        final_data = _merge_models(existing_data, model_data, comparison)
                    else:
                        final_data = existing_data
            except (json.JSONDecodeError, KeyError):
                # If existing file is corrupted, use incoming data
                final_data = model_data
        
        with model_file.open("w") as f:
            json.dump(final_data, f, indent=2)
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
