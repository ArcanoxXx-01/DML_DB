from pathlib import Path
import csv
from app.config.manager import MODELS_META, METRICS
from datetime import datetime, timedelta
from app.utils.utils import gen_id, row


def create_models_for_training(training_id: str, model_names: list[str]):
    created = []
    with open(MODELS_META, "a") as w:
        for name in model_names:
            model_id = gen_id("model")
            r = [model_id, training_id, name, "pending", datetime.utcnow().isoformat()]
            r.extend(["0" for _ in range(METRICS)])
            w.write('\n'+','.join(r))
            created.append(model_id)
    return created


def list_models_by_training_id(training_id: str):
    ids = []
    with MODELS_META.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["training_id"] == training_id:
                ids.append(row["model_id"])
    return ids


def update_health(model_id: str):
    rows = []
    found = False
    with MODELS_META.open("r") as f:
        reader = csv.reader(f)
        rows = list(reader)

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
    with MODELS_META.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                h = datetime.fromisoformat(row["health"])
            except:
                return row["model_id"]
            if h < limit:
                return row["model_id"]
    return None


def get_model_metrics(model_id: str):
    with MODELS_META.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["model_id"] == model_id:
                metrics = [row[k] for k in row.keys() if k.startswith("metric_")]
                return {"model_data": metrics}
    return None


def update_model_metrics(model_id: str, results: list[str]):
    with MODELS_META.open("r") as f:
        rows = list(csv.reader(f))

    header = rows[0]
    metric_indices = [i for i, h in enumerate(header) if h.startswith("metric_")]
    found = False

    for i in range(1, len(rows)):
        if rows[i][0] == model_id:
            for idx, val in zip(metric_indices, results):
                rows[i][idx] = val
            found = True
            break

    if not found:
        return False

    with MODELS_META.open("w", newline="") as f:
        csv.writer(f).writerows(rows)
    return True
