from datetime import datetime
from config.manager import TRAININGS_META
from api.services.models_services import (
    create_models_for_training,
    list_models_by_training_id,
)
import csv


def create_training(
    training_id: str, dataset_id: str, task: str, training_type: str, models: list[str]
):
    created_at = datetime.utcnow().isoformat()
    status = "pending"
    with open(TRAININGS_META, "a") as w:
        w.write(
            "\n"
            + ",".join(
                [training_id, dataset_id, training_type, task, status, created_at]
            )
        )
        w.close()
    created_models = create_models_for_training(training_id, models, task)

    return {"training_id": training_id, "models_created": created_models}


def get_training_by_id(training_id: str):
    with TRAININGS_META.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["training_id"] == training_id:
                models = list_models_by_training_id(training_id)
                return {**row, "models_id": models}
    return None
