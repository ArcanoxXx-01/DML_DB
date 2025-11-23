from fastapi import APIRouter, HTTPException, Form
from app.api.services.trainings_services import create_training, get_training_by_id
import pandas as pd

router = APIRouter(prefix="/trainings", tags=["trainings"])


@router.post("")
def create_training_endpoint(
    training_id: str = Form(...),
    dataset_id: str = Form(...),
    task: str = Form(...),
    training_type: str = Form(...),
    models: list[str] = Form(...),
):

    data = {
        "training_id": training_id,
        "dataset_id": dataset_id,
        "task": task,
        "training_type": training_type,
        "models": models,
    }

    df = pd.DataFrame(data)
    df.to_csv(".input.csv")
    return create_training(training_id, dataset_id, task, training_type, models)


@router.get("/{training_id}")
def get_training_endpoint(training_id: str):
    training = get_training_by_id(training_id)
    if not training:
        raise HTTPException(status_code=404, detail="Training no encontrado")
    return training
