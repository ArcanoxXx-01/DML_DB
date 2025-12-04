from fastapi import APIRouter, HTTPException, Body
from typing import List
from api.services.trainings_services import create_training, get_training_by_id, get_all_trainings
from schemas.trainings import TrainingCreateRequest, TrainingResponse


router = APIRouter(prefix="/trainings", tags=["trainings"])


@router.get("", response_model=List[TrainingResponse])
def get_all_trainings_endpoint():
    """
    Retrieve all training sessions.
    """
    trainings = get_all_trainings()
    if not trainings:
        raise HTTPException(status_code=404, detail="No trainings found")
    return trainings


@router.post("")
def create_training_endpoint(req: TrainingCreateRequest = Body(...)):
    data = req.model_dump()
    return create_training(**data)


@router.get("/{training_id}", response_model=TrainingResponse)
def get_training_endpoint(training_id: str):
    training = get_training_by_id(training_id)
    if not training:
        raise HTTPException(status_code=404, detail="Training no encontrado")
    # Only return the fields required by TrainingResponse
    filtered = {
        "training_id": training["training_id"],
        "dataset_id": training["dataset_id"],
        "training_type": training["training_type"],
        "models_ids": training["models_id"],
    }
    return TrainingResponse(**filtered)
