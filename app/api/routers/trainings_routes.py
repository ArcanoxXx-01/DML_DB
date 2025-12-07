from fastapi import APIRouter, HTTPException, Body
from typing import List, Dict, Any
from api.services.trainings_services import create_training, get_training_by_id, get_all_trainings, get_all_training_ids
from api.services.models_services import list_models_by_training_id, check_model_trained, check_model_dataset_completed, get_training_metrics
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


@router.get("/ids", response_model=List[str])
def get_all_training_ids_endpoint():
    """
    Retrieve all training IDs.
    """
    training_ids = get_all_training_ids()
    return training_ids


@router.get("/models/{training_id}", response_model=List[str])
def get_models_by_training_endpoint(training_id: str):
    """
    Retrieve all model IDs for a specific training.
    """
    model_ids = list_models_by_training_id(training_id)
    if not model_ids:
        raise HTTPException(status_code=404, detail="No se encontraron modelos para este training")
    return model_ids


@router.get("/model/trained/{model_id}")
def check_model_trained_endpoint(model_id: str):
    """
    Check if a model has completed training.
    Returns: {model_id, trained: bool, status}
    """
    result = check_model_trained(model_id)
    return result


@router.get("/metrics/{model_id}")
def get_training_metrics_endpoint(model_id: str):
    """
    Get training metrics for a specific model.
    Returns: {model_id, training_id, dataset_id, status, metrics: {accuracy, f1_score, ...}}
    """
    result = get_training_metrics(model_id)
    if not result:
        raise HTTPException(status_code=404, detail="Modelo no encontrado o no tiene entrenamiento")
    return result


@router.get("/completed/{model_id}/{dataset_id}")
def check_model_dataset_completed_endpoint(model_id: str, dataset_id: str):
    """
    Check if a model + dataset combination has completed (training or prediction).
    Returns: {model_id, dataset_id, completed: bool, task, status}
    """
    result = check_model_dataset_completed(model_id, dataset_id)
    return result


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
