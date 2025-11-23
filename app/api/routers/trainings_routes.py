from fastapi import APIRouter, HTTPException
from api.services.trainings_services import create_training, get_training_by_id
from schemas.trainings import TrainingCreateRequest, TrainingResponse


router = APIRouter(prefix="/trainings", tags=["trainings"])


@router.post("")
def create_training_endpoint(req: TrainingCreateRequest):
    data = req.model_dump()
    return create_training(**data)


@router.get("/{training_id}", response_model=TrainingResponse)
def get_training_endpoint(training_id: str):
    training = get_training_by_id(training_id)
    if not training:
        raise HTTPException(status_code=404, detail="Training no encontrado")
    return TrainingResponse(**training)
