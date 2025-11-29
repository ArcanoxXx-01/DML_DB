from fastapi import APIRouter, HTTPException, Body
from api.services.predictions_services import save_prediction_session
from schemas.prediction import savePredictionRequest, SavePredictionResponse


router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post("/", response_model=SavePredictionResponse)
def create_prediction_endpoint(req: savePredictionRequest = Body(...)):
    
    saved = save_prediction_session(req.model_id, req.dataset_id)
    if not saved:
        raise HTTPException(status_code=400, detail=saved.get("reason", "failed to save prediction"))
    return SavePredictionResponse(
        model_id=req.model_id,
        dataset_id=req.dataset_id,
    )
