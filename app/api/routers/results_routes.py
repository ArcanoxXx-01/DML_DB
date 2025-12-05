from fastapi import APIRouter, HTTPException, Body
from api.services.trainings_services import save_results
from api.services.predictions_services import save_prediction_session, save_prediction_results
from api.services.models_services import get_training_metrics
from schemas.trainings import ResultsCreateRequest, ResultsResponse, TrainingMetricsResponse
from schemas.prediction import savePredictionRequest, SavePredictionResponse, savePredictionResultsRequest


router = APIRouter(prefix="/results", tags=["results"])


@router.get("/training/{model_id}", response_model=TrainingMetricsResponse)
def get_training_metrics_endpoint(model_id: str):
    metrics = get_training_metrics(model_id)
    if metrics is None:
        raise HTTPException(status_code=404, detail=f"Metrics for model '{model_id}' not found")
    return TrainingMetricsResponse(metrics=metrics)


@router.post("/training", response_model=ResultsResponse)
def create_results_endpoint(req: ResultsCreateRequest = Body(...)):
    data = req.model_dump()
    resp = save_results(**data)
    if not resp.get("saved"):
        print("###################################",resp.get("reason", "failed to save results"))
        raise HTTPException(status_code=400, detail=resp.get("reason", "failed to save results"))
    return ResultsResponse(training_id=resp["training_id"], model_id=resp["model_id"], results=resp["results"])


@router.post("/prediction", response_model=SavePredictionResponse)
def create_prediction_results_endpoint(req: savePredictionResultsRequest = Body(...)):
    
    model_id = req.model_id
    dataset_id = req.dataset_id
    predictions = req.predictions_list

    if not model_id or not dataset_id:
        raise HTTPException(status_code=400, detail="model_id and dataset_id are required")

    saved = save_prediction_results(model_id, dataset_id, predictions)
    if not saved:
        raise HTTPException(status_code=500, detail="failed to save prediction results")

    return SavePredictionResponse(saved=True, model_id=model_id, dataset_id=dataset_id)


