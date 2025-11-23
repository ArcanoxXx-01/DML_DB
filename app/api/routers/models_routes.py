from fastapi import APIRouter, HTTPException, Form
from schemas.models import (
    ModelHealthUpdateResponse,
    ModelMetricsResponse,
    ModelMetricsUpdateRequest,
    ModelUpdatedResponse,
    ModelToRunResponse,
)
from api.services.models_services import (
    update_health,
    find_model_to_run,
    get_model_metrics,
    update_model_metrics,
)

router = APIRouter(prefix="/models", tags=["models"])


@router.put("/health/{model_id}", response_model=ModelHealthUpdateResponse)
def update_health_endpoint(model_id: str):
    ok = update_health(model_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    return ModelHealthUpdateResponse(model_id=model_id, health_updated=True)


@router.get("/torun", response_model=ModelToRunResponse)
def get_model_to_run_endpoint():
    model = find_model_to_run()
    if not model:
        model = None
    return ModelToRunResponse(**model)


@router.get("/{model_id}", response_model=ModelMetricsResponse)
def get_model_endpoint(model_id: str):
    data = get_model_metrics(model_id)
    if not data:
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    return ModelMetricsResponse(**data)


@router.put("/{model_id}", response_model=ModelUpdatedResponse)
def update_model_endpoint(model_id: str, req: ModelMetricsUpdateRequest= Form(...)):
    ok = update_model_metrics(model_id, req.results)
    if not ok:
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    return ModelUpdatedResponse(model_id=model_id, updated=True)
