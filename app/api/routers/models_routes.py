from fastapi import APIRouter, HTTPException, Body
from schemas.models import (
    ModelHealthUpdateResponse,
    ModelUpdatedResponse,
    ModelToRunResponse,
    SaveModelRequest,
    GetModelResponse,
    ModelInfoResponse,
)
from api.services.models_services import (
    update_health,
    find_model_to_run,
    get_model_metrics,
    get_model_info,
    save_model_file,
    load_model,
)

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/health/{model_id}", response_model=ModelHealthUpdateResponse)
def update_health_endpoint(model_id: str):
    ok = update_health(model_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    return ModelHealthUpdateResponse(model_id=model_id, health_updated=True)


@router.get("/torun", response_model=ModelToRunResponse)
def get_model_to_run_endpoint():
    model = find_model_to_run()
    if not model:
        model = {"model_id": "", "dataset_id": "", "running_type": "training"}
    return ModelToRunResponse(**model)


@router.get("/{model_id}", response_model=GetModelResponse)
def get_model_endpoint(model_id: str):
    response = load_model(model_id)
    if not response:
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    return response


@router.get("/info/{model_id}", response_model=ModelInfoResponse)
def get_model_info_endpoint(model_id: str):
    data = get_model_info(model_id)
    if not data:
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    return ModelInfoResponse(**data)


@router.post("/{model_id}", response_model=ModelUpdatedResponse)
def save_model_endpoint(model_id: str, req: SaveModelRequest = Body(...)):
    ok = save_model_file(model_id, req.update, req.model_data)
    if not ok:
        raise HTTPException(status_code=500, detail="No se pudo guardar el modelo")
    return ModelUpdatedResponse(model_id=model_id, updated=True)

