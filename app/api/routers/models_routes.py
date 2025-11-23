from fastapi import APIRouter, HTTPException, Form
from datetime import datetime, timedelta
from app.api.services.models_services import (
    update_health,
    find_model_to_run,
    get_model_metrics,
    update_model_metrics,
)

router = APIRouter(prefix="/models", tags=["models"])


@router.put("/health/{model_id}")
def update_health_endpoint(model_id: str):
    ok = update_health(model_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    return {"model_id": model_id, "health_updated": True}


@router.get("/torun")
def get_model_to_run_endpoint():
    model = find_model_to_run()
    return {"model_id": model} if model else {"model_id": None}


@router.get("/{model_id}")
def get_model_endpoint(model_id: str):
    data = get_model_metrics(model_id)
    if not data:
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    return data


@router.put("/{model_id}")
def update_model_endpoint(model_id: str, results: list[str] = Form(...)):
    ok = update_model_metrics(model_id, results)
    if not ok:
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    return {"model_id": model_id, "updated": True}
