from typing import List
from fastapi import APIRouter, HTTPException, Body, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
import json
import requests
from config.manager import middleware
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


@router.get("/health/{model_id}/{dataset_id}", response_model=ModelHealthUpdateResponse)
def update_health_endpoint(model_id: str, dataset_id: str):
    ok = update_health(model_id, dataset_id)
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
    # Check peer metadata: if this node has the model, serve local
    holders = middleware.peer_metadata.get_model_nodes(model_id)
    own_ip = middleware._get_own_ip()

    if own_ip in holders:
        response = load_model(model_id)
        if not response:
            raise HTTPException(status_code=404, detail="Modelo no encontrado (metadata inconsistency)")
        return response

    # Otherwise, try peers that are known holders
    if not holders:
        raise HTTPException(status_code=404, detail="Modelo no encontrado en la red")

    last_error = None
    for holder in holders:
        if holder == own_ip:
            continue
        try:
            if not middleware.check_ip_alive(holder):
                continue
        except Exception:
            continue

        try:
            url = f"http://{holder}:8000/api/v1/models/{model_id}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                try:
                    return JSONResponse(content=resp.json(), status_code=200)
                except Exception:
                    return JSONResponse(content=resp.content.decode("utf-8"), status_code=200)
            else:
                last_error = f"peer {holder} returned status {resp.status_code}"
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            continue

    detail = "No se pudo obtener el modelo desde los peers"
    if last_error:
        detail += f": {last_error}"
    raise HTTPException(status_code=502, detail=detail)


@router.get("/info/{model_id}", response_model=ModelInfoResponse)
def get_model_info_endpoint(model_id: str):
    data = get_model_info(model_id)
    if not data:
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    return ModelInfoResponse(**data)


# @router.post("/{model_id}", response_model=ModelUpdatedResponse)
# def save_model_endpoint(model_id: str, req: SaveModelRequest = Body(...)):
#     ok = save_model_file(model_id, req.update, req.model_data)
#     if not ok:
#         raise HTTPException(status_code=500, detail="No se pudo guardar el modelo")
#     return ModelUpdatedResponse(model_id=model_id, updated=True)


@router.post("/{model_id}")
def save_and_replicate_model_endpoint(
    model_id: str,
    req: SaveModelRequest = Body(...),
    background_tasks: BackgroundTasks = None,
):
    """
    Save model and trigger replication to peers in background.
    This replaces the simple save endpoint to also replicate the model JSON.
    """
    ok = save_model_file(model_id, req.update, req.model_data)
    if not ok:
        raise HTTPException(status_code=500, detail="No se pudo guardar el modelo")

    # Update local peer metadata saying we have the model
    try:
        middleware.peer_metadata.update_model(model_id, middleware._get_own_ip())
    except Exception:
        pass

    # Prepare JSON bytes for replication
    try:
        payload_bytes = json.dumps(req.model_data).encode("utf-8")
    except Exception:
        payload_bytes = None

    # Schedule replication in background to avoid blocking the request
    if background_tasks is not None and payload_bytes is not None:
        background_tasks.add_task(middleware.replicate_model_json, model_id, payload_bytes)
    else:
        # Best-effort: try synchronous replicate if background tasks not provided
        try:
            if payload_bytes is not None:
                middleware.replicate_model_json(model_id, payload_bytes)
        except Exception:
            pass

    return ModelUpdatedResponse(model_id=model_id, updated=True)


@router.post("/replicate")
async def receive_model_replication(
    model_id: str = Body(...),
    nodes_ips: List[str] = Body(...),
    file: UploadFile = File(...),
):
    """
    Internal Endpoint: Receive a model JSON from another node and save it locally.
    """
    print(f"[Receive-Model-Replica] Receiving model copy {model_id}")
    content = await file.read()
    try:
        data = json.loads(content.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON file")

    ok = save_model_file(model_id, True, data)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not save replicated model")

    # Update local metadata about who has the model
    try:
        middleware.peer_metadata.update_model(model_id, middleware._get_own_ip())
        for node_ip in nodes_ips:
            middleware.peer_metadata.update_model(model_id, node_ip)
    except Exception:
        pass

    return {"status": "replicated", "model_id": model_id}

