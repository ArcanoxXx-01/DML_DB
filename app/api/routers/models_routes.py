from typing import List
from fastapi import APIRouter, HTTPException, Body, UploadFile, File, BackgroundTasks, Request, Form
from fastapi.responses import JSONResponse
import json
import requests
from config.config import middleware
from schemas.models import (
    ModelHealthUpdateResponse,
    ModelUpdatedResponse,
    ModelToRunResponse,
    SaveModelRequest,
    GetModelResponse,
    ModelInfoResponse,
    ModelVersionInfo,
    ModelVersionCompareRequest,
    ModelVersionCompareResponse,
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


@router.get("/info/{model_id}", response_model=ModelInfoResponse)
def get_model_info_endpoint(model_id: str):
    data = get_model_info(model_id)
    if not data:
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    return ModelInfoResponse(**data)


# ============================================================================
# REPLICATION ENDPOINTS - MUST COME BEFORE PARAMETRIC ROUTES
# ============================================================================

@router.post("/replicate")
async def receive_model_replication(
    model_id: str = Form(...),
    nodes_ips: str = Form(...),
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
        # nodes_ips arrives as a JSON string in the multipart form, parse it
        parsed_nodes = []
        try:
            parsed_nodes = json.loads(nodes_ips)
        except Exception:
            parsed_nodes = []

        middleware.peer_metadata.update_model(model_id, middleware._get_own_ip())
        for node_ip in parsed_nodes:
            middleware.peer_metadata.update_model(model_id, node_ip)
    except Exception:
        pass

    return {"status": "replicated", "model_id": model_id}


@router.post("/replicate_update")
async def receive_model_update(
    request: Request,
    model_id: str = Form(...),
    file: UploadFile = File(...),
    nodes_ips: str | None = Form(None),
):
    """
    Internal Endpoint: Receive an update for an existing model from another DB node.
    Only accepts requests coming from known healthy peers and will only save the
    update if peer metadata indicates this node already has the model.
    """
    # Verify caller is another healthy peer
    try:
        caller_ip = request.client.host
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot determine caller IP")

    healthy = middleware.get_healthy_peers()
    if caller_ip not in healthy:
        raise HTTPException(status_code=403, detail="Endpoint available only to other DB nodes")

    # Check peer metadata: only accept update if this node already has the model
    own_ip = middleware._get_own_ip()
    holders = middleware.peer_metadata.get_model_nodes(model_id)
    if own_ip not in holders:
        raise HTTPException(status_code=404, detail="Este nodo no tiene el modelo registrado; no se acepta actualizaciones")

    print(f"[Receive-Model-Update] Receiving update for model {model_id} from {caller_ip}")
    content = await file.read()
    try:
        data = json.loads(content.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON file")

    ok = save_model_file(model_id, True, data)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not save updated model")

    # We don't change peer metadata for updates (holders remain same), but ensure we
    # have ourselves recorded. Also optionally update metadata for nodes_ips provided.
    try:
        middleware.peer_metadata.update_model(model_id, own_ip)
        if nodes_ips:
            try:
                parsed_nodes = json.loads(nodes_ips)
                for node_ip in parsed_nodes:
                    middleware.peer_metadata.update_model(model_id, node_ip)
            except Exception:
                pass
    except Exception:
        pass

    return {"status": "update-applied", "model_id": model_id}


@router.get("/version/{model_id}", response_model=ModelVersionInfo)
def get_model_version_info(model_id: str):
    """
    Get version information for a model (for comparison between nodes).
    Returns training_completed, last_trained_batch, and last_predicted_batch_by_dataset.
    """
    response = load_model(model_id)
    if not response or "model_data" not in response:
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    
    model_data = response["model_data"]
    metadata = model_data.get("metadata", {})
    
    # Training fields are inside metadata, not at model_data root
    training_completed = metadata.get("training_completed", False)
    last_trained_batch = metadata.get("last_trained_batch")
    last_predicted_batch_by_dataset = metadata.get("last_predicted_batch_by_dataset", {})
    
    return ModelVersionInfo(
        model_id=model_id,
        training_completed=training_completed,
        last_trained_batch=last_trained_batch,
        last_predicted_batch_by_dataset=last_predicted_batch_by_dataset or {}
    )


@router.post("/compare/{model_id}", response_model=ModelVersionCompareResponse)
def compare_model_version(model_id: str, req: ModelVersionCompareRequest = Body(...)):
    """
    Compare requester's model version with local model.
    Returns whether local model is better and in what aspects.
    
    Better training means:
    - Local training_completed=True and requester's is False
    - OR both not completed but local last_trained_batch > requester's
    
    Better predictions means:
    - Local has a dataset_id that requester doesn't have
    - OR local has higher last_predicted_batch for a dataset_id
    """
    response = load_model(model_id)
    if not response or "model_data" not in response:
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    
    model_data = response["model_data"]
    metadata = model_data.get("metadata", {})
    
    # Training fields are inside metadata, not at model_data root
    local_training_completed = metadata.get("training_completed", False)
    local_last_trained_batch = metadata.get("last_trained_batch")
    local_last_predicted = metadata.get("last_predicted_batch_by_dataset", {}) or {}
    
    req_training_completed = req.training_completed
    req_last_trained_batch = req.last_trained_batch
    req_last_predicted = req.last_predicted_batch_by_dataset or {}
    
    # Check if local training is better
    better_training = False
    if local_training_completed and not req_training_completed:
        # Local completed, requester hasn't -> local is better
        better_training = True
    elif not local_training_completed and not req_training_completed:
        # Both not completed, compare last_trained_batch
        local_batch = local_last_trained_batch if local_last_trained_batch is not None else -1
        req_batch = req_last_trained_batch if req_last_trained_batch is not None else -1
        if local_batch > req_batch:
            better_training = True
    # If requester completed and local hasn't, local is NOT better for training
    
    # Check if local predictions are better for any dataset
    better_predictions = []
    for dataset_id, local_batch in local_last_predicted.items():
        if dataset_id not in req_last_predicted:
            # Local has predictions for a dataset requester doesn't have
            better_predictions.append(dataset_id)
        else:
            req_batch = req_last_predicted.get(dataset_id, -1)
            local_b = local_batch if local_batch is not None else -1
            req_b = req_batch if req_batch is not None else -1
            if local_b > req_b:
                better_predictions.append(dataset_id)
    
    is_better = better_training or len(better_predictions) > 0
    
    return ModelVersionCompareResponse(
        is_better=is_better,
        better_training=better_training,
        better_predictions=better_predictions
    )


# ============================================================================
# PARAMETRIC ROUTES - MUST COME AFTER SPECIFIC ROUTES
# ============================================================================

@router.get("/{model_id}", response_model=GetModelResponse)
def get_model_endpoint(model_id: str):
    """
    Get model by ID. If this node has the model, it first checks if any peer
    has a better version (more trained or with more predictions) and updates
    local model if needed before returning.
    """
    holders = middleware.peer_metadata.get_model_nodes(model_id)
    own_ip = middleware._get_own_ip()

    if own_ip in holders:
        response = load_model(model_id)
        if not response:
            raise HTTPException(status_code=404, detail="Modelo no encontrado (metadata inconsistency)")
        
        # Check if any peer has a better version
        local_model_data = response["model_data"]
        try:
            updated_model = middleware.check_and_update_model_from_peers(model_id, local_model_data, holders)
            if updated_model:
                # Model was updated, save and return the updated version
                save_model_file(model_id, True, updated_model)
                return {"model_data": updated_model}
        except Exception as e:
            print(f"[get_model_endpoint] Error checking peers for better model: {e}")
        
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
    raise HTTPException(status_code=404, detail=detail)


@router.post("/{model_id}")
def save_and_replicate_model_endpoint(
    background_tasks: BackgroundTasks,
    model_id: str,
    req: SaveModelRequest = Body(...),
):
    """
    Save model and trigger replication to peers in background.
    This replaces the simple save endpoint to also replicate the model JSON.
    """
    # If this is an update, only accept it if we already have the model according
    # to peer metadata. Do not create new model files on update if we didn't
    # previously have the model.

    if not req.update:
        # New model: just save it
        ok = save_model_file(model_id, req.update, req.model_data)
        print("[Save-Model] Saved model locally:", model_id)
        if not ok:
            raise HTTPException(status_code=500, detail="No se pudo guardar el modelo")

        # Update local peer metadata saying we have the model
        try:
            middleware.peer_metadata.update_model(model_id, middleware._get_own_ip())
        except Exception:
            pass
    else:
        # Update: check we have the model first
        holders = middleware.peer_metadata.get_model_nodes(model_id)
        own_ip = middleware._get_own_ip()
        if own_ip in holders:
            ok = save_model_file(model_id, req.update, req.model_data)
            print("[Update-Model] Updated model locally:", model_id)
            if not ok:
                raise HTTPException(status_code=500, detail="No se pudo guardar el modelo actualizado")

    # Prepare JSON bytes for replication
    try:
        payload_bytes = json.dumps(req.model_data).encode("utf-8")
    except Exception:
        payload_bytes = None

    # Schedule replication in background to avoid blocking the request
    if background_tasks is not None and payload_bytes is not None:
        print("[Replicate-Model] Scheduling replication for model:", model_id)
        # If this is an update, replicate only to nodes that already have this model
        if req.update:
            background_tasks.add_task(middleware.replicate_model_update, model_id, payload_bytes)
        else:
            background_tasks.add_task(middleware.replicate_model_json, model_id, payload_bytes)
    else:
        # Best-effort: try synchronous replicate if background tasks not provided
        try:
            if payload_bytes is not None:
                if req.update:
                    middleware.replicate_model_update(model_id, payload_bytes)
                else:
                    middleware.replicate_model_json(model_id, payload_bytes)
        except Exception:
            pass

    return ModelUpdatedResponse(model_id=model_id, updated=True)