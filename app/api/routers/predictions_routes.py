from fastapi import APIRouter, HTTPException, Body, UploadFile, File, BackgroundTasks, Form
from fastapi.responses import JSONResponse
import json
import requests
from config.config import middleware
from api.services.predictions_services import save_prediction_session, save_prediction_results, get_prediction_results, get_all_predictions_by_model
from schemas.prediction import savePredictionRequest, SavePredictionResponse, SavePredictionResultsRequest, SavePredictionResultsResponse, GetPredictionResponse, GetAllPredictionsByModelResponse


router = APIRouter(prefix="/predictions", tags=["predictions"])


# ============================================================================
# REPLICATION ENDPOINTS - MUST COME BEFORE PARAMETRIC ROUTES
# ============================================================================

@router.post("/replicate")
async def receive_prediction_replication(
    model_id: str = Form(...),
    dataset_id: str = Form(...),
    nodes_ips: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Internal Endpoint: Receive a prediction JSON from another node and save it locally.
    """
    print(f"[Receive-Prediction-Replica] Receiving prediction copy {model_id}_{dataset_id}")
    content = await file.read()

    # Save the prediction file
    ok = middleware.save_prediction_json_content(model_id, dataset_id, content)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not save replicated prediction")

    # Update local metadata about who has the prediction
    try:
        parsed_nodes = []
        try:
            parsed_nodes = json.loads(nodes_ips)
        except Exception:
            parsed_nodes = []

        middleware.peer_metadata.update_prediction(model_id, dataset_id, middleware._get_own_ip())
        for node_ip in parsed_nodes:
            middleware.peer_metadata.update_prediction(model_id, dataset_id, node_ip)
    except Exception:
        pass

    return {"status": "replicated", "model_id": model_id, "dataset_id": dataset_id}


# ============================================================================
# STANDARD ENDPOINTS
# ============================================================================

@router.post("/", response_model=SavePredictionResponse)
def create_prediction_endpoint(req: savePredictionRequest = Body(...)):
    
    saved = save_prediction_session(req.model_id, req.dataset_id)
    if not saved:
        raise HTTPException(status_code=400, detail="failed to save prediction")
    return SavePredictionResponse(
        model_id=req.model_id,
        dataset_id=req.dataset_id,
    )


@router.post("/results", response_model=SavePredictionResultsResponse)
def save_prediction_results_endpoint(
    background_tasks: BackgroundTasks,
    req: SavePredictionResultsRequest = Body(...),
):
    """
    Save prediction results and trigger replication to peers in background.
    """
    # Save locally first
    saved = save_prediction_results(req.model_id, req.dataset_id, req.predictions)
    if not saved:
        raise HTTPException(status_code=500, detail="Failed to save prediction results")

    # Update local peer metadata saying we have the prediction
    try:
        middleware.peer_metadata.update_prediction(req.model_id, req.dataset_id, middleware._get_own_ip())
    except Exception:
        pass

    # Load the prediction JSON for replication
    try:
        prediction_bytes = middleware.load_prediction_json_content(req.model_id, req.dataset_id)
    except Exception:
        prediction_bytes = None

    # Schedule replication in background to avoid blocking the request
    if background_tasks is not None and prediction_bytes is not None:
        print(f"[Replicate-Prediction] Scheduling replication for prediction: {req.model_id}_{req.dataset_id}")
        background_tasks.add_task(
            middleware.replicate_prediction_json,
            req.model_id,
            req.dataset_id,
            prediction_bytes
        )
    else:
        # Best-effort: try synchronous replicate if background tasks not provided
        try:
            if prediction_bytes is not None:
                middleware.replicate_prediction_json(req.model_id, req.dataset_id, prediction_bytes)
        except Exception:
            pass

    return SavePredictionResultsResponse(
        model_id=req.model_id,
        dataset_id=req.dataset_id,
        saved=True
    )


@router.get("/model/{model_id}", response_model=GetAllPredictionsByModelResponse)
def get_all_predictions_by_model_endpoint(model_id: str):
    """
    Get all dataset IDs where predictions were created for a specific model,
    including those that haven't finished yet (PENDING status).
    """
    predictions_data = get_all_predictions_by_model(model_id)
    
    dataset_ids = [pred.get("dataset_id", "") for pred in predictions_data if pred.get("dataset_id")]
    
    return GetAllPredictionsByModelResponse(
        model_id=model_id,
        dataset_ids=dataset_ids,
        total=len(dataset_ids)
    )


@router.get("/{model_id}/{dataset_id}", response_model=GetPredictionResponse)
def get_prediction_endpoint(model_id: str, dataset_id: str):
    """
    Get prediction results. First checks local storage, then queries peers.
    """
    # Check peer metadata: if this node has the prediction, serve local
    holders = middleware.peer_metadata.get_prediction_nodes(model_id, dataset_id)
    own_ip = middleware._get_own_ip()

    if own_ip in holders:
        response = get_prediction_results(model_id, dataset_id)
        if response:
            return GetPredictionResponse(**response)
        # Metadata inconsistency - fall through to try peers

    # Otherwise, try peers that are known holders
    if not holders:
        raise HTTPException(status_code=404, detail="Predicción no encontrada en la red")

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
            url = f"http://{holder}:8000/api/v1/predictions/{model_id}/{dataset_id}"
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

    detail = "No se pudo obtener la predicción desde los peers"
    if last_error:
        detail += f": {last_error}"
    raise HTTPException(status_code=404, detail=detail)
