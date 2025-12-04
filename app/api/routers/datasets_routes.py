from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, Body, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import requests
from schemas.datasets import DatasetUploadResponse, DatasetListResponse
from api.services.dataset_services import (
    save_batches,
    get_batch_file,
    update_meta,
    list_meta,
    append_to_csv_file # You likely need a helper to write to 'datasets.csv'
)
import csv
from config.config import middleware

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("/", response_model=DatasetListResponse)
def get_all_datasets():
    """
    Returns the list of all dataset IDs stored locally.
    """
    datasets = list_meta()
    return DatasetListResponse(datasets=datasets)


# Helper to write raw row to datasets.csv (Implement based on your project structure)
def _append_meta_row(row: list):
    with open('storage/datasets.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

@router.post("/", response_model=DatasetUploadResponse)
async def upload_dataset(
    background_tasks: BackgroundTasks,
    dataset_id: str = Body(...), 
    file: UploadFile = File(...)
):
    """
    User-facing upload. Saves locally, then triggers replication and sync in background.
    """
    content = await file.read()
    text = content.decode("utf-8").splitlines()
    rows = list(csv.reader(text))
    
    if not rows:
        raise HTTPException(status_code=400, detail="CSV is empty")
        
    # 1. Save locally
    batches = save_batches(dataset_id, rows)
    update_meta(dataset_id, batches) # Assuming this updates datasets.csv locally
    
    # Update in-memory metadata
    middleware.peer_metadata.update_csv('datasets.csv', middleware._get_own_ip())
    middleware.peer_metadata.update_dataset(dataset_id, middleware._get_own_ip())

    # 2. Coordinate Replication (Background Task)
    # We pass the raw content bytes to the background task
    background_tasks.add_task(middleware.replicate_dataset, dataset_id, content)
    
    return DatasetUploadResponse(dataset_id=dataset_id, batches=batches)


@router.post("/replicate")
async def receive_replication(
    dataset_id: str = Body(...),
    nodes_ips: List[str] = Body(...), 
    file: UploadFile = File(...)
):
    """
    Internal Endpoint: Received a dataset from another node.
    Save it, but DO NOT trigger further replication (to avoid loops).
    """
    print(f"[Receive-Replica] Receiving copy of {dataset_id}")
    content = await file.read()
    text = content.decode("utf-8").splitlines()
    rows = list(csv.reader(text))
    
    # Save locally
    batches = save_batches(dataset_id, rows)
    update_meta(dataset_id, batches)
    
    # Update local knowledge
    middleware.peer_metadata.update_dataset(dataset_id, middleware._get_own_ip())
    for node_ip in nodes_ips:
        middleware.peer_metadata.update_dataset(dataset_id, node_ip)

    return {"status": "replicated", "batches": batches}


@router.get("/{dataset_id}/{batch}")
def get_batch(dataset_id: str, batch: int):
    # Check peer metadata first: if this node is a known holder, serve local file
    holders = middleware.peer_metadata.get_dataset_nodes(dataset_id)
    own_ip = middleware._get_own_ip()

    if own_ip in holders:
        batch_file = get_batch_file(dataset_id, batch)
        if not batch_file.exists():
            # Metadata says we have it but file missing
            raise HTTPException(status_code=404, detail="Batch no encontrado (metadata inconsistency)")
        return FileResponse(batch_file)

    # Otherwise, try to find a peer that has the dataset and proxy the request
    # Prefer healthy peers
    if not holders:
        raise HTTPException(status_code=404, detail="Dataset no encontrado en la red")

    last_error = None
    for holder in holders:
        # skip self just in case
        if holder == own_ip:
            continue
        # Quick health check using middleware helper
        try:
            if not middleware.check_ip_alive(holder):
                continue
        except Exception:
            # If the health check itself fails, skip this holder
            continue

        try:
            url = f"http://{holder}:8000/api/v1/datasets/{dataset_id}/{batch}"
            resp = requests.get(url, stream=True, timeout=10)
            if resp.status_code == 200:
                # Stream the response back to the client preserving content-type
                content_type = resp.headers.get("Content-Type", "application/octet-stream")
                return StreamingResponse(resp.iter_content(chunk_size=8192), media_type=content_type)
            else:
                last_error = f"peer {holder} returned status {resp.status_code}"
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            continue

    # If we reach here, none of the holders responded successfully
    detail = "No se pudo obtener el batch desde los peers"
    if last_error:
        detail += f": {last_error}"
    raise HTTPException(status_code=404, detail=detail)


@router.get("/list")
def list_datasets():
    return list_meta()
