from fastapi import APIRouter, UploadFile, File, HTTPException, Body, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from schemas.datasets import DatasetUploadResponse
from api.services.dataset_services import (
    save_batches,
    get_batch_file,
    update_meta,
    list_meta,
    append_to_csv_file # You likely need a helper to write to 'datasets.csv'
)
import csv
from config.manager import middleware

router = APIRouter(prefix="/datasets", tags=["datasets"])

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

    return {"status": "replicated", "batches": batches}


@router.get("/{dataset_id}/{batch}")
def get_batch(dataset_id: str, batch: int):
    batch_file = get_batch_file(dataset_id, batch)
    if not batch_file.exists():
        raise HTTPException(status_code=404, detail="Batch no encontrado")
    return FileResponse(batch_file)


@router.get("/list")
def list_datasets():
    return list_meta()
