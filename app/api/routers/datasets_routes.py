from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from schemas.datasets import DatasetUploadResponse
from api.services.dataset_services import (
    save_batches,
    get_batch_file,
    update_meta,
    list_meta,
)
import csv


router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/", response_model=DatasetUploadResponse)
async def upload_dataset(dataset_id: str = Form(...), file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8").splitlines()
    rows = list(csv.reader(text))
    if not rows:
        raise HTTPException(status_code=400, detail="CSV vacío")
    batches = save_batches(dataset_id, rows)
    update_meta(dataset_id, batches)
    return DatasetUploadResponse(dataset_id=dataset_id, batches=batches)


@router.get("/{dataset_id}/{batch}")
def get_batch(dataset_id: str, batch: int):
    batch_file = get_batch_file(dataset_id, batch)
    if not batch_file.exists():
        raise HTTPException(status_code=404, detail="Batch no encontrado")
    return FileResponse(batch_file)


@router.get("/list")
def list_datasets():
    return list_meta()
