from typing import List
from pydantic import BaseModel


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    batches: int


class DatasetListResponse(BaseModel):
    datasets: List[str]
