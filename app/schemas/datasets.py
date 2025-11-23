from pydantic import BaseModel


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    batches: int
