from pydantic import BaseModel
from typing import List, Literal
from datetime import datetime


class TrainingCreateRequest(BaseModel):
    training_id: str
    dataset_id: str
    training_type: Literal["regression", "classification"]
    models_names: List[str]


class TrainingResponse(BaseModel):
    training_id: str
    dataset_id: str
    training_type: str
    task: str
    status: str
    created_at: datetime
    models_id: List[str]
