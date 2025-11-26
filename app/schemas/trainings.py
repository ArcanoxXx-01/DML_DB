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
    models_ids: List[str]


from typing import Dict


class ResultsCreateRequest(BaseModel):
    training_id: str
    model_id: str
    results: Dict[str, float]


class ResultsResponse(BaseModel):
    training_id: str
    model_id: str
    results: Dict[str, float]
