from pydantic import BaseModel
from typing import List, Literal, Dict


class ModelHealthUpdateResponse(BaseModel):
    model_id: str
    health_updated: bool


class SaveModelRequest(BaseModel):
    model_id: str
    model_data: Dict


class GetModelResponse(BaseModel):
    model_data: List[str]


class ModelToRunResponse(BaseModel):
    model_id: str
    dataset_id: str
    task: Literal["training", "prediction"]


class ModelMetricsUpdateRequest(BaseModel):
    results: List[str]


class ModelUpdatedResponse(BaseModel):
    model_id: str
    updated: bool
