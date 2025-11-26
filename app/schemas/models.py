from pydantic import BaseModel
from typing import List, Literal, Dict, Any


class ModelHealthUpdateResponse(BaseModel):
    model_id: str
    health_updated: bool


class SaveModelRequest(BaseModel):
    update: bool
    model_data: Dict[str, Any]


class GetModelResponse(BaseModel):
    model_data: Dict[str, Any]


class ModelInfoResponse(BaseModel):
    model_id: str
    model_name: str
    model_type: str
    training_id: str
    dataset_id: str


class ModelToRunResponse(BaseModel):
    model_id: str
    dataset_id: str
    running_type: Literal["training", "prediction"]


class ModelMetricsUpdateRequest(BaseModel):
    results: List[str]


class ModelUpdatedResponse(BaseModel):
    model_id: str
    updated: bool
