from pydantic import BaseModel
from typing import List


class ModelHealthUpdateResponse(BaseModel):
    model_id: str
    health_updated: bool


class ModelToRunResponse(BaseModel):
    model_id: str


class ModelMetricsResponse(BaseModel):
    model_data: List[str]


class ModelMetricsUpdateRequest(BaseModel):
    results: List[str]


class ModelUpdatedResponse(BaseModel):
    model_id: str
    updated: bool
