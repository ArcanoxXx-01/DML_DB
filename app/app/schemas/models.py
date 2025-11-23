from pydantic import BaseModel
from typing import Dict


class modelToRunResponse(BaseModel):
    model_id: str
    dataset_id: str
    running_type: str


class saveModelRequest(BaseModel):
    model_id: str
    model_data: Dict


class getModelRequest(BaseModel):
    model_id: str


class getModelResponse(BaseModel):
    model_data: Dict
