from pydantic import BaseModel
from typing import Dict, List, Optional, Any


class savePredictionRequest(BaseModel):
    model_id: str
    dataset_id: str

class SavePredictionResponse(BaseModel):
    model_id: str
    dataset_id: str

class savePredictionResultsRequest(BaseModel):
    model_id: str
    dataset_id: str
    predictions_list: List[float]


class SavePredictionResultsRequest(BaseModel):
    model_id: str
    dataset_id: str
    predictions: List[float]


class SavePredictionResultsResponse(BaseModel):
    model_id: str
    dataset_id: str
    saved: bool


class GetPredictionResponse(BaseModel):
    model_id: str
    dataset_id: str
    created_at: Optional[str] = None
    predictions: List[Any]