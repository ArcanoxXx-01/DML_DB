from pydantic import BaseModel
from typing import Dict, List


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