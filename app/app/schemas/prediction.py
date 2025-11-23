from pydantic import BaseModel
from typing import Dict


class savePredictionRequest(BaseModel):
    model_id: str
    dataset_id: str
    prediction_list: Dict  # es un json
