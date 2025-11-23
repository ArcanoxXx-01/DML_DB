from pydantic import BaseModel
from typing import List, Dict


class createTrainingRequest(BaseModel):
    training_id: str
    dataset_id: str
    training_type: str
    models: list[str]  # lista de model_name


class getTrainingRespanse(BaseModel):
    training_id: str
    dataset_id: str
    training_type: str
    models_ids: List


class saveTrainingResultsRequest(BaseModel):
    model_id: str
    results: Dict  # { metric_1, metric_2, ...}


class uploadTrainingRespanse(BaseModel):
    pass
