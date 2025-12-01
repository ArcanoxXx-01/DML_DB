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


class ModelVersionInfo(BaseModel):
    """Model version information for comparison between nodes."""
    model_id: str
    training_completed: bool
    last_trained_batch: int | None
    last_predicted_batch_by_dataset: Dict[str, int]  # dataset_id -> last_predicted_batch


class ModelVersionCompareRequest(BaseModel):
    """Request to compare model versions."""
    model_id: str
    training_completed: bool
    last_trained_batch: int | None
    last_predicted_batch_by_dataset: Dict[str, int]


class ModelVersionCompareResponse(BaseModel):
    """Response indicating if local model is better and in what way."""
    is_better: bool
    better_training: bool  # True if training state is better
    better_predictions: List[str]  # List of dataset_ids where predictions are better
