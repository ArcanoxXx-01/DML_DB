from pydantic import BaseModel
from typing import List
from datetime import datetime
from app.models.task import TaskType
from app.models.status import Status


class Training(BaseModel):
    training_id: str
    dataset_id: str
    training_type: str
    task: TaskType
    models_ids: List[str]
    status: List[Status]
    created_at: datetime
    modified_at: datetime
