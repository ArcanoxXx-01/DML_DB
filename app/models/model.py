from pydantic import BaseModel
from typing import List
from datetime import datetime


class Model(BaseModel):
    model_id: str
    metrics: List[float]
    health: datetime
