from pydantic import BaseModel
from typing import Optional


class HealthResponse(BaseModel):
    """Health check response schema"""
    status: str
    timestamp: Optional[float] = None  # Unix timestamp used for time sync
