from fastapi import APIRouter
from schemas.health import HealthResponse
from utils.utils import now_ts

router = APIRouter(prefix="/health", tags=["health"])

@router.get("/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    Returns a simple status indicating the service is running and the
    current (adjusted) timestamp for time synchronization.
    """
    return HealthResponse(status="healthy", timestamp=now_ts())
