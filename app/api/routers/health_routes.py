from fastapi import APIRouter
from schemas.health import HealthResponse

router = APIRouter(prefix="/health", tags=["health"])

@router.get("/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    Returns a simple status indicating the service is running.
    """
    return HealthResponse(status="healthy")
