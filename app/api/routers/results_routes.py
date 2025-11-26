from fastapi import APIRouter, HTTPException, Body
from api.services.trainings_services import save_results
from schemas.trainings import ResultsCreateRequest, ResultsResponse


router = APIRouter(prefix="/results", tags=["results"])


@router.post("/training", response_model=ResultsResponse)
def create_results_endpoint(req: ResultsCreateRequest = Body(...)):
    data = req.model_dump()
    resp = save_results(**data)
    if not resp.get("saved"):
        raise HTTPException(status_code=400, detail=resp.get("reason", "failed to save results"))
    return ResultsResponse(training_id=resp["training_id"], model_id=resp["model_id"], results=resp["results"])
