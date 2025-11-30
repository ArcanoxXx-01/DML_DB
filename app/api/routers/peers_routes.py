from fastapi import APIRouter, Body
from config.manager import middleware

router = APIRouter(prefix="/peers", tags=["peers"])


@router.post("/metadata")
async def receive_peer_metadata(payload: dict = Body(...)):
    """Internal endpoint: receive metadata from a peer and merge it.

    Expects a JSON payload with keys: datasets, model_jsons, prediction_jsons, csvs
    The endpoint merges the incoming metadata and returns this node's metadata
    so the caller can merge our view as well.
    """
    try:
        middleware.peer_metadata.merge_peer_metadata(payload)
        # Reply with our current metadata for the caller to merge
        return middleware.peer_metadata.to_dict()
    except Exception as e:
        return {"error": str(e)}
