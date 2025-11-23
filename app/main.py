from fastapi import FastAPI
from app.api.routers import datasets_routes, trainings_routes, models_routes
from app.config.manager import API
from app.utils.utils import ensure_paths_exists

ensure_paths_exists()

app = FastAPI()
app.include_router(datasets_routes.router, prefix=API)
app.include_router(trainings_routes.router, prefix=API)
app.include_router(models_routes.router, prefix=API)

