from fastapi import FastAPI
from api.routers import datasets_routes, trainings_routes, models_routes, results_routes, predictions_routes, health_routes
from config.manager import API
from utils.utils import ensure_paths_exists
from config.manager import middleware

ensure_paths_exists()


def create_app():
    app = FastAPI()
    app.include_router(datasets_routes.router, prefix=API)
    app.include_router(trainings_routes.router, prefix=API)
    app.include_router(models_routes.router, prefix=API)
    app.include_router(results_routes.router, prefix=API)
    app.include_router(predictions_routes.router, prefix=API)
    app.include_router(health_routes.router, prefix=API)

    @app.on_event("startup")
    async def startup_event():
        """Start middleware monitoring after FastAPI is ready."""
        print("[FastAPI] Application started, beginning middleware monitoring...")
        middleware.start_monitoring()


    # Add shutdown event to cleanup
    @app.on_event("shutdown")
    async def shutdown_event():
        """Stop middleware monitoring on shutdown."""
        print("[FastAPI] Application shutting down, stopping middleware monitoring...")
        middleware.stop_monitoring_thread()


    return app

app = create_app()
