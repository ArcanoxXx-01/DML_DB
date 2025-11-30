# Initialize middleware with CSV paths for syncing
from config.manager import CSV_PATHS
from core.middleware import Middleware


middleware = Middleware(timeout=30.0, csv_paths=CSV_PATHS)