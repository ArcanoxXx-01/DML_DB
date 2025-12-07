# Initialize middleware with CSV paths for syncing
from config.manager import CSV_PATHS
from core.middleware import Middleware


ip_cache = {
    "db": ["172.18.0.2","172.18.0.3 ","172.18.0.4", "172.18.0.5", "172.18.0.6"],
    "backend": ["172.18.0.7", "172.18.0.8"]
}

middleware = Middleware(timeout=30.0, csv_paths=CSV_PATHS, ip_cache=ip_cache)