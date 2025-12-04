# Initialize middleware with CSV paths for syncing
from config.manager import CSV_PATHS
from core.middleware import Middleware


ip_cache = {
    "db": ["196.168.0.2","196.168.0.3 ","196.168.0.4", "196.168.0.5", "196.168.0.6"],
    "backend": ["196.168.0.7", "196.168.0.8"]
}

middleware = Middleware(timeout=30.0, csv_paths=CSV_PATHS, ip_cache=ip_cache)