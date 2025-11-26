from pathlib import Path
from datetime import datetime
import json
from typing import Optional
from config.manager import PREDICTIONS
from utils.utils import gen_id, now_iso
from config.manager import MODELS_META, METRICS, MODELS


def save_prediction_session(model_id: str, dataset_id: str) -> bool:
    try:
        with open(MODELS_META, "a") as w:
            r = [
                model_id,
                "N/A",  # No training_id for predictions
                dataset_id,
                "prediction_model",
                "N/A",  # No training_type for predictions
                "prediction",
                "pending",
                datetime.utcnow().isoformat(),
            ]
            r.extend(["0" for _ in range(METRICS)])
            w.write(",".join(r))

        return True
    except Exception:
        return False

def save_prediction_results(model_id: str, dataset_id: str, predictions_list: list[float]) -> bool:
    try:
        # Determine predictions directory: prefer configured PREDICTIONS if set
        if PREDICTIONS:
            predictions_dir = Path(PREDICTIONS)
        else:
            # repository root is three parents up from this services file: services -> api -> app -> repo
            repo_root = Path(__file__).resolve().parents[3]
            predictions_dir = repo_root / "predictions"

        predictions_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{model_id}_{dataset_id}.json"
        file_path = predictions_dir / filename

        payload = {
            "model_id": model_id,
            "dataset_id": dataset_id,
            "created_at": now_iso(),
            "predictions": predictions_list,
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return True
    except Exception:
        return False
