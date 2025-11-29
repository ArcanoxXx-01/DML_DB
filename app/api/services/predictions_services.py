from pathlib import Path
from datetime import datetime
import json
from typing import Optional
from config.manager import PREDICTIONS
from utils.utils import gen_id, now_iso
from config.manager import MODELS_META, METRICS, MODELS
import csv
from models.status import Status


def save_prediction_session(model_id: str, dataset_id: str) -> bool:
    try:
        exists = MODELS_META.exists()
        with MODELS_META.open("a", newline="", encoding="utf-8") as w:
            writer = csv.writer(w)
            # if file didn't exist, write header first (manager.HEADERS is used elsewhere)
            if not exists:
                try:
                    from config.manager import HEADERS
                    header = HEADERS.get(MODELS_META, "")
                    if header:
                        writer.writerow(header.split(","))
                except Exception:
                    pass

            r = [
                model_id,
                "N/A",  # No training_id for predictions
                dataset_id,
                "prediction_model",
                "N/A",  # No training_type for predictions
                "prediction",
                Status.PENDING.value,
                datetime.utcnow().isoformat(),
            ]
            r.extend(["0" for _ in range(METRICS)])
            writer.writerow(r)

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

        # After saving predictions, update MODELS_META: mark status as completed and refresh health
        try:
            rows = []
            with MODELS_META.open("r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
        except FileNotFoundError:
            # If models meta doesn't exist, nothing to update
            return True

        if rows:
            header = rows[0]
            header_index = {h: i for i, h in enumerate(header)}
            status_idx = header_index.get("status")
            health_idx = header_index.get("health")

            updated = False
            for i in range(1, len(rows)):
                if len(rows[i]) > 0 and rows[i][0] == model_id and rows[i][2] == dataset_id:
                    if status_idx is not None:
                        # mark completed
                        rows[i][status_idx] = Status.COMPLETED.value
                    if health_idx is not None:
                        rows[i][health_idx] = now_iso()
                    updated = True
                    break

            if updated:
                with MODELS_META.open("w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerows(rows)

        return True
    except Exception:
        return False
