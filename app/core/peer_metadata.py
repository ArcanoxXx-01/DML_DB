import threading
from typing import Dict, List, Set

class PeerMetadata:
    """
    Tracks which node has which dataset batches, model JSONs, and ensures all nodes have the CSVs.
    """
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.lock = threading.Lock()
        # dataset_id -> set(node_id)
        self.datasets: Dict[str, Set[str]] = {}
        # model_id -> set(node_id)
        self.model_jsons: Dict[str, Set[str]] = {}
        # (model_id, dataset_id) -> set(node_id)
        self.prediction_jsons: Dict[tuple, Set[str]] = {}
        # All nodes should have these CSVs, but we track for completeness
        self.csvs: Dict[str, Set[str]] = {
            'datasets.csv': set(),
            'models.csv': set(),
            'trainings.csv': set()
        }

    def update_dataset(self, dataset_id: str, node_id: str):
        with self.lock:
            if dataset_id not in self.datasets:
                self.datasets[dataset_id] = set()
            self.datasets[dataset_id].add(node_id)

    def update_model(self, model_id: str, node_id: str):
        with self.lock:
            if model_id not in self.model_jsons:
                self.model_jsons[model_id] = set()
            self.model_jsons[model_id].add(node_id)

    def update_prediction(self, model_id: str, dataset_id: str, node_id: str):
        key = (model_id, dataset_id)
        with self.lock:
            if key not in self.prediction_jsons:
                self.prediction_jsons[key] = set()
            self.prediction_jsons[key].add(node_id)

    def update_csv(self, csv_name: str, node_id: str):
        with self.lock:
            if csv_name in self.csvs:
                self.csvs[csv_name].add(node_id)

    def get_dataset_nodes(self, dataset_id: str) -> Set[str]:
        return self.datasets.get(dataset_id, set())

    def get_model_nodes(self, model_id: str) -> Set[str]:
        return self.model_jsons.get(model_id, set())

    def get_prediction_nodes(self, model_id: str, dataset_id: str) -> Set[str]:
        key = (model_id, dataset_id)
        return self.prediction_jsons.get(key, set())

    def get_csv_nodes(self, csv_name: str) -> Set[str]:
        return self.csvs.get(csv_name, set())
