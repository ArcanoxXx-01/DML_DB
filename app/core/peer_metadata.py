import threading
import time
from typing import Dict, List, Set, Any
import requests

class PeerMetadata:
    """
    Tracks which node has which dataset batches, model JSONs, and ensures all nodes have the CSVs.
    """
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.lock = threading.Lock()
        # Sync thread control
        self._sync_thread: threading.Thread | None = None
        self._stop_sync = threading.Event()
        self._middleware = None
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

    def get_datasets_by_node(self, node_id: str) -> Set[str]:
        """
        Get all dataset IDs that a specific node has.
        
        Args:
            node_id: The node ID to search for
            
        Returns:
            Set of dataset IDs that the node has
        """
        with self.lock:
            datasets_on_node = set()
            for dataset_id, nodes in self.datasets.items():
                if node_id in nodes:
                    datasets_on_node.add(dataset_id)
            return datasets_on_node

    def get_datasets_by_node(self, node_id: str, own_ip: str = None) -> Set[str]:
        """
        Get all dataset IDs that a specific node has.

        Args:
            node_id: The node ID to search for
            own_ip: Optional IP of the current node. If provided and the node_id
                    is among holders, only returns dataset_id when `own_ip` is
                    the minimum of holders (used for coordination).

        Returns:
            Set of dataset IDs that the node has
        """
        with self.lock:
            datasets_on_node = set()
            for dataset_id, nodes in self.datasets.items():
                if node_id in nodes:
                    if own_ip is None or own_ip == min(nodes):
                        datasets_on_node.add(dataset_id)
            return datasets_on_node

    def remove_peer(self, node_id: str):
        """
        Remove a peer/node from all tracked metadata sets.
        This will remove `node_id` from datasets, model_jsons, prediction_jsons and csvs.
        If any entry's set becomes empty, the entry is removed from the dictionary.
        """
        with self.lock:
            # datasets
            for dataset_id in list(self.datasets.keys()):
                nodes = self.datasets.get(dataset_id)

                if nodes and node_id in nodes:
                    nodes.discard(node_id)
                    if not nodes:
                        del self.datasets[dataset_id]

            # model jsons
            for model_id in list(self.model_jsons.keys()):
                nodes = self.model_jsons.get(model_id)
                if nodes and node_id in nodes:
                    nodes.discard(node_id)
                    if not nodes:
                        del self.model_jsons[model_id]

            # prediction jsons
            for key in list(self.prediction_jsons.keys()):
                nodes = self.prediction_jsons.get(key)
                if nodes and node_id in nodes:
                    nodes.discard(node_id)
                    if not nodes:
                        del self.prediction_jsons[key]

            # csvs
            for csv_name in list(self.csvs.keys()):
                nodes = self.csvs.get(csv_name)
                if nodes and node_id in nodes:
                    nodes.discard(node_id)
                    # Keep the csv key even if empty (CSV list is predefined), but clear set
                    # If you prefer to delete, uncomment the following lines.
                    # if not nodes:
                    #     del self.csvs[csv_name]
            # Finished removal

    # ----- Sync helpers -----
    def to_dict(self) -> Dict[str, Any]:
        """Export the current metadata as JSON-serializable dict."""
        with self.lock:
            return {
                "node_id": self.node_id,
                "datasets": {k: list(v) for k, v in self.datasets.items()},
                "model_jsons": {k: list(v) for k, v in self.model_jsons.items()},
                # prediction keys serialized as "model_id::dataset_id"
                "prediction_jsons": {f"{mi}:: {di}": list(v) for (mi, di), v in self.prediction_jsons.items()},
                "csvs": {k: list(v) for k, v in self.csvs.items()},
            }

    def merge_peer_metadata(self, data: Dict[str, Any]):
        """Merge metadata received from a peer into local metadata.

        The incoming `data` should follow the structure returned by `to_dict()`.
        """
        with self.lock:
            # datasets
            for dataset_id, nodes in data.get("datasets", {}).items():
                if dataset_id not in self.datasets:
                    self.datasets[dataset_id] = set()
                self.datasets[dataset_id].update(nodes)

            # model_jsons
            for model_id, nodes in data.get("model_jsons", {}).items():
                if model_id not in self.model_jsons:
                    self.model_jsons[model_id] = set()
                self.model_jsons[model_id].update(nodes)

            # prediction_jsons (keys like "model_id:: dataset_id")
            for key, nodes in data.get("prediction_jsons", {}).items():
                try:
                    # allow either "model:: dataset" or "model::dataset"
                    parts = key.split("::")
                    model_id = parts[0].strip()
                    dataset_id = "::".join(parts[1:]).strip()
                except Exception:
                    continue
                tuple_key = (model_id, dataset_id)
                if tuple_key not in self.prediction_jsons:
                    self.prediction_jsons[tuple_key] = set()
                self.prediction_jsons[tuple_key].update(nodes)

            # csvs
            for csv_name, nodes in data.get("csvs", {}).items():
                if csv_name not in self.csvs:
                    self.csvs[csv_name] = set()
                self.csvs[csv_name].update(nodes)

    def _sync_loop(self):
        """Background loop that every second exchanges metadata with healthy peers."""
        if not self._middleware:
            return

        while not self._stop_sync.is_set():
            try:
                peers = self._middleware.get_healthy_peers()
                payload = self.to_dict()
                # include our node id explicitly
                payload["node_id"] = self.node_id

                for peer in peers:
                    # skip self
                    if peer == self.node_id:
                        continue
                    try:
                        url = f"http://{peer}:8000/api/v1/peers/metadata"
                        resp = requests.post(url, json=payload, timeout=2.0)
                        if resp.status_code == 200:
                            try:
                                remote = resp.json()
                                # remote may include their metadata, merge it
                                if isinstance(remote, dict):
                                    self.merge_peer_metadata(remote)
                            except Exception:
                                pass
                    except Exception:
                        # ignore per-peer errors
                        pass
            except Exception:
                pass

            # sleep 1 second as requested
            self._stop_sync.wait(1.0)

    def start_sync(self, middleware):
        """Start the background sync thread. Call from the Middleware after startup."""
        self._middleware = middleware
        if self._sync_thread is None or not self._sync_thread.is_alive():
            self._stop_sync.clear()
            self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
            self._sync_thread.start()

    def stop_sync(self):
        """Stop the background sync thread if running."""
        self._stop_sync.set()
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
