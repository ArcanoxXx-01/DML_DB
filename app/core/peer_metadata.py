import threading
import time
from typing import Dict, List, Set, Any, Tuple, Optional
import requests
import csv
import json
from pathlib import Path
from datetime import datetime

class PeerMetadata:
    """
    Tracks which node has which dataset batches, model JSONs, and ensures all nodes have the CSVs.
    """
    def __init__(self, node_id: str, csv_paths: Dict[str, Path] = None, metadata_dir: Path = None):
        self.node_id = node_id
        self.lock = threading.Lock()
        # Sync thread control
        self._sync_thread: threading.Thread | None = None
        self._stop_sync = threading.Event()
        self._middleware = None
        
        # Persistence thread control
        self._persist_thread: threading.Thread | None = None
        self._stop_persist = threading.Event()
        
        # CSV file paths
        self.csv_paths = csv_paths or {}
        
        # Metadata persistence path
        self.metadata_dir = metadata_dir or Path("./data/metadata")
        self.metadata_file = self.metadata_dir / "peer_metadata.json"
        
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
        # Track CSV timestamps for conflict resolution
        self.csv_timestamps: Dict[str, float] = {
            'datasets.csv': 0.0,
            'models.csv': 0.0,
            'trainings.csv': 0.0
        }
        
        # Load persisted metadata on initialization
        self._load_from_json()

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

    def _load_from_json(self):
        """Load persisted metadata from JSON file on initialization."""
        if not self.metadata_file.exists():
            print(f"[_load_from_json] No persisted metadata found at {self.metadata_file}")
            return
        
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            with self.lock:
                # Load datasets
                self.datasets = {
                    dataset_id: set(nodes) 
                    for dataset_id, nodes in data.get("datasets", {}).items()
                }
                
                # Load model_jsons
                self.model_jsons = {
                    model_id: set(nodes) 
                    for model_id, nodes in data.get("model_jsons", {}).items()
                }
                
                # Load prediction_jsons (keys are "model_id::dataset_id")
                self.prediction_jsons = {}
                for key, nodes in data.get("prediction_jsons", {}).items():
                    try:
                        parts = key.split("::")
                        model_id = parts[0].strip()
                        dataset_id = "::".join(parts[1:]).strip()
                        tuple_key = (model_id, dataset_id)
                        self.prediction_jsons[tuple_key] = set(nodes)
                    except Exception as e:
                        print(f"[_load_from_json] Error parsing prediction key {key}: {e}")
                
                # Load csvs
                for csv_name, nodes in data.get("csvs", {}).items():
                    if csv_name in self.csvs:
                        self.csvs[csv_name] = set(nodes)
                
                # Load csv_timestamps
                for csv_name, timestamp in data.get("csv_timestamps", {}).items():
                    if csv_name in self.csv_timestamps:
                        self.csv_timestamps[csv_name] = timestamp
            
            print(f"[_load_from_json] Successfully loaded metadata from {self.metadata_file}")
            print(f"  - Datasets: {len(self.datasets)}")
            print(f"  - Models: {len(self.model_jsons)}")
            print(f"  - Predictions: {len(self.prediction_jsons)}")
            
        except Exception as e:
            print(f"[_load_from_json] Error loading metadata from JSON: {e}")

    def _save_to_json(self):
        """Save current metadata to JSON file for persistence."""
        try:
            with self.lock:
                data = {
                    "node_id": self.node_id,
                    "datasets": {k: list(v) for k, v in self.datasets.items()},
                    "model_jsons": {k: list(v) for k, v in self.model_jsons.items()},
                    "prediction_jsons": {
                        f"{mi}::{di}": list(v) 
                        for (mi, di), v in self.prediction_jsons.items()
                    },
                    "csvs": {k: list(v) for k, v in self.csvs.items()},
                    "csv_timestamps": self.csv_timestamps.copy(),
                    "last_updated": time.time()
                }
            
            # Ensure directory exists
            self.metadata_dir.mkdir(parents=True, exist_ok=True)
            
            # Write to temporary file first, then rename (atomic operation)
            temp_file = self.metadata_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            # Atomic rename
            temp_file.replace(self.metadata_file)
            
        except Exception as e:
            print(f"[_save_to_json] Error saving metadata to JSON: {e}")

    def _persist_loop(self):
        """Background loop that saves metadata to JSON every few seconds."""
        while not self._stop_persist.is_set():
            try:
                self._save_to_json()
            except Exception as e:
                print(f"[_persist_loop] Error in persist loop: {e}")
            
            # Save every 5 seconds
            self._stop_persist.wait(5.0)

    def get_datasets_by_node(self, node_id: str) -> Set[str]:
        """
        Get all dataset IDs that a specific node has.
        """
        with self.lock:
            datasets_on_node = set()
            for dataset_id, nodes in self.datasets.items():
                if node_id in nodes:
                    datasets_on_node.add(dataset_id)
            return datasets_on_node

    def get_predictions_by_node(self, node_id: str) -> Set[Tuple[str, str]]:
        """
        Get all prediction keys (model_id, dataset_id) that a specific node has.
        """
        with self.lock:
            predictions_on_node = set()
            for key, nodes in self.prediction_jsons.items():
                if node_id in nodes:
                    predictions_on_node.add(key)
            return predictions_on_node

    def get_predictions_by_node_for_own_ip(self, node_id: str, own_ip: str = None) -> Set[Tuple[str, str]]:
        """
        Get all prediction keys (model_id, dataset_id) that a specific node has,
        filtered by responsibility (own_ip is the smallest holder).
        """
        with self.lock:
            predictions_on_node = set()
            for key, nodes in self.prediction_jsons.items():
                if node_id in nodes:
                    if own_ip is None or own_ip == min(nodes):
                        predictions_on_node.add(key)
            return predictions_on_node

    def get_datasets_by_node_for_own_ip(self, node_id: str, own_ip: str = None) -> Set[str]:
        """
        Get all dataset IDs that a specific node has.
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

    def read_csv_content(self, csv_name: str) -> List[List[str]]:
        """Read CSV file content and return as list of rows."""
        csv_path = self.csv_paths.get(csv_name)
        if not csv_path or not csv_path.exists():
            return []
        
        try:
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                return list(reader)
        except Exception as e:
            print(f"Error reading CSV {csv_name}: {e}")
            return []

    def write_csv_content(self, csv_name: str, rows: List[List[str]]):
        """Write CSV content to file."""
        csv_path = self.csv_paths.get(csv_name)
        if not csv_path:
            return
        
        try:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            
            # Update timestamp
            with self.lock:
                self.csv_timestamps[csv_name] = time.time()
            print(f"[write_csv_content] Successfully wrote {csv_name} with {len(rows)} rows")
        except Exception as e:
            print(f"Error writing CSV {csv_name}: {e}")

    def get_csv_timestamp(self, csv_name: str) -> float:
        """Get the last modification timestamp for a CSV."""
        csv_path = self.csv_paths.get(csv_name)
        if csv_path and csv_path.exists():
            return csv_path.stat().st_mtime
        return 0.0

    def _merge_csv_rows(self, csv_name: str, local_rows: List[List[str]], remote_rows: List[List[str]]) -> List[List[str]]:
        """
        Merge CSV rows based on specific logic per file type.
        
        Logic:
        - datasets.csv: Merge keeping unique keys (dataset_id).
        - trainings.csv: Merge keeping unique keys (training_id).
        - models.csv: Unique key (model_id, training_id, dataset_id).
          Conflict resolution:
          1. Status 'completed' takes precedence.
          2. Health (most recent timestamp) takes precedence.
        """
        if not local_rows and not remote_rows:
            return []
        
        # If one is empty, return the other
        if not local_rows:
            return remote_rows
        if not remote_rows:
            return local_rows

        # Separate headers and data
        header = local_rows[0]
        # We assume both files have the same header structure
        
        # Filter out headers from data processing
        local_data = local_rows[1:]
        remote_data = remote_rows[1:]
        
        merged_data = []

        if csv_name == 'models.csv':
            merged_data = self._merge_models_logic(local_data, remote_data)
        elif csv_name == 'datasets.csv':
            # Unique key is dataset_id (index 0)
            merged_data = self._merge_simple_unique(local_data, remote_data, key_indices=[0])
        elif csv_name == 'trainings.csv':
            # Unique key is training_id (index 0)
            merged_data = self._merge_simple_unique(local_data, remote_data, key_indices=[0])
        else:
            # Fallback for unknown CSVs: simple dedupe by first 3 columns
            merged_data = self._merge_simple_unique(local_data, remote_data, key_indices=[0, 1, 2])

        # Prepend header and return
        return [header] + merged_data

    def _merge_simple_unique(self, local_data: List[List[str]], remote_data: List[List[str]], key_indices: List[int]) -> List[List[str]]:
        """
        Merges two lists preserving all rows, deduplicating based on specific key indices.
        Local rows are preserved if keys collide (standard set union).
        """
        seen_keys = set()
        merged = []

        def get_key(row):
            # safely extract key, using tuple of values at key_indices
            return tuple(row[i] for i in key_indices if i < len(row))

        # Add local rows
        for row in local_data:
            k = get_key(row)
            if k not in seen_keys:
                merged.append(row)
                seen_keys.add(k)
        
        # Add remote rows if key not seen
        for row in remote_data:
            k = get_key(row)
            if k not in seen_keys:
                merged.append(row)
                seen_keys.add(k)
        
        return merged

    def _merge_models_logic(self, local_data: List[List[str]], remote_data: List[List[str]]) -> List[List[str]]:
        """
        Specific merge logic for models.csv.
        
        Columns based on provided headers:
        0: model_id, 1: training_id, 2: dataset_id, 3: model_name, 4: training_type, 
        5: task, 6: status, 7: health, ...
        
        Key: (model_id, training_id, dataset_id) -> indices (0, 1, 2)
        Precedence:
        1. status == 'completed'
        2. health (float timestamp) -> larger (more recent) wins
        """
        # Map key -> row
        merged_dict = {}

        STATUS_IDX = 6
        HEALTH_IDX = 7

        def get_model_key(row):
            if len(row) < 3:
                return tuple(row)
            return (row[0], row[1], row[2])

        def parse_health(val: str) -> float:
            try:
                return float(val)
            except ValueError:
                return 0.0

        def is_better_row(current_row, new_row):
            """Returns True if new_row should replace current_row."""
            if len(new_row) <= STATUS_IDX: return False
            if len(current_row) <= STATUS_IDX: return True

            curr_status = current_row[STATUS_IDX].strip().lower()
            new_status = new_row[STATUS_IDX].strip().lower()

            # 1. Status Precedence: Completed wins
            if new_status == 'completed' and curr_status != 'completed':
                return True
            if curr_status == 'completed' and new_status != 'completed':
                return False
            
            # 2. Health Precedence: Most recent time wins
            curr_health = parse_health(current_row[HEALTH_IDX]) if len(current_row) > HEALTH_IDX else 0.0
            new_health = parse_health(new_row[HEALTH_IDX]) if len(new_row) > HEALTH_IDX else 0.0

            if new_health > curr_health:
                return True
            
            return False

        # 1. Load Local Data
        for row in local_data:
            k = get_model_key(row)
            merged_dict[k] = row

        # 2. Merge Remote Data with Precedence Logic
        for row in remote_data:
            k = get_model_key(row)
            if k not in merged_dict:
                # New entry, just add it
                merged_dict[k] = row
            else:
                # Conflict: Check precedence
                current_row = merged_dict[k]
                if is_better_row(current_row, row):
                    merged_dict[k] = row

        return list(merged_dict.values())

    # ----- Sync helpers -----
    def to_dict(self) -> Dict[str, Any]:
        """Export the current metadata as JSON-serializable dict, including CSV data."""
        with self.lock:
            # Update timestamps from file system
            for csv_name in self.csvs.keys():
                self.csv_timestamps[csv_name] = self.get_csv_timestamp(csv_name)
            
            # Read CSV contents
            csv_data = {}
            for csv_name in self.csvs.keys():
                csv_data[csv_name] = {
                    'timestamp': self.csv_timestamps[csv_name],
                    'content': self.read_csv_content(csv_name)
                }
            
            return {
                "node_id": self.node_id,
                "datasets": {k: list(v) for k, v in self.datasets.items()},
                "model_jsons": {k: list(v) for k, v in self.model_jsons.items()},
                # prediction keys serialized as "model_id::dataset_id"
                "prediction_jsons": {f"{mi}::{di}": list(v) for (mi, di), v in self.prediction_jsons.items()},
                "csvs": {k: list(v) for k, v in self.csvs.items()},
                "csv_data": csv_data,
            }

    def merge_peer_metadata(self, data: Dict[str, Any]):
        """Merge metadata received from a peer into local metadata.

        The incoming `data` should follow the structure returned by `to_dict()`.
        For CSVs, merge content from both local and remote to maintain all data.
        """
        # First collect merge tasks while holding the lock
        merge_tasks = []
        
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

            # prediction_jsons (keys like "model_id::dataset_id")
            for key, nodes in data.get("prediction_jsons", {}).items():
                try:
                    parts = key.split("::")
                    model_id = parts[0].strip()
                    dataset_id = "::".join(parts[1:]).strip()
                except Exception:
                    continue
                tuple_key = (model_id, dataset_id)
                if tuple_key not in self.prediction_jsons:
                    self.prediction_jsons[tuple_key] = set()
                self.prediction_jsons[tuple_key].update(nodes)

            # csvs - node tracking
            for csv_name, nodes in data.get("csvs", {}).items():
                if csv_name not in self.csvs:
                    self.csvs[csv_name] = set()
                self.csvs[csv_name].update(nodes)

            # Prepare CSV merge tasks
            csv_data = data.get("csv_data", {})
            for csv_name, csv_info in csv_data.items():
                if csv_name not in self.csvs:
                    continue

                remote_timestamp = csv_info.get('timestamp', 0.0)
                remote_content = csv_info.get('content', []) or []

                if remote_content:
                    # Read local content while we have the lock
                    local_content = self.read_csv_content(csv_name)
                    merge_tasks.append((csv_name, local_content, remote_content, remote_timestamp))
        
        # Now perform the actual merging and file I/O outside the lock
        for csv_name, local_content, remote_content, remote_timestamp in merge_tasks:
            try:
                print(f"[merge_peer_metadata] Merging CSV: {csv_name}")
                # Calls the new specialized merge logic
                merged_content = self._merge_csv_rows(csv_name, local_content, remote_content)
                
                # Write merged content
                self.write_csv_content(csv_name, merged_content)
                
                # Update our tracking that this node now has this CSV
                with self.lock:
                    self.csvs[csv_name].add(self.node_id)
                    
            except Exception as e:
                print(f"[merge_peer_metadata] Error merging CSV {csv_name}: {e}")

    def cleanup_dead_peers(self, alive_nodes: List[str]):
        """
        Compare currently tracked nodes against a list of alive nodes.
        Remove any tracked node that is not in the alive list.
        """
        # Convert to set for O(1) lookups
        alive_set = set(alive_nodes)
        # Ensure we never remove ourselves, even if not in the list
        alive_set.add(self.node_id)

        known_nodes = set()

        # 1. Identify all known nodes (inside lock)
        with self.lock:
            for nodes in self.datasets.values():
                known_nodes.update(nodes)
            for nodes in self.model_jsons.values():
                known_nodes.update(nodes)
            for nodes in self.prediction_jsons.values():
                known_nodes.update(nodes)
            for nodes in self.csvs.values():
                known_nodes.update(nodes)

        # 2. Calculate dead nodes (outside lock to prevent deadlock with remove_peer)
        dead_nodes = known_nodes - alive_set

        if not dead_nodes:
            return

        print(f"[cleanup_dead_peers_by_detecting_alives] Pruning {len(dead_nodes)} dead nodes: {dead_nodes}")
        
        # 3. Remove them one by one
        for node in dead_nodes:
            self.remove_peer(node)

    def _sync_loop(self):
        """Background loop that every 10 seconds exchanges metadata with healthy peers."""
        if not self._middleware:
            return

        while not self._stop_sync.is_set():
            try:
                peers = self._middleware.get_healthy_peers()
                self._middleware._refresh_service_ip_cahe()
                payload = self.to_dict()
                # include our node id explicitly
                payload["node_id"] = self.node_id
                
                for peer in peers:
                    # skip self
                    if peer == self.node_id:
                        continue
                    try:
                        url = f"http://{peer}:8000/api/v1/peers/metadata"
                        resp = requests.post(url, json=payload, timeout=5.0)
                        if resp.status_code == 200:
                            try:
                                remote = resp.json()
                                # remote may include their metadata, merge it
                                if isinstance(remote, dict):
                                    self.merge_peer_metadata(remote)
                            except Exception as e:
                                print(f"[_sync_loop] Error parsing response from {peer}: {e}")
                    except Exception as e:
                        # ignore per-peer errors
                        print(f"[_sync_loop] Error syncing with {peer}: {e}")
            except Exception as e:
                print(f"[_sync_loop] Error in sync loop: {e}")

            # sleep 7 seconds
            self._stop_sync.wait(7.0)

    def start_sync(self, middleware):
        """Start the background sync thread. Call from the Middleware after startup."""
        self._middleware = middleware
        if self._sync_thread is None or not self._sync_thread.is_alive():
            self._stop_sync.clear()
            self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
            self._sync_thread.start()
            print("[start_sync] Peer metadata sync thread started")
        
        # Start persistence thread
        if self._persist_thread is None or not self._persist_thread.is_alive():
            self._stop_persist.clear()
            self._persist_thread = threading.Thread(target=self._persist_loop, daemon=True)
            self._persist_thread.start()
            print("[start_sync] Peer metadata persistence thread started")

    def stop_sync(self):
        """Stop the background sync thread if running."""
        self._stop_sync.set()
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
            print("[stop_sync] Peer metadata sync thread stopped")
        
        # Stop persistence thread and save one last time
        self._stop_persist.set()
        if self._persist_thread and self._persist_thread.is_alive():
            self._persist_thread.join(timeout=5.0)
            print("[stop_sync] Peer metadata persistence thread stopped")
        
        # Final save before shutdown
        self._save_to_json()
        print("[stop_sync] Final metadata saved to disk")