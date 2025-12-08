from typing import Optional, Type, TypeVar, Generic, List
import socket
import requests
import threading
import time
from datetime import datetime
import random
from pathlib import Path

from core.peer_metadata import PeerMetadata
import json
from api.services.models_services import load_model


class Middleware:
    """Middleware class for sending HTTP requests with Pydantic schema validation and IP caching."""

    def __init__(self, timeout: float = 30.0, health_check_timeout: float = 10.0, csv_paths: dict[str, Path] = None, ip_cache: Optional[dict[str, List[str]]] = None):
        """
        Initialize the middleware.
        
        Args:
            timeout: Request timeout in seconds
            health_check_timeout: Timeout for health check requests in seconds
        """
        
        self.service_type = "db"
        self.timeout = timeout
        self.health_check_timeout = health_check_timeout
        self.own_ip = self._get_own_ip()
        self.ip_cache: dict[str, List[str]] = ip_cache or {}
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        self.discovery_thread: Optional[threading.Thread] = None
        self.stop_discovery = threading.Event()
        self.cache_lock = threading.Lock()
        self.peer_metadata: PeerMetadata = PeerMetadata(self.own_ip,csv_paths=csv_paths or {})
        self.REPLICATION_FACTOR = 3

        print("Middleware initialized with own IP:", self.own_ip)

    def _resolve_domain_ips(self, domain: str) -> List[str]:
        """
        Resolve domain to list of IP addresses using DNS lookup.
        
        Args:
            domain: Domain name to resolve
            
        Returns:
            List of IP addresses for the domain
        """
        print(f"[_resolve_domain_ips] Resolving domain: {domain}")
        
        try:
            # Resolve domain to IP addresses
            # getaddrinfo returns all addresses for the hostname
            addr_info = socket.getaddrinfo(domain, None, socket.AF_INET)
            
            # Extract unique IP addresses from the results
            ips = list(set(addr[4][0] for addr in addr_info))
            
            print(f"[_resolve_domain_ips] Resolved {domain} to {len(ips)} IP(s): {ips}")
            return ips
            
        except socket.gaierror as e:
            print(f"[_resolve_domain_ips] Failed to resolve domain {domain}: {e}")
            return []
    
    def _refresh_service_ip_cahe(self):
        """
        Refresh the cached IP addresses for a given domain.
        """
        domain = self.service_type
        print(f"[_refresh_service_ip_cache] Refreshing IP cache for domain: {domain}")
        new_ips = self._resolve_domain_ips(domain)
        
        with self.cache_lock:
            if domain not in self.ip_cache:
                self.ip_cache[domain] = new_ips
                print(f"[_refresh_service_ip_cache] Created cache for {domain}: {new_ips}")
            else:
                # Add only new IPs that aren't already in the cache
                added_count = 0
                for ip in new_ips:
                    if ip not in self.ip_cache[domain]:
                        self.ip_cache[domain].append(ip)
                        added_count += 1
                        print(f"[_refresh_service_ip_cache] Added new IP {ip} to cache for {domain}")
                
                if added_count == 0:
                    print(f"[_refresh_service_ip_cache] No new IPs to add for {domain}. Current cache: {self.ip_cache[domain]}")
                else:
                    print(f"[_refresh_service_ip_cache] Added {added_count} new IP(s). Updated cache for {domain}: {self.ip_cache[domain]}")

        self.check_cache_ips_alive(domain)        
    
    def check_ip_alive(self, ip: str, port: int = 8000, health_path: str = "api/v1/health", remove_if_dead: bool = True) -> bool:
        """
        Check if a specific IP is alive by making a health check request.
        
        Args:
            ip: IP address to check
            port: Port to use for the health check (default: 8000)
            health_path: Health check endpoint path (default: "api/v1/health")
            remove_if_dead: Whether to remove the IP from cache if it's dead (default: True)
            
        Returns:
            True if the IP responds successfully, False otherwise
        """
        # Skip check if it's our own IP
        own_ip = self._get_own_ip()
        if ip == own_ip:
            print(f"[check_ip_alive] Skipping check for own IP: {ip}")
            return True
        
        print(f"[check_ip_alive] Checking IP: {ip}:{port}/{health_path}")
        try:
            url = f"http://{ip}:{port}/{health_path}"
            response = requests.get(url, timeout=self.health_check_timeout)
            response.raise_for_status()
            print(f"[check_ip_alive] IP {ip} is ALIVE")
            return True
        except requests.exceptions.RequestException as e:
            print(f"[check_ip_alive] IP {ip} is DEAD - {type(e).__name__}")
            # Remove from cache if dead and removal is enabled
            if remove_if_dead:
                with self.cache_lock:
                    for domain, ips in self.ip_cache.items():
                        if ip in ips:
                            ips.remove(ip)
                            print(f"[check_ip_alive] Removed {ip} from cache for domain {domain}")
            return False
    
    def check_cache_ips_alive(self, domain: str, port: int = 8000, health_path: str = "api/v1/health", remove_dead: bool = True) -> dict[str, bool]:
        """
        Check which IPs in the cache for a given domain are alive.
        
        Args:
            domain: Domain name whose cached IPs should be checked
            port: Port to use for the health check (default: 8000)
            health_path: Health check endpoint path (default: "api/v1/health")
            remove_dead: Whether to remove dead IPs from cache (default: True)
            
        Returns:
            Dictionary mapping IP addresses to their alive status (True/False)
        """
        print(f"\n{'='*80}")
        print(f"[check_cache_ips_alive] Checking cached IPs for domain: {domain}")
        print(f"{'='*80}")
        
        # Print current datasets state BEFORE checking
        self._print_datasets_state("INITIAL STATE")
        
        results = {}
        
        with self.cache_lock:
            if domain not in self.ip_cache:
                print(f"[check_cache_ips_alive] Domain {domain} not found in cache")
                return results
            
            # Create a copy to avoid modification during iteration
            ips_to_check = self.ip_cache[domain].copy()
        
        print(f"\n[check_cache_ips_alive] Found {len(ips_to_check)} IPs to check: {ips_to_check}")
        for ip in ips_to_check:
            is_alive = self.check_ip_alive(ip, port, health_path, remove_if_dead=remove_dead)
            results[ip] = is_alive
        
        alive_count = sum(1 for status in results.values() if status)
        print(f"\n[check_cache_ips_alive] Health Check Results: {alive_count}/{len(results)} IPs alive")
        print(f"[check_cache_ips_alive] Detailed results: {results}")

        dead_ips = [ip for ip, alive in results.items() if not alive]
        
        if dead_ips:
            print(f"\n{'─'*80}")
            print(f"[check_cache_ips_alive] 🔴 DEAD PEERS DETECTED: {dead_ips}")
            print(f"{'─'*80}")
            self._print_datasets_state("BEFORE CLEANUP")
            
            self.cleanup_dead_peers(dead_ips)
            
            self._print_datasets_state("AFTER CLEANUP & RE-REPLICATION")
            print(f"{'='*80}\n")
        else:
            print(f"\n✅ All peers are healthy!\n{'='*80}\n")

        alive_ips = [ip for ip, alive in results.items() if alive]
        self.peer_metadata.cleanup_dead_peers(alive_nodes=alive_ips)

        return results

    def _print_datasets_state(self, label: str):
        """
        Pretty print the current state of datasets and their replicas.
        
        Args:
            label: Label to identify this state snapshot
        """
        print(f"\n📊 PEER METADATA STATE - {label}")
        print(f"{'─'*80}")

        with self.peer_metadata.lock:
            # Node id
            try:
                print(f"Node ID: {self.peer_metadata.node_id}")
            except Exception:
                print("Node ID: <unavailable>")

            # Datasets
            print('\nDatasets:')
            if not self.peer_metadata.datasets:
                print("   No datasets tracked yet")
            else:
                for dataset_id, nodes in sorted(self.peer_metadata.datasets.items()):
                    node_list = sorted(list(nodes))
                    replica_count = len(node_list)
                    status = "✅" if replica_count >= self.REPLICATION_FACTOR else "⚠️"
                    print(f"   {status} Dataset: {dataset_id}")
                    print(f"      Replicas: {replica_count}/{self.REPLICATION_FACTOR}")
                    print(f"      Nodes: {node_list}")

            # Models
            print('\nModels (model_jsons):')
            if not self.peer_metadata.model_jsons:
                print("   No models tracked yet")
            else:
                for model_id, nodes in sorted(self.peer_metadata.model_jsons.items()):
                    holders = sorted(list(nodes))
                    print(f"   Model: {model_id} -> Nodes: {holders}")

            # Predictions
            print('\nPredictions (model_id::dataset_id -> nodes):')
            if not self.peer_metadata.prediction_jsons:
                print("   No prediction JSONs tracked yet")
            else:
                for (model_id, dataset_id), nodes in sorted(self.peer_metadata.prediction_jsons.items()):
                    holders = sorted(list(nodes))
                    print(f"   {model_id} :: {dataset_id} -> Nodes: {holders}")

            # CSVs and timestamps
            print('\nCSV tracking:')
            for csv_name, nodes in self.peer_metadata.csvs.items():
                holders = sorted(list(nodes))
                timestamp = self.peer_metadata.csv_timestamps.get(csv_name, 0.0)
                # try to get basic CSV info (rows, header) without raising
                rows_info = ''
                try:
                    content = self.peer_metadata.read_csv_content(csv_name)
                    if content:
                        rows_info = f"rows={len(content)} header={content[0]}"
                    else:
                        rows_info = "rows=0"
                except Exception:
                    rows_info = "rows=<error>"

                print(f"   {csv_name} -> Nodes: {holders} | ts={timestamp} | {rows_info}")

        print(f"{'─'*80}")

    def cleanup_dead_peers(self, dead_ips: List[str]):
        """
        Remove dead peers from peer metadata and from cache.
        After cleanup, check all datasets, models, and predictions and re-replicate those that are under-replicated.
        
        Args:
            dead_ips: List of IP addresses that are dead/unreachable
        """
        if not dead_ips:
            return
            
        print(f"[cleanup_dead_peers] Cleaning up {len(dead_ips)} dead peer(s): {dead_ips}")
        
        # Track which datasets, models, and predictions were affected
        affected_datasets = set()
        affected_models = set()
        affected_predictions = set()
        
        # Remove dead peers from peer metadata
        for ip in dead_ips:
            try:
                print(f"[cleanup_dead_peers] Removing peer {ip} from PeerMetadata")
                # Get datasets this peer had before removing
                datasets_on_peer = self.peer_metadata.get_datasets_by_node_for_own_ip(ip, own_ip=self.own_ip)
                affected_datasets.update(datasets_on_peer)

                # models held by this peer (collect before removal)
                models_on_peer = set()
                with self.peer_metadata.lock:
                    for model_id, holders in self.peer_metadata.model_jsons.items():
                        if ip in holders and self.own_ip == min(holders):
                            models_on_peer.add(model_id)
                affected_models.update(models_on_peer)

                # predictions held by this peer (collect before removal)
                predictions_on_peer = set()
                with self.peer_metadata.lock:
                    for key, holders in self.peer_metadata.prediction_jsons.items():
                        if ip in holders and self.own_ip in holders and self.own_ip == min(holders):
                            predictions_on_peer.add(key)
                affected_predictions.update(predictions_on_peer)

                self.peer_metadata.remove_peer(ip)
            except Exception as e:
                print(f"[cleanup_dead_peers] Error removing peer {ip} from metadata: {e}")
        
        # Check and re-replicate affected datasets, models, and predictions
        if affected_datasets:
            print(f"[cleanup_dead_peers] {len(affected_datasets)} dataset(s) potentially affected: {affected_datasets}")
            self.check_and_rereplicate_datasets(affected_datasets)
        if affected_models:
            print(f"[cleanup_dead_peers] {len(affected_models)} model(s) potentially affected: {affected_models}")
            self.check_and_rereplicate_models(affected_models)
        if affected_predictions:
            print(f"[cleanup_dead_peers] {len(affected_predictions)} prediction(s) potentially affected: {affected_predictions}")
            self.check_and_rereplicate_predictions(affected_predictions)
        
        if not affected_datasets and not affected_models and not affected_predictions:
            print(f"[cleanup_dead_peers] No resources affected by peer removal")

    def check_and_rereplicate_datasets(self, dataset_ids: set = None):
        """
        Check replication factor for datasets and re-replicate if needed.
        
        Args:
            dataset_ids: Optional set of specific dataset IDs to check. 
                        If None, checks all known datasets.
        """
        if dataset_ids is None:
            # Check all datasets in metadata
            dataset_ids = set(self.peer_metadata.datasets.keys())
        
        if not dataset_ids:
            print("[check_and_rereplicate_datasets] No datasets to check")
            return
        
        print(f"[check_and_rereplicate_datasets] Checking {len(dataset_ids)} dataset(s)")
        
        for dataset_id in dataset_ids:
            current_holders = self.peer_metadata.get_dataset_nodes(dataset_id)
            current_count = len(current_holders)
            needed = self.REPLICATION_FACTOR - current_count
            
            if needed > 0:
                print(f"[check_and_rereplicate_datasets] Dataset {dataset_id} is under-replicated: "
                      f"{current_count}/{self.REPLICATION_FACTOR} copies. Need {needed} more.")
                
                # Check if this node has the dataset
                if self.own_ip in current_holders:
                    print(f"[check_and_rereplicate_datasets] This node has {dataset_id}, initiating re-replication")
                    # Load the dataset and replicate
                    try:
                        file_content = self.load_dataset_content(dataset_id)
                        if file_content:
                            self.replicate_dataset(dataset_id, file_content)
                        else:
                            print(f"[check_and_rereplicate_datasets] Could not load dataset {dataset_id} for re-replication")
                    except Exception as e:
                        print(f"[check_and_rereplicate_datasets] Error re-replicating {dataset_id}: {e}")
                else:
                    print(f"[check_and_rereplicate_datasets] This node doesn't have {dataset_id}, "
                          f"held by: {current_holders}")
            else:
                print(f"[check_and_rereplicate_datasets] Dataset {dataset_id} has sufficient replicas: "
                      f"{current_count}/{self.REPLICATION_FACTOR}")
                
                 
    def load_dataset_content(self, dataset_id: str) -> Optional[bytes]:
        """
        Load dataset content from local storage.
        Reads all batch CSV files, keeping only the header from batch_0.
        
        Args:
            dataset_id: ID of the dataset to load
            
        Returns:
            Combined dataset file content as bytes, or None if not found
        """
        try:
            import os
            import glob
            from config.manager import DATASETS
            
            dataset_dir = DATASETS
            
            # Find all batch files for this dataset
            pattern = os.path.join(dataset_dir, f"{dataset_id}_batch_*.csv")
            batch_files = sorted(glob.glob(pattern), key=lambda x: int(x.split('_batch_')[1].split('.')[0]))
            
            if not batch_files:
                print(f"[load_dataset_content] No batch files found for dataset {dataset_id}")
                return None
            
            print(f"[load_dataset_content] Found {len(batch_files)} batch files for dataset {dataset_id}")
            
            # Combine all batches
            combined_lines = []
            
            for i, batch_file in enumerate(batch_files):
                with open(batch_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                    if i == 0:
                        # First batch: include header + data
                        combined_lines.extend(lines)
                    else:
                        # Other batches: skip header (first line), only add data
                        if len(lines) > 1:
                            combined_lines.extend(lines[1:])
            
            # Convert back to bytes
            result = ''.join(combined_lines).encode('utf-8')
            print(f"[load_dataset_content] Combined {len(batch_files)} batches into {len(result)} bytes")
            
            return result
            
        except Exception as e:
            print(f"[load_dataset_content] Error loading dataset {dataset_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    
    def _get_own_ip(self) -> str:
        """
        Get the IP address of the current instance.
        
        Returns:
            Current instance's IP address
        """
        try:
            # Get hostname and resolve to IP
            hostname = socket.gethostname()
            own_ip = socket.gethostbyname(hostname)
            return own_ip
        except socket.error:
            return "127.0.0.1"
    
    def _discover_ips(self):
        """
        Background thread function to periodically refresh service IP cache.
        Runs every 10 seconds.
        """
        print(f"[_discover_ips] IP discovery thread started")
        
        # Initial delay to allow service to fully start
        print(f"[_discover_ips] Waiting 5 seconds for service startup...")
        self.stop_discovery.wait(5.0)
        
        while not self.stop_discovery.is_set():
            try:
                self._refresh_service_ip_cahe()
            except Exception as e:
                print(f"[_discover_ips] Error during IP refresh: {e}")

            # Wait 3 seconds or until stop signal
            self.stop_discovery.wait(3.0)
        
        print(f"[_discover_ips] IP discovery thread stopped")

    def _replication_monitor(self, interval: float = 20.0):
        """
        Background thread that periodically checks peer metadata for datasets,
        models, and predictions that are under the replication factor. If this node is the
        smallest ("menor") among the current holders and this node actually
        holds the resource, it will initiate re-replication.
        """
        print(f"[_replication_monitor] Replication monitor thread started")

        # Allow service to settle
        print(f"[_replication_monitor] Waiting 5 seconds for startup...")
        self.stop_monitoring.wait(5.0)

        while not self.stop_monitoring.is_set():
            try:
                # Snapshot keys under lock to avoid long lock holding
                with self.peer_metadata.lock:
                    dataset_ids = list(self.peer_metadata.datasets.keys())
                    model_ids = list(self.peer_metadata.model_jsons.keys())
                    prediction_keys = list(self.peer_metadata.prediction_jsons.keys())

                # Check datasets
                for dataset_id in dataset_ids:
                    try:
                        holders = self.peer_metadata.get_dataset_nodes(dataset_id)
                        if not holders:
                            continue
                        if len(holders) < self.REPLICATION_FACTOR:
                            # Responsibility: only the smallest holder in the set triggers re-replication
                            if self.own_ip in holders and self.own_ip == min(holders):
                                print(f"[_replication_monitor] This node is responsible for dataset {dataset_id}; initiating re-replication")
                                self.check_and_rereplicate_datasets({dataset_id})
                            else:
                                print(f"[_replication_monitor] Dataset {dataset_id} under-replicated but this node is not the responsible holder")
                    except Exception as e:
                        print(f"[_replication_monitor] Error checking dataset {dataset_id}: {e}")

                # Check models
                for model_id in model_ids:
                    try:
                        holders = self.peer_metadata.get_model_nodes(model_id)
                        if not holders:
                            continue
                        if len(holders) < self.REPLICATION_FACTOR:
                            if self.own_ip in holders and self.own_ip == min(holders):
                                print(f"[_replication_monitor] This node is responsible for model {model_id}; initiating re-replication")
                                self.check_and_rereplicate_models({model_id})
                            else:
                                print(f"[_replication_monitor] Model {model_id} under-replicated but this node is not the responsible holder")
                    except Exception as e:
                        print(f"[_replication_monitor] Error checking model {model_id}: {e}")

                # Check predictions
                for key in prediction_keys:
                    try:
                        model_id, dataset_id = key
                        holders = self.peer_metadata.get_prediction_nodes(model_id, dataset_id)
                        if not holders:
                            continue
                        if len(holders) < self.REPLICATION_FACTOR:
                            if self.own_ip in holders and self.own_ip == min(holders):
                                print(f"[_replication_monitor] This node is responsible for prediction {model_id}_{dataset_id}; initiating re-replication")
                                self.check_and_rereplicate_predictions({key})
                            else:
                                print(f"[_replication_monitor] Prediction {model_id}_{dataset_id} under-replicated but this node is not the responsible holder")
                    except Exception as e:
                        print(f"[_replication_monitor] Error checking prediction {key}: {e}")

            except Exception as e:
                print(f"[_replication_monitor] Unexpected error during monitor loop: {e}")

            # Wait interval or until stopped
            self.stop_monitoring.wait(interval)

        print(f"[_replication_monitor] Replication monitor thread stopped")
    
    def start_monitoring(self):
        """
        Start the IP discovery background threads.
        """
        # Start IP discovery thread
        if self.discovery_thread is None or not self.discovery_thread.is_alive():
            print(f"[start_monitoring] Starting IP discovery thread")
            self.stop_discovery.clear()
            self.discovery_thread = threading.Thread(target=self._discover_ips, daemon=True)
            self.discovery_thread.start()
        else:
            print(f"[start_monitoring] IP discovery thread already running")

        # Start peer metadata syncing thread
        try:
            print(f"[start_monitoring] Starting peer metadata sync thread")
            self.peer_metadata.start_sync(self)
        except Exception as e:
            print(f"[start_monitoring] Error starting peer metadata sync: {e}")
        
        # Start replication monitor thread
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            try:
                print(f"[start_monitoring] Starting replication monitor thread")
                self.stop_monitoring.clear()
                self.monitoring_thread = threading.Thread(target=self._replication_monitor, daemon=True)
                self.monitoring_thread.start()
            except Exception as e:
                print(f"[start_monitoring] Error starting replication monitor thread: {e}")
        else:
            print(f"[start_monitoring] Replication monitor thread already running")
    
    def stop_monitoring_thread(self):
        """
        Stop IP discovery background thread.
        """
        print(f"[stop_monitoring_thread] Stopping discovery thread")
        
        # Stop IP discovery thread
        self.stop_discovery.set()
        if self.discovery_thread and self.discovery_thread.is_alive():
            self.discovery_thread.join(timeout=15.0)
            print(f"[stop_monitoring_thread] IP discovery thread stopped")

        # Stop peer metadata sync thread
        try:
            print(f"[stop_monitoring_thread] Stopping peer metadata sync thread")
            self.peer_metadata.stop_sync()
        except Exception as e:
            print(f"[stop_monitoring_thread] Error stopping peer metadata sync: {e}")

        # Stop replication monitor thread
        try:
            print(f"[stop_monitoring_thread] Stopping replication monitor thread")
            self.stop_monitoring.set()
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=15.0)
                print(f"[stop_monitoring_thread] Replication monitor thread stopped")
        except Exception as e:
            print(f"[stop_monitoring_thread] Error stopping replication monitor thread: {e}")

    def get_healthy_peers(self) -> List[str]:
        """Returns a list of alive IPs excluding self."""
        domain = self.service_type
        # Ensure cache is populated
        if domain not in self.ip_cache:
            self._refresh_service_ip_cahe()
            
        alive_ips = []
        with self.cache_lock:
            candidates = self.ip_cache.get(domain, [])
        
        for ip in candidates:
            # Skip self and check health
            if ip != self.own_ip and self.check_ip_alive(ip):
                alive_ips.append(ip)
        return alive_ips

    def replicate_dataset(self, dataset_id: str, file_content: bytes):
        """
        Ensures the dataset exists on at least (REPLICATION_FACTOR) nodes.
        Coordinate: Decide who gets the data based on who doesn't have it yet.
        """
        # 1. Update own state first
        self.peer_metadata.update_dataset(dataset_id, self.own_ip)
        
        # 2. Get current holders (including self)
        current_holders = self.peer_metadata.get_dataset_nodes(dataset_id)
        current_count = len(current_holders)
        
        needed = self.REPLICATION_FACTOR - current_count
        
        if needed <= 0:
            print(f"[Replication] Dataset {dataset_id} satisfies RF={self.REPLICATION_FACTOR}. No action needed.")
            return

        print(f"[Replication] Dataset {dataset_id} needs {needed} more copies.")

        # 3. Find candidates (Healthy IPs that are NOT in current_holders)
        all_peers = self.get_healthy_peers()
        candidates = [ip for ip in all_peers if ip not in current_holders]

        if not candidates:
            print("[Replication] No available candidates found to replicate to.")
            return

        # 4. Coordinate: Randomly select targets to balance load
        # If we need 2 but have 5 candidates, pick 2 random ones.
        targets = random.sample(candidates, min(needed, len(candidates)))
        
        print(f"[Replication] Selected targets: {targets}")

        # 5. Send data
        files = {'file': (f"{dataset_id}.csv", file_content, "text/csv")}
        
        for ip in targets:
            try:
                print(f"[Replication] Sending {dataset_id} to {ip}...")
                # We call a specific replication endpoint on the peer
                # Reset file pointer for each request if needed, or send bytes directly
                response = requests.post(
                    f"http://{ip}:8000/api/v1/datasets/replicate",
                    data={"dataset_id": dataset_id, "nodes_ips": targets},
                    files={'file': file_content}, # Requests handles bytes automatically
                    timeout=self.timeout
                )
                if response.status_code == 200:
                    print(f"[Replication] Successfully replicated to {ip}")
                    # Update local metadata knowledge immediately
                    self.peer_metadata.update_dataset(dataset_id, ip)
                else:
                    print(f"[Replication] Failed to send to {ip}: {response.text}")
            except Exception as e:
                print(f"[Replication] Error sending to {ip}: {e}")

    def load_model_json_content(self, model_id: str) -> Optional[bytes]:
        """
        Load model JSON content from local storage.
        
        Args:
            model_id: ID of the model to load
            
        Returns:
            Model JSON file content as bytes, or None if not found
        """
        try:
            from config.manager import MODELS  # Add this to your config
            
            model_path = Path(MODELS) / f"{model_id}.json"
            
            if not model_path.exists():
                print(f"[load_model_json_content] Model file not found: {model_path}")
                return None
            
            with open(model_path, 'rb') as f:
                content = f.read()
            
            print(f"[load_model_json_content] Loaded model {model_id}: {len(content)} bytes")
            return content
            
        except Exception as e:
            print(f"[load_model_json_content] Error loading model {model_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_model_json_content(self, model_id: str, model_json: bytes) -> bool:
        """
        Save model JSON content to local storage.
        
        Args:
            model_id: ID of the model
            model_json: Model JSON content as bytes
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from config.manager import MODELS_DIR
            
            models_dir = Path(MODELS_DIR)
            models_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = models_dir / f"{model_id}.json"
            
            with open(model_path, 'wb') as f:
                f.write(model_json)
            
            print(f"[save_model_json_content] Saved model {model_id}: {len(model_json)} bytes")
            
            # Update metadata to track that this node has the model
            self.peer_metadata.update_model(model_id, self.own_ip)
            
            return True
            
        except Exception as e:
            print(f"[save_model_json_content] Error saving model {model_id}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def check_and_rereplicate_models(self, model_ids: set = None):
        """
        Check replication factor for models and re-replicate if needed.

        Args:
            model_ids: Optional set of specific model IDs to check. If None, checks all known models.
        """
        if model_ids is None:
            model_ids = set(self.peer_metadata.model_jsons.keys())

        if not model_ids:
            print("[check_and_rereplicate_models] No models to check")
            return

        print(f"[check_and_rereplicate_models] Checking {len(model_ids)} model(s)")

        for model_id in model_ids:
            current_holders = self.peer_metadata.get_model_nodes(model_id)
            current_count = len(current_holders)
            needed = self.REPLICATION_FACTOR - current_count

            if needed > 0:
                print(f"[check_and_rereplicate_models] Model {model_id} is under-replicated: "
                    f"{current_count}/{self.REPLICATION_FACTOR} copies. Need {needed} more.")

                # If this node has the model, initiate replication
                if self.own_ip in current_holders:
                    print(f"[check_and_rereplicate_models] This node has {model_id}, initiating re-replication")
                    try:
                        # Load model JSON from disk
                        model_bytes = self.load_model_json_content(model_id)
                        
                        if model_bytes:
                            self.replicate_model_json(model_id, model_bytes)
                        else:
                            print(f"[check_and_rereplicate_models] Could not load model {model_id} for re-replication")
                    except Exception as e:
                        print(f"[check_and_rereplicate_models] Error re-replicating model {model_id}: {e}")
                else:
                    print(f"[check_and_rereplicate_models] This node doesn't have {model_id}, held by: {current_holders}")
            else:
                print(f"[check_and_rereplicate_models] Model {model_id} has sufficient replicas: {current_count}/{self.REPLICATION_FACTOR}")

    def replicate_model_json(self, model_id: str, model_json: bytes):
        """
        Ensure the model JSON exists on at least `REPLICATION_FACTOR` nodes.
        """
        # 1. Update own state first (if we have it)
        try:
            self.peer_metadata.update_model(model_id, self.own_ip)
        except Exception as e:
            print(f"[Model-Replication] Error updating own metadata: {e}")

        # 2. Get current holders (including self)
        current_holders = self.peer_metadata.get_model_nodes(model_id)
        current_count = len(current_holders)
        needed = self.REPLICATION_FACTOR - current_count

        if needed <= 0:
            print(f"[Model-Replication] Model {model_id} satisfies RF={self.REPLICATION_FACTOR}. No action needed.")
            return

        print(f"[Model-Replication] Model {model_id} needs {needed} more copies.")

        # 3. Find candidates (Healthy IPs that are NOT in current_holders)
        all_peers = self.get_healthy_peers()
        candidates = [ip for ip in all_peers if ip not in current_holders]

        if not candidates:
            print("[Model-Replication] No available candidates found to replicate to.")
            return

        # 4. Select targets
        import random
        targets = random.sample(candidates, min(needed, len(candidates)))
        print(f"[Model-Replication] Selected targets: {targets}")

        # 5. Send model JSON to targets
        for ip in targets:
            try:
                print(f"[Model-Replication] Sending model {model_id} to {ip}...")
                response = requests.post(
                    f"http://{ip}:8000/api/v1/models/replicate",
                    data={"model_id": model_id, "nodes_ips": json.dumps(targets)},
                    files={"file": (f"{model_id}.json", model_json, "application/json")},
                    timeout=self.timeout,
                )
                if response.status_code == 200:
                    print(f"[Model-Replication] Successfully replicated to {ip}")
                    self.peer_metadata.update_model(model_id, ip)
                else:
                    print(f"[Model-Replication] Failed to send to {ip}: {response.text}")
            except Exception as e:
                print(f"[Model-Replication] Error sending to {ip}: {e}")

    def replicate_model_update(self, model_id: str, model_json: bytes):
        """
        Send a model update to nodes that already hold the model (according to peer metadata).
        This is used for updates: we do not create new holders here, just push the new
        JSON to nodes that are already known to have it.
        """
        try:
            current_holders = set(self.peer_metadata.get_model_nodes(model_id))
        except Exception as e:
            print(f"[Model-Update-Replication] Error getting holders for {model_id}: {e}")
            return

        # Exclude self
        own = self._get_own_ip()
        targets = [ip for ip in current_holders if ip != own]

        if not targets:
            print(f"[Model-Update-Replication] No targets to send update for {model_id}")
            return

        print(f"[Model-Update-Replication] Sending update for {model_id} to {targets}")

        for ip in targets:
            try:
                response = requests.post(
                    f"http://{ip}:8000/api/v1/models/replicate_update",
                    data={"model_id": model_id, "nodes_ips": json.dumps(targets)},
                    files={"file": (f"{model_id}.json", model_json, "application/json")},
                    timeout=self.timeout,
                )
                if response.status_code == 200:
                    print(f"[Model-Update-Replication] Successfully sent update to {ip}")
                else:
                    print(f"[Model-Update-Replication] Failed to send update to {ip}: {response.status_code} {response.text}")
            except Exception as e:
                print(f"[Model-Update-Replication] Error sending update to {ip}: {e}")

    def check_and_update_model_from_peers(self, model_id: str, local_model_data: dict, holders: set) -> Optional[dict]:
        """
        Check if any peer has a better version of the model and update local model if needed.
        
        A peer's model is better if:
        1. Training: peer's training_completed=True and ours is False
        2. Training: both not completed but peer's last_trained_batch > ours
        3. Predictions: peer has predictions for a dataset we don't have
        4. Predictions: peer has higher last_predicted_batch for a dataset
        
        If peer has better training -> replace entire model with peer's
        If peer has better predictions only -> update metadata.predictions_by_dataset
        
        Args:
            model_id: ID of the model to check
            local_model_data: Current local model data
            holders: Set of node IPs that have the model
            
        Returns:
            Updated model data if changes were made, None otherwise
        """
        own_ip = self._get_own_ip()
        
        # Extract local version info from metadata (training fields are inside metadata)
        metadata = local_model_data.get("metadata", {})
        local_training_completed = metadata.get("training_completed", False)
        local_last_trained_batch = metadata.get("last_trained_batch")
        local_last_predicted = metadata.get("last_predicted_batch_by_dataset", {}) or {}
        
        print(f"[check_and_update_model_from_peers] Checking model {model_id}")
        print(f"  Local: training_completed={local_training_completed}, last_trained_batch={local_last_trained_batch}")
        print(f"  Local predictions: {local_last_predicted}")
        
        # Track the best version found
        best_peer = None
        best_training_peer = None  # Peer with better training
        better_predictions_peers = {}  # dataset_id -> (peer_ip, peer's model data)
        
        for holder in holders:
            if holder == own_ip:
                continue
            
            # Check if peer is alive
            if not self.check_ip_alive(holder, remove_if_dead=False):
                continue
            
            try:
                # Call the compare endpoint on the peer
                compare_url = f"http://{holder}:8000/api/v1/models/compare/{model_id}"
                compare_payload = {
                    "model_id": model_id,
                    "training_completed": local_training_completed,
                    "last_trained_batch": local_last_trained_batch,
                    "last_predicted_batch_by_dataset": local_last_predicted
                }
                
                resp = requests.post(compare_url, json=compare_payload, timeout=self.timeout)
                
                if resp.status_code != 200:
                    print(f"[check_and_update_model_from_peers] Peer {holder} compare failed: {resp.status_code}")
                    continue
                
                compare_result = resp.json()
                
                if not compare_result.get("is_better", False):
                    print(f"[check_and_update_model_from_peers] Peer {holder} is not better")
                    continue
                
                print(f"[check_and_update_model_from_peers] Peer {holder} has better version!")
                print(f"  better_training={compare_result.get('better_training')}")
                print(f"  better_predictions={compare_result.get('better_predictions')}")
                
                # If peer has better training, we need to fetch the full model
                if compare_result.get("better_training", False):
                    best_training_peer = holder
                
                # Track which datasets this peer has better predictions for
                for dataset_id in compare_result.get("better_predictions", []):
                    if dataset_id not in better_predictions_peers:
                        better_predictions_peers[dataset_id] = holder
                        
            except Exception as e:
                print(f"[check_and_update_model_from_peers] Error comparing with {holder}: {e}")
                continue
        
        # No better version found
        if not best_training_peer and not better_predictions_peers:
            print(f"[check_and_update_model_from_peers] No better version found for {model_id}")
            return None
        
        updated_model = None
        
        # If a peer has better training, fetch and replace the entire model
        if best_training_peer:
            print(f"[check_and_update_model_from_peers] Fetching full model from {best_training_peer}")
            try:
                fetch_url = f"http://{best_training_peer}:8000/api/v1/models/{model_id}"
                resp = requests.get(fetch_url, timeout=self.timeout)
                
                if resp.status_code == 200:
                    peer_response = resp.json()
                    updated_model = peer_response.get("model_data", peer_response)
                    print(f"[check_and_update_model_from_peers] Replaced model with version from {best_training_peer}")
            except Exception as e:
                print(f"[check_and_update_model_from_peers] Error fetching model from {best_training_peer}: {e}")
        
        # If we only need to update predictions (and didn't replace the whole model)
        if better_predictions_peers and not updated_model:
            updated_model = local_model_data.copy()
            
            for dataset_id, peer_ip in better_predictions_peers.items():
                print(f"[check_and_update_model_from_peers] Fetching prediction data for {dataset_id} from {peer_ip}")
                try:
                    # Fetch the peer's model to get their prediction data
                    fetch_url = f"http://{peer_ip}:8000/api/v1/models/{model_id}"
                    resp = requests.get(fetch_url, timeout=self.timeout)
                    
                    if resp.status_code == 200:
                        peer_response = resp.json()
                        peer_model_data = peer_response.get("model_data", peer_response)
                        
                        # Update predictions_by_dataset in metadata
                        peer_metadata = peer_model_data.get("metadata", {})
                        peer_predictions_by_dataset = peer_metadata.get("predictions_by_dataset", {})
                        
                        if dataset_id in peer_predictions_by_dataset:
                            # Ensure our metadata structure exists
                            if "metadata" not in updated_model:
                                updated_model["metadata"] = {}
                            if "predictions_by_dataset" not in updated_model["metadata"]:
                                updated_model["metadata"]["predictions_by_dataset"] = {}
                            
                            # Copy the prediction data for this dataset
                            updated_model["metadata"]["predictions_by_dataset"][dataset_id] = \
                                peer_predictions_by_dataset[dataset_id]
                            
                            # Also update last_predicted_batch_by_dataset
                            if "last_predicted_batch_by_dataset" not in updated_model:
                                updated_model["last_predicted_batch_by_dataset"] = {}
                            
                            peer_last_predicted = peer_model_data.get("last_predicted_batch_by_dataset", {})
                            if dataset_id in peer_last_predicted:
                                updated_model["last_predicted_batch_by_dataset"][dataset_id] = \
                                    peer_last_predicted[dataset_id]
                            
                            print(f"[check_and_update_model_from_peers] Updated predictions for {dataset_id}")
                except Exception as e:
                    print(f"[check_and_update_model_from_peers] Error fetching predictions from {peer_ip}: {e}")
        
        return updated_model

    # =====================================================================
    # PREDICTION JSON REPLICATION METHODS
    # =====================================================================

    def load_prediction_json_content(self, model_id: str, dataset_id: str) -> Optional[bytes]:
        """
        Load prediction JSON content from local storage.
        
        Args:
            model_id: ID of the model used for the prediction
            dataset_id: ID of the dataset used for the prediction
            
        Returns:
            Prediction JSON file content as bytes, or None if not found
        """
        try:
            from config.manager import PREDICTIONS
            
            prediction_path = Path(PREDICTIONS) / f"{model_id}_{dataset_id}.json"
            
            if not prediction_path.exists():
                print(f"[load_prediction_json_content] Prediction file not found: {prediction_path}")
                return None
            
            with open(prediction_path, 'rb') as f:
                content = f.read()
            
            print(f"[load_prediction_json_content] Loaded prediction {model_id}_{dataset_id}: {len(content)} bytes")
            return content
            
        except Exception as e:
            print(f"[load_prediction_json_content] Error loading prediction {model_id}_{dataset_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_prediction_json_content(self, model_id: str, dataset_id: str, prediction_json: bytes) -> bool:
        """
        Save prediction JSON content to local storage.
        
        Args:
            model_id: ID of the model used for the prediction
            dataset_id: ID of the dataset used for the prediction
            prediction_json: Prediction JSON content as bytes
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from config.manager import PREDICTIONS
            
            predictions_dir = Path(PREDICTIONS)
            predictions_dir.mkdir(parents=True, exist_ok=True)
            
            prediction_path = predictions_dir / f"{model_id}_{dataset_id}.json"
            
            with open(prediction_path, 'wb') as f:
                f.write(prediction_json)
            
            print(f"[save_prediction_json_content] Saved prediction {model_id}_{dataset_id}: {len(prediction_json)} bytes")
            
            # Update metadata to track that this node has the prediction
            self.peer_metadata.update_prediction(model_id, dataset_id, self.own_ip)
            
            return True
            
        except Exception as e:
            print(f"[save_prediction_json_content] Error saving prediction {model_id}_{dataset_id}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def check_and_rereplicate_predictions(self, prediction_keys: set = None):
        """
        Check replication factor for predictions and re-replicate if needed.

        Args:
            prediction_keys: Optional set of specific prediction keys (model_id, dataset_id) to check.
                            If None, checks all known predictions.
        """
        if prediction_keys is None:
            prediction_keys = set(self.peer_metadata.prediction_jsons.keys())

        if not prediction_keys:
            print("[check_and_rereplicate_predictions] No predictions to check")
            return

        print(f"[check_and_rereplicate_predictions] Checking {len(prediction_keys)} prediction(s)")

        for key in prediction_keys:
            model_id, dataset_id = key
            current_holders = self.peer_metadata.get_prediction_nodes(model_id, dataset_id)
            current_count = len(current_holders)
            needed = self.REPLICATION_FACTOR - current_count

            if needed > 0:
                print(f"[check_and_rereplicate_predictions] Prediction {model_id}_{dataset_id} is under-replicated: "
                    f"{current_count}/{self.REPLICATION_FACTOR} copies. Need {needed} more.")

                # If this node has the prediction, initiate replication
                if self.own_ip in current_holders:
                    print(f"[check_and_rereplicate_predictions] This node has {model_id}_{dataset_id}, initiating re-replication")
                    try:
                        # Load prediction JSON from disk
                        prediction_bytes = self.load_prediction_json_content(model_id, dataset_id)
                        
                        if prediction_bytes:
                            self.replicate_prediction_json(model_id, dataset_id, prediction_bytes)
                        else:
                            print(f"[check_and_rereplicate_predictions] Could not load prediction {model_id}_{dataset_id} for re-replication")
                    except Exception as e:
                        print(f"[check_and_rereplicate_predictions] Error re-replicating prediction {model_id}_{dataset_id}: {e}")
                else:
                    print(f"[check_and_rereplicate_predictions] This node doesn't have {model_id}_{dataset_id}, held by: {current_holders}")
            else:
                print(f"[check_and_rereplicate_predictions] Prediction {model_id}_{dataset_id} has sufficient replicas: {current_count}/{self.REPLICATION_FACTOR}")

    def replicate_prediction_json(self, model_id: str, dataset_id: str, prediction_json: bytes):
        """
        Ensure the prediction JSON exists on at least `REPLICATION_FACTOR` nodes.
        """
        # 1. Update own state first (if we have it)
        try:
            self.peer_metadata.update_prediction(model_id, dataset_id, self.own_ip)
        except Exception as e:
            print(f"[Prediction-Replication] Error updating own metadata: {e}")

        # 2. Get current holders (including self)
        current_holders = self.peer_metadata.get_prediction_nodes(model_id, dataset_id)
        current_count = len(current_holders)
        needed = self.REPLICATION_FACTOR - current_count

        if needed <= 0:
            print(f"[Prediction-Replication] Prediction {model_id}_{dataset_id} satisfies RF={self.REPLICATION_FACTOR}. No action needed.")
            return

        print(f"[Prediction-Replication] Prediction {model_id}_{dataset_id} needs {needed} more copies.")

        # 3. Find candidates (Healthy IPs that are NOT in current_holders)
        all_peers = self.get_healthy_peers()
        candidates = [ip for ip in all_peers if ip not in current_holders]

        if not candidates:
            print("[Prediction-Replication] No available candidates found to replicate to.")
            return

        # 4. Select targets
        targets = random.sample(candidates, min(needed, len(candidates)))
        print(f"[Prediction-Replication] Selected targets: {targets}")

        # 5. Send prediction JSON to targets
        # Include origin node (self) and all targets in nodes_ips so recipients know all holders
        all_holders = list(current_holders) + targets
        for ip in targets:
            try:
                print(f"[Prediction-Replication] Sending prediction {model_id}_{dataset_id} to {ip}...")
                response = requests.post(
                    f"http://{ip}:8000/api/v1/predictions/replicate",
                    data={
                        "model_id": model_id,
                        "dataset_id": dataset_id,
                        "nodes_ips": json.dumps(all_holders)
                    },
                    files={"file": (f"{model_id}_{dataset_id}.json", prediction_json, "application/json")},
                    timeout=self.timeout,
                )
                if response.status_code == 200:
                    print(f"[Prediction-Replication] Successfully replicated to {ip}")
                    self.peer_metadata.update_prediction(model_id, dataset_id, ip)
                else:
                    print(f"[Prediction-Replication] Failed to send to {ip}: {response.text}")
            except Exception as e:
                print(f"[Prediction-Replication] Error sending to {ip}: {e}")

