from typing import Optional, Type, TypeVar, Generic, List
import socket
import requests
import threading
import time
from datetime import datetime
import random

from core.peer_metadata import PeerMetadata
import json
from api.services.models_services import load_model


class Middleware:
    """Middleware class for sending HTTP requests with Pydantic schema validation and IP caching."""
    
    def __init__(self, timeout: float = 30.0, health_check_timeout: float = 10.0):
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
        self.ip_cache: dict[str, List[str]] = {}
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        self.discovery_thread: Optional[threading.Thread] = None
        self.stop_discovery = threading.Event()
        self.cache_lock = threading.Lock()
        self.peer_metadata: PeerMetadata = PeerMetadata(self.own_ip)
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

        return results

    def _print_datasets_state(self, label: str):
        """
        Pretty print the current state of datasets and their replicas.
        
        Args:
            label: Label to identify this state snapshot
        """
        print(f"\n📊 DATASETS STATE - {label}")
        print(f"{'─'*80}")
        
        with self.peer_metadata.lock:
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
        
        print(f"{'─'*80}")

    def cleanup_dead_peers(self, dead_ips: List[str]):
        """
        Remove dead peers from peer metadata and from cache.
        After cleanup, check all datasets and re-replicate those that are under-replicated.
        
        Args:
            dead_ips: List of IP addresses that are dead/unreachable
        """
        if not dead_ips:
            return
            
        print(f"[cleanup_dead_peers] Cleaning up {len(dead_ips)} dead peer(s): {dead_ips}")
        
        # Track which datasets and models were affected
        affected_datasets = set()
        affected_models = set()
        
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
                        if ip in holders:
                            models_on_peer.add(model_id)
                affected_models.update(models_on_peer)

                self.peer_metadata.remove_peer(ip)
            except Exception as e:
                print(f"[cleanup_dead_peers] Error removing peer {ip} from metadata: {e}")
        
        # Check and re-replicate affected datasets and models
        if affected_datasets:
            print(f"[cleanup_dead_peers] {len(affected_datasets)} dataset(s) potentially affected: {affected_datasets}")
            self.check_and_rereplicate_datasets(affected_datasets)
        if affected_models:
            print(f"[cleanup_dead_peers] {len(affected_models)} model(s) potentially affected: {affected_models}")
            self.check_and_rereplicate_models(affected_models)
        else:
            print(f"[cleanup_dead_peers] No datasets affected by peer removal")

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
                        resp = load_model(model_id)
                        if resp and isinstance(resp, dict) and "model_data" in resp:
                            try:
                                model_bytes = json.dumps(resp["model_data"]).encode("utf-8")
                                self.replicate_model_json(model_id, model_bytes)
                            except Exception as e:
                                print(f"[check_and_rereplicate_models] Error serializing model {model_id}: {e}")
                        else:
                            print(f"[check_and_rereplicate_models] Could not load model {model_id} for re-replication")
                    except Exception as e:
                        print(f"[check_and_rereplicate_models] Error re-replicating model {model_id}: {e}")
                else:
                    print(f"[check_and_rereplicate_models] This node doesn't have {model_id}, held by: {current_holders}")
            else:
                print(f"[check_and_rereplicate_models] Model {model_id} has sufficient replicas: {current_count}/{self.REPLICATION_FACTOR}")
                
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

            # Wait 5 seconds or until stop signal
            self.stop_discovery.wait(5.0)
        
        print(f"[_discover_ips] IP discovery thread stopped")
    
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

    def replicate_model_json(self, model_id: str, model_json: bytes):
        """
        Ensure the model JSON exists on at least `REPLICATION_FACTOR` nodes.
        Similar to `replicate_dataset` but for model JSONs.
        """
        # 1. Update own state first
        try:
            self.peer_metadata.update_model(model_id, self.own_ip)
        except Exception:
            pass

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
        targets = random.sample(candidates, min(needed, len(candidates)))
        print(f"[Model-Replication] Selected targets: {targets}")

        # 5. Send model JSON to targets
        for ip in targets:
            try:
                print(f"[Model-Replication] Sending model {model_id} to {ip}...")
                response = requests.post(
                    f"http://{ip}:8000/api/v1/models/replicate",
                    data={"model_id": model_id, "nodes_ips": targets},
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
        