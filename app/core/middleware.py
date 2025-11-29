from typing import Optional, Type, TypeVar, Generic, List
import socket
import requests
import threading
import time
from datetime import datetime
import random


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
        self.ip_cache: dict[str, List[str]] = {}
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        self.discovery_thread: Optional[threading.Thread] = None
        self.stop_discovery = threading.Event()
        self.cache_lock = threading.Lock()
        
        print("Middleware initialized with own IP:", self._get_own_ip())
    
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
        print(f"[check_cache_ips_alive] Checking cached IPs for domain: {domain}")
        results = {}
        
        with self.cache_lock:
            if domain not in self.ip_cache:
                print(f"[check_cache_ips_alive] Domain {domain} not found in cache")
                return results
            
            # Create a copy to avoid modification during iteration
            ips_to_check = self.ip_cache[domain].copy()
        
        print(f"[check_cache_ips_alive] Found {len(ips_to_check)} IPs to check: {ips_to_check}")
        for ip in ips_to_check:
            is_alive = self.check_ip_alive(ip, port, health_path, remove_if_dead=remove_dead)
            results[ip] = is_alive
        
        alive_count = sum(1 for status in results.values() if status)
        print(f"[check_cache_ips_alive] Results: {alive_count}/{len(results)} IPs alive - {results}")
        return results
        
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
            
            # Wait 10 seconds or until stop signal
            self.stop_discovery.wait(10.0)
        
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
        