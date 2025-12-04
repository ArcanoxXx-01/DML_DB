"""
TimeSync service: synchronizes this node's TIME_OFFSET_SECONDS with its
peers once per calendar month.

How it works:
1. A background thread wakes up every TIME_SYNC_CHECK_INTERVAL_SECONDS.
2. On wake-up it checks whether the calendar month changed compared to
   the last sync.
3. If yes, it fetches the timestamp from all healthy peers (via the
   /api/v1/health endpoint), computes the median offset relative to
   the local clock, and updates config.manager.TIME_OFFSET_SECONDS.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime
from statistics import median
from typing import List, Optional

import requests

import config.manager as manager
from core.middleware import Middleware


class TimeSync:
    """
    Time synchronization service.

    Use `start(middleware)` to launch the background loop and
    `stop()` to gracefully shut it down on app shutdown.
    """

    def __init__(self, check_interval: Optional[float] = None):
        """
        Args:
            check_interval: Seconds between wake-ups. Defaults to
                            manager.TIME_SYNC_CHECK_INTERVAL_SECONDS (1 hour).
        """
        self._interval = check_interval or manager.TIME_SYNC_CHECK_INTERVAL_SECONDS
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Store year-month of the last successful sync (e.g. (2025, 12))
        self._last_sync_month: Optional[tuple[int, int]] = None
        self._middleware: Optional[Middleware] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self, middleware: "Middleware"):
        """Start the background thread that checks/performs monthly sync."""
        if self._thread and self._thread.is_alive():
            print("[TimeSync] Already running, ignoring start()")
            return

        self._middleware = middleware
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print("[TimeSync] Background thread started")

    def stop(self):
        """Stop the background thread (called on shutdown)."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=15.0)
        print("[TimeSync] Background thread stopped")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _run_loop(self):
        # Small initial delay to let the middleware populate healthy peers
        self._stop_event.wait(10.0)

        while not self._stop_event.is_set():
            try:
                now = datetime.utcnow()
                current_month = (now.year, now.month)

                if self._last_sync_month != current_month:
                    print(f"[TimeSync] New month detected ({current_month}), syncing time...")
                    self._perform_sync()
                    self._last_sync_month = current_month
            except Exception as e:
                print(f"[TimeSync] Error during sync check: {e}")

            self._stop_event.wait(self._interval)

    def _perform_sync(self):
        """Query healthy peers, compute median offset, update TIME_OFFSET_SECONDS."""
        if not self._middleware:
            print("[TimeSync] Middleware not set, skipping sync")
            return

        peers = self._middleware.get_healthy_peers()
        if not peers:
            print("[TimeSync] No healthy peers found, keeping offset unchanged")
            return

        offsets: List[float] = []
        for ip in peers:
            offset = self._fetch_offset_from_peer(ip)
            if offset is not None:
                offsets.append(offset)

        if not offsets:
            print("[TimeSync] Could not fetch timestamp from any peer")
            return

        median_offset = median(offsets)
        manager.TIME_OFFSET_SECONDS = median_offset
        print(f"[TimeSync] Updated TIME_OFFSET_SECONDS to {median_offset:.3f} s "
              f"(sampled {len(offsets)} peers)")

    def _fetch_offset_from_peer(self, ip: str, port: int = 8000) -> Optional[float]:
        """
        Return the clock offset (peer_time - local_time) in seconds, or None on
        failure.
        """
        url = f"http://{ip}:{port}/api/v1/health"
        try:
            before = time.time()
            resp = requests.get(url, timeout=5.0)
            after = time.time()
            resp.raise_for_status()

            data = resp.json()
            peer_ts = data.get("timestamp")
            if peer_ts is None:
                return None

            # Round-trip estimate: assume symmetric latency -> peer time at midpoint
            local_mid = (before + after) / 2.0
            offset = peer_ts - local_mid
            return offset
        except Exception as e:
            print(f"[TimeSync] Failed to fetch timestamp from {ip}: {e}")
            return None
