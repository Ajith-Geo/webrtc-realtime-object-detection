"""Lightweight runtime resource limiter utilities.

Provides asynchronous polling of system CPU and (if available) GPU utilization
and exposes awaitable gates to keep usage under configured ceilings.

Strategy:
 - Sample CPU percent via psutil every POLL_INTERVAL seconds (non blocking).
 - Sample GPU utilization via NVIDIA Management Library (pynvml) if present.
 - Before each expensive inference we call `await wait_for_resources()` which
   sleeps in small increments while utilization is above limits, with an upper
   bound backoff to avoid starvation.
 - Uses a singleton pattern; cheap to import.

Environment variables:
  MAX_CPU_PCT (default 90)
  MAX_GPU_PCT (default 85)
  RESOURCE_POLL_INTERVAL (default 0.5s)
  RESOURCE_MIN_SLEEP_MS  (default 5)
  RESOURCE_MAX_SLEEP_MS  (default 100)
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Optional

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore

try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlShutdown  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    nvmlInit = None  # type: ignore


@dataclass
class ResourceSnapshot:
    cpu: float = 0.0
    gpu: Optional[float] = None


class ResourceLimiter:
    def __init__(self):
        self.max_cpu = float(os.getenv('MAX_CPU_PCT', '90'))
        self.max_gpu = float(os.getenv('MAX_GPU_PCT', '85'))
        self.poll_interval = float(os.getenv('RESOURCE_POLL_INTERVAL', '0.5'))
        self.min_sleep_ms = int(os.getenv('RESOURCE_MIN_SLEEP_MS', '5'))
        self.max_sleep_ms = int(os.getenv('RESOURCE_MAX_SLEEP_MS', '100'))
        self._snapshot = ResourceSnapshot()
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None
        self._gpu_handle = None
        # Init GPU if available
        if nvmlInit is not None:
            try:  # pragma: no cover - environment specific
                nvmlInit()
                self._gpu_handle = nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self._gpu_handle = None

    async def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._poll_loop())

    async def _poll_loop(self):  # pragma: no cover - timing loop
        # Prime psutil cpu_percent first call for meaningful value
        if psutil is not None:
            try:
                psutil.cpu_percent(interval=None)
            except Exception:
                pass
        while True:
            try:
                snap = ResourceSnapshot()
                if psutil is not None:
                    try:
                        snap.cpu = float(psutil.cpu_percent(interval=None))
                    except Exception:
                        snap.cpu = 0.0
                if self._gpu_handle is not None:
                    try:
                        util = nvmlDeviceGetUtilizationRates(self._gpu_handle)  # type: ignore
                        snap.gpu = float(util.gpu)
                    except Exception:
                        snap.gpu = None
                async with self._lock:
                    self._snapshot = snap
            except Exception:
                pass
            await asyncio.sleep(self.poll_interval)

    async def wait_for_resources(self):
        """Block (cooperatively) until resources below thresholds.

        Implements exponential backoff sleep between polls up to max_sleep_ms.
        """
        sleep_ms = self.min_sleep_ms
        while True:
            async with self._lock:
                snap = self._snapshot
            over_cpu = snap.cpu >= self.max_cpu if snap.cpu else False
            over_gpu = snap.gpu is not None and snap.gpu >= self.max_gpu
            if not over_cpu and not over_gpu:
                return
            await asyncio.sleep(sleep_ms / 1000.0)
            sleep_ms = min(self.max_sleep_ms, sleep_ms * 2)

    async def shutdown(self):  # pragma: no cover - simple cleanup
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass
        if self._gpu_handle is not None and nvmlInit is not None:
            try:
                nvmlShutdown()
            except Exception:
                pass


_limiter: Optional[ResourceLimiter] = None


def get_limiter() -> ResourceLimiter:
    global _limiter
    if _limiter is None:
        _limiter = ResourceLimiter()
    return _limiter
