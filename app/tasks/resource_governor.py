"""Adaptive resource governor for background tasks.

Watches how busy the machine is **outside** our own work and tells background
tasks how hard to push:

* when the machine is otherwise idle, tasks run at full speed and grab every
  core (throttle == 0, worker cap == the hard cap);
* when *other* applications start demanding CPU, or free memory runs low,
  tasks insert short cooperative sleeps at their checkpoints and shrink their
  thread pools — yielding the machine back to whatever the user is doing.

This implements the product rule "use the machine harder when it is idle, back
off when it starts doing something else."

Design notes
------------
* Only load caused by **other** processes matters.  A background task that
  saturates the CPU on an otherwise-idle machine is *desired* behaviour, so the
  governor subtracts our own process' CPU share from the system total before
  deciding to back off.
* System load is sampled at most once per second — a single ``psutil`` call
  behind a lock — and every reader (potentially many worker threads) shares
  that cached sample.  The per-checkpoint cost is therefore a dict/lock read.
* Everything degrades gracefully to "no throttling" when ``psutil`` is missing
  or the feature is disabled in Settings.

The governor is a cheap, thread-safe singleton obtained via
:func:`get_resource_governor`.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Tuple

log = logging.getLogger(__name__)

#: Minimum wall-clock gap between real psutil samples (seconds).
_SAMPLE_INTERVAL = 1.0

#: External CPU load (0..1) below which we never back off — the machine is
#: effectively ours to use.
_CPU_LOW = 0.20
#: External CPU load at/above which we back off as hard as allowed.
_CPU_HIGH = 0.85

#: Available-RAM fraction where memory back-off begins …
_MEM_PRESSURE_START = 0.25
#: … and where it reaches full strength.
_MEM_PRESSURE_FULL = 0.10

#: At full load, sleep up to this multiple of the measured work slice at each
#: checkpoint (1.0 → roughly a 50% duty cycle for a fully-yielding task).
_MAX_DUTY = 1.0
#: Never sleep longer than this in one checkpoint, so cancel stays responsive.
_MAX_SLEEP = 0.25
#: Work slices longer than this (a pause, a stall, first checkpoint) are clamped
#: so a long gap can never translate into a long sleep.
_WORK_WINDOW_CAP = 2.0

_SETTING_KEY = "tasks/adaptive_throttle_enabled"


def _ramp(value: float, lo: float, hi: float) -> float:
    """Linear 0→1 ramp of *value* across the ``[lo, hi]`` window, clamped."""
    if hi <= lo:
        return 1.0 if value >= hi else 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _read_enabled() -> bool:
    """Whether adaptive throttling is on (Settings → Task Manager, default on)."""
    try:
        from app.app_settings import app_qsettings

        return app_qsettings().value(_SETTING_KEY, True, type=bool)
    except Exception:  # noqa: BLE001 — settings optional, never break tasks
        return True


class ResourceGovernor:
    """Samples system pressure and advises tasks on how much to back off."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_sample = 0.0
        self._load = 0.0
        self._enabled = True
        self._ncores = 1
        self._psutil = None
        self._proc = None
        # Test hook: (load, enabled) overriding all live sampling when set.
        self._override: Optional[Tuple[float, bool]] = None
        self._init_psutil()

    def _init_psutil(self) -> None:
        try:
            import psutil

            self._psutil = psutil
            self._proc = psutil.Process()
            self._proc.cpu_percent(None)   # prime the per-process sampler
            psutil.cpu_percent(None)       # prime the system-wide sampler
            self._ncores = psutil.cpu_count(logical=True) or 1
        except Exception:  # noqa: BLE001 — psutil optional
            self._psutil = None
            self._proc = None

    # -- sampling ---------------------------------------------------------

    def _refresh_locked(self, now: float) -> None:
        """Re-read the enabled flag and system load (caller holds the lock)."""
        self._last_sample = now
        self._enabled = _read_enabled()
        if self._psutil is None:
            self._load = 0.0
            return
        try:
            total = self._psutil.cpu_percent(None) / 100.0
            own = (self._proc.cpu_percent(None) / self._ncores) / 100.0
            external = max(0.0, total - own)
            cpu_comp = _ramp(external, _CPU_LOW, _CPU_HIGH)

            vm = self._psutil.virtual_memory()
            avail = (vm.available / vm.total) if vm.total else 1.0
            mem_comp = _ramp(
                _MEM_PRESSURE_START - avail,
                0.0,
                _MEM_PRESSURE_START - _MEM_PRESSURE_FULL,
            )

            self._load = max(cpu_comp, mem_comp)
        except Exception:  # noqa: BLE001 — a sampling hiccup must never break work
            self._load = 0.0

    def load(self) -> float:
        """Current back-off pressure in ``[0.0, 1.0]`` (0 == run at full speed).

        Returns 0 when the feature is disabled or psutil is unavailable.  A
        real sample is taken at most once per :data:`_SAMPLE_INTERVAL` seconds;
        between samples the cached value is returned.
        """
        with self._lock:
            if self._override is not None:
                ov_load, ov_enabled = self._override
                return ov_load if ov_enabled else 0.0
            now = time.monotonic()
            if now - self._last_sample >= _SAMPLE_INTERVAL:
                self._refresh_locked(now)
            return self._load if self._enabled else 0.0

    # -- advice -----------------------------------------------------------

    def throttle_delay(self, work_slice: float, weight: float = 1.0) -> float:
        """Seconds to sleep after a work slice of *work_slice* seconds.

        *weight* scales how strongly this task yields (0 == never sleep, 1 ==
        yield fully).  The result is proportional to the work just done, so it
        self-scales to whatever checkpoint frequency a task happens to use, and
        is capped at :data:`_MAX_SLEEP` for cancel responsiveness.
        """
        if weight <= 0.0:
            return 0.0
        load = self.load()
        if load <= 0.0:
            return 0.0
        slice_ = min(max(work_slice, 0.0), _WORK_WINDOW_CAP)
        return min(slice_ * _MAX_DUTY * load * weight, _MAX_SLEEP)

    def recommended_workers(self, hard_cap: int) -> int:
        """Thread-pool size for the current load, in ``[1, hard_cap]``.

        Full ``hard_cap`` when the machine is idle; scales down linearly to 1
        as external CPU/memory pressure rises.
        """
        hard_cap = max(1, int(hard_cap))
        load = self.load()
        if load <= 0.0:
            return hard_cap
        workers = round(hard_cap - load * (hard_cap - 1))
        return max(1, min(hard_cap, int(workers)))

    # -- test hooks -------------------------------------------------------

    def set_test_override(self, load: Optional[float], enabled: bool = True) -> None:
        """Force a fixed (load, enabled) pair, bypassing live sampling.

        Pass ``load=None`` to clear the override and resume real sampling.
        """
        with self._lock:
            if load is None:
                self._override = None
            else:
                self._override = (max(0.0, min(1.0, load)), enabled)
            self._last_sample = 0.0  # force a fresh sample once cleared


_governor: Optional[ResourceGovernor] = None
_governor_lock = threading.Lock()


def get_resource_governor() -> ResourceGovernor:
    """Application-wide singleton (created on first use)."""
    global _governor
    if _governor is None:
        with _governor_lock:
            if _governor is None:
                _governor = ResourceGovernor()
    return _governor
