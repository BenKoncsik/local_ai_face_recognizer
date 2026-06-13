"""Lightweight performance instrumentation.

Central measurement points for the whole application:

* :func:`timed_block` / :func:`timed` — wall-clock timing of any code block,
  aggregated per name (count / total / max) and logged when slow.
* :func:`attach_sql_timing` — per-statement SQL timing on a SQLAlchemy engine;
  counts every query and warns about slow ones.
* :class:`PerfCounter` — hit/miss counters for caches.
* :func:`memory_mb` — current process RSS.
* :func:`report` — human-readable summary of everything recorded so far.

The overhead is a dict update behind a lock per measurement, so the
instrumentation can stay enabled in production builds.  All records are
in-memory only; nothing is persisted.
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, Generator, Optional

log = logging.getLogger("app.perf")

# Operations slower than this are logged at WARNING level.
SLOW_OP_MS = 200.0
# SQL statements slower than this are logged at WARNING level.
SLOW_SQL_MS = 50.0


@dataclass
class PerfRecord:
    """Aggregated timing for one named operation."""

    name: str
    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0
    last_ms: float = 0.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0


@dataclass
class PerfCounter:
    """Hit/miss counter for caches."""

    name: str
    hits: int = 0
    misses: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def hit(self) -> None:
        with self._lock:
            self.hits += 1

    def miss(self) -> None:
        with self._lock:
            self.misses += 1

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


_records: Dict[str, PerfRecord] = {}
_counters: Dict[str, PerfCounter] = {}
_lock = threading.Lock()


def record(name: str, elapsed_ms: float, *, slow_ms: Optional[float] = None) -> None:
    """Add one timing sample under *name* and warn when it is slow."""
    with _lock:
        rec = _records.get(name)
        if rec is None:
            rec = _records[name] = PerfRecord(name=name)
        rec.count += 1
        rec.total_ms += elapsed_ms
        rec.last_ms = elapsed_ms
        if elapsed_ms > rec.max_ms:
            rec.max_ms = elapsed_ms
    threshold = SLOW_OP_MS if slow_ms is None else slow_ms
    if elapsed_ms >= threshold:
        log.warning("SLOW %s: %.0f ms", name, elapsed_ms)
    else:
        log.debug("%s: %.1f ms", name, elapsed_ms)


@contextmanager
def timed_block(name: str, *, slow_ms: Optional[float] = None) -> Generator[None, None, None]:
    """Time the enclosed block and record it under *name*."""
    start = time.perf_counter()
    try:
        yield
    finally:
        record(name, (time.perf_counter() - start) * 1000.0, slow_ms=slow_ms)


def timed(name: Optional[str] = None, *, slow_ms: Optional[float] = None) -> Callable:
    """Decorator variant of :func:`timed_block`."""

    def decorator(fn: Callable) -> Callable:
        op_name = name or f"{fn.__module__}.{fn.__qualname__}"

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with timed_block(op_name, slow_ms=slow_ms):
                return fn(*args, **kwargs)

        return wrapper

    return decorator


def counter(name: str) -> PerfCounter:
    """Return (creating on first use) the cache counter called *name*."""
    with _lock:
        c = _counters.get(name)
        if c is None:
            c = _counters[name] = PerfCounter(name=name)
        return c


def memory_mb() -> float:
    """Current process resident set size in MiB (0.0 when unavailable)."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:  # noqa: BLE001 — psutil optional
        pass
    try:
        import resource
        import sys

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is bytes on macOS, kilobytes on Linux.
        return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024
    except Exception:  # noqa: BLE001
        return 0.0


def attach_sql_timing(engine, *, slow_ms: float = SLOW_SQL_MS) -> None:
    """Instrument a SQLAlchemy engine: count queries, warn on slow statements.

    Aggregates under ``sql.total``; slow statements are logged with the first
    120 characters of the SQL so the offending query is identifiable.
    """
    from sqlalchemy import event

    @event.listens_for(engine, "before_cursor_execute")
    def _before(conn, cursor, statement, parameters, context, executemany):  # noqa: ANN001
        conn.info.setdefault("perf_query_start", []).append(time.perf_counter())

    @event.listens_for(engine, "after_cursor_execute")
    def _after(conn, cursor, statement, parameters, context, executemany):  # noqa: ANN001
        starts = conn.info.get("perf_query_start")
        if not starts:
            return
        elapsed_ms = (time.perf_counter() - starts.pop()) * 1000.0
        record("sql.total", elapsed_ms, slow_ms=slow_ms)
        if elapsed_ms >= slow_ms:
            log.warning("SLOW SQL %.0f ms: %s", elapsed_ms, statement[:120])


def snapshot() -> Dict[str, PerfRecord]:
    """Copy of all timing records (safe to iterate)."""
    with _lock:
        return dict(_records)


def reset() -> None:
    """Drop all collected records and counters (used by benchmarks/tests)."""
    with _lock:
        _records.clear()
        _counters.clear()


def report() -> str:
    """Multi-line human-readable summary of all recorded measurements."""
    with _lock:
        records = sorted(_records.values(), key=lambda r: r.total_ms, reverse=True)
        counters = list(_counters.values())
    lines = [
        f"{'operation':<48} {'count':>7} {'total ms':>10} {'avg ms':>8} {'max ms':>8}"
    ]
    for r in records:
        lines.append(
            f"{r.name:<48} {r.count:>7} {r.total_ms:>10.1f} {r.avg_ms:>8.1f} {r.max_ms:>8.1f}"
        )
    for c in counters:
        lines.append(
            f"cache {c.name}: hits={c.hits} misses={c.misses} hit_rate={c.hit_rate:.1%}"
        )
    lines.append(f"process RSS: {memory_mb():.1f} MiB")
    return "\n".join(lines)
