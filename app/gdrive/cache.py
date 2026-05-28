"""Google Drive cache manager — ephemeral, OS-level cache directory only.

Rules enforced here:
* All downloaded files go to the OS cache/temp directory, never the app data dir.
* Files are deleted as soon as the operation that needed them completes.
* An active-reference set prevents deletion of files that are currently open.
* Startup cleanup removes leftover files from any previous (crashed) run.
* Shutdown cleanup removes all unreferenced files on normal exit.
* Cache size is capped at MAX_CACHE_BYTES; oldest unreferenced files are evicted
  when the limit is approached.
* Failed deletes are retried with exponential back-off; anything still stuck is
  logged and will be caught by the next startup cleanup.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from pathlib import Path
from typing import Iterator, Optional, Set

log = logging.getLogger(__name__)

MAX_CACHE_BYTES: int = 500 * 1024 * 1024  # 500 MB hard cap
_RETRY_DELAYS = (0.25, 0.5, 1.0, 2.0, 4.0)  # seconds between delete retries


class GDriveCacheManager:
    """Lifecycle manager for files downloaded from Google Drive.

    All files are kept inside *cache_dir* (an OS temp/cache path).  The
    manager tracks which files are currently in use and never deletes those.
    Everything else is swept on startup and shutdown.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._root: Path = cache_dir.resolve()
        self._lock = threading.Lock()
        self._active: Set[Path] = set()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def cache_dir(self) -> Path:
        return self._root

    # ------------------------------------------------------------------
    # Lifecycle hooks (called from main.py)
    # ------------------------------------------------------------------

    def startup_cleanup(self) -> int:
        """Delete ALL files left over from a previous run.

        Called once at application start, before any Drive operation begins.
        Returns the number of files successfully removed.
        """
        if not self._root.exists():
            return 0
        n = 0
        for path in sorted(self._root.rglob("*"), reverse=True):
            if path.is_file():
                n += self._delete_with_retry(path)
            elif path.is_dir() and not any(path.iterdir()):
                try:
                    path.rmdir()
                except OSError:
                    pass
        if n:
            log.info("GDrive cache: startup cleanup removed %d file(s) from %s", n, self._root)
        return n

    def shutdown_cleanup(self) -> int:
        """Delete all files not currently held by an active reference.

        Called when the application is about to exit.
        Returns the number of files successfully removed.
        """
        if not self._root.exists():
            return 0
        with self._lock:
            active = set(self._active)
        n = 0
        for path in sorted(self._root.rglob("*"), reverse=True):
            if path.is_file() and path not in active:
                n += self._delete_with_retry(path)
            elif path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
        if n:
            log.info("GDrive cache: shutdown cleanup removed %d file(s)", n)
        return n

    # ------------------------------------------------------------------
    # File lifecycle — context-manager API
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def temp_file(self, filename: str) -> Iterator[Path]:
        """Yield a path in the cache dir; delete the file when the block exits.

        Usage::

            with cache.temp_file("img_123.jpg") as path:
                drive_client.download(file_id, path)
                process(path)
            # file is deleted here
        """
        self._ensure_dir()
        self._enforce_size_limit()
        path = (self._root / filename).resolve()
        self.mark_active(path)
        try:
            yield path
        finally:
            self.release(path, delete=True)

    # ------------------------------------------------------------------
    # Manual reference tracking (for callers that can't use a context manager)
    # ------------------------------------------------------------------

    def allocate_path(self, filename: str) -> Path:
        """Return a resolved path for a new cache file (does not create it)."""
        self._ensure_dir()
        self._enforce_size_limit()
        return (self._root / filename).resolve()

    def mark_active(self, path: Path) -> None:
        """Register *path* as actively in use; prevents deletion until released."""
        with self._lock:
            self._active.add(path.resolve())

    def release(self, path: Path, *, delete: bool = True) -> None:
        """Deregister *path* and optionally delete it from disk."""
        resolved = path.resolve()
        with self._lock:
            self._active.discard(resolved)
        if delete and resolved.exists():
            self._delete_with_retry(resolved)

    # ------------------------------------------------------------------
    # Size accounting
    # ------------------------------------------------------------------

    def current_size_bytes(self) -> int:
        """Return the total byte size of all files currently in the cache."""
        if not self._root.exists():
            return 0
        total = 0
        for p in self._root.rglob("*"):
            if p.is_file():
                with contextlib.suppress(OSError):
                    total += p.stat().st_size
        return total

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_dir(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)

    def _enforce_size_limit(self) -> None:
        """Evict the oldest unreferenced files until we are under MAX_CACHE_BYTES."""
        if self.current_size_bytes() <= MAX_CACHE_BYTES:
            return
        with self._lock:
            active = set(self._active)
        candidates = sorted(
            (p for p in self._root.rglob("*") if p.is_file() and p not in active),
            key=lambda p: p.stat().st_mtime,
        )
        for candidate in candidates:
            if self.current_size_bytes() <= MAX_CACHE_BYTES:
                break
            self._delete_with_retry(candidate)
            log.debug("GDrive cache evicted (size limit): %s", candidate.name)

    def _delete_with_retry(self, path: Path) -> int:
        """Try to delete *path*, retrying with back-off on failure.

        Returns 1 on success, 0 if all retries were exhausted.  A failed
        delete is NOT fatal — it will be cleaned up on the next startup.
        """
        for i, delay in enumerate((0, *_RETRY_DELAYS)):
            if delay:
                time.sleep(delay)
            try:
                path.unlink(missing_ok=True)
                return 1
            except OSError as exc:
                log.debug(
                    "GDrive cache delete attempt %d failed for %s: %s",
                    i + 1, path.name, exc,
                )
        log.warning(
            "GDrive cache: could not delete %s after %d attempts — "
            "will be removed on next startup",
            path, len(_RETRY_DELAYS) + 1,
        )
        return 0


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_manager: Optional[GDriveCacheManager] = None
_manager_lock = threading.Lock()


def get_cache_manager() -> GDriveCacheManager:
    """Return the application-wide :class:`GDriveCacheManager` instance."""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                from app.paths import gdrive_cache_dir
                _manager = GDriveCacheManager(gdrive_cache_dir())
    return _manager
