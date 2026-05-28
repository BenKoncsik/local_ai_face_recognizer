"""Google Drive database synchronisation.

In Google Drive mode the application never opens faces.db from a persistent
local path.  Instead:

1. ``open()``  — download the DB from Drive into the OS cache dir.
2. Work happens against the local cache copy (normal SQLAlchemy / SQLite).
3. ``sync()``  — upload the modified DB back to Drive.
4. ``close()`` — release the active-reference lock; the file stays on disk
                  until the next startup cleanup removes it.

If there is no internet connection, ``open()`` raises
:class:`~app.gdrive.connectivity.GDriveOfflineError` with a user-friendly
message.  There is no offline fallback and no persistent local copy.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from app.gdrive.cache import GDriveCacheManager, get_cache_manager
from app.gdrive.connectivity import require_online

if TYPE_CHECKING:
    from app.gdrive.protocols import GDriveClient

log = logging.getLogger(__name__)

_LOCAL_DB_FILENAME = "faces.db"


class GDriveDbSync:
    """Manages the working copy of ``faces.db`` during a Google Drive session."""

    def __init__(
        self,
        drive_db_path: str,
        cache_manager: Optional[GDriveCacheManager] = None,
    ) -> None:
        """
        Args:
            drive_db_path: Path (or file ID) of ``faces.db`` on Google Drive.
            cache_manager:  Override the module-level singleton (useful in tests).
        """
        self._drive_path = drive_db_path
        self._cache = cache_manager or get_cache_manager()
        self._local: Optional[Path] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def local_db_path(self) -> Optional[Path]:
        """The active local copy of the database, or ``None`` if not open."""
        return self._local

    def open(self, client: "GDriveClient") -> Path:
        """Download faces.db from Drive into the cache dir.

        Returns the local :class:`~pathlib.Path` to be used as the SQLite
        database for this session.

        Raises:
            GDriveOfflineError: if the host is unreachable.
            RuntimeError: if the download fails for any other reason.
        """
        require_online()

        local = self._cache.allocate_path(_LOCAL_DB_FILENAME)
        self._cache.mark_active(local)
        try:
            client.download_file(self._drive_path, str(local))
        except Exception:
            self._cache.release(local, delete=True)
            raise

        self._local = local
        log.info("GDrive DB opened — local copy at: %s", local)
        return local

    def sync(self, client: "GDriveClient") -> None:
        """Upload the current local DB back to Google Drive.

        Raises:
            RuntimeError: if ``open()`` was never called or the file is gone.
            GDriveOfflineError: if the host is unreachable.
        """
        if self._local is None or not self._local.exists():
            raise RuntimeError("GDriveDbSync.sync() called without an open session.")
        require_online()
        client.upload_file(str(self._local), self._drive_path)
        log.info("GDrive DB synced back to Drive: %s", self._drive_path)

    def close(self) -> None:
        """Release the active reference.

        The local file is left on disk for the *next* startup cleanup to
        remove, so it is never forcibly deleted while the process might still
        be winding down its DB connections.
        """
        if self._local is not None:
            self._cache.release(self._local, delete=False)
            log.info("GDrive DB session closed, cache path: %s", self._local)
            self._local = None
