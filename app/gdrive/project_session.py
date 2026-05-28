"""End-to-end Google Drive project session.

A "project" on Drive is just a folder owned (or shared with) the user.
Inside that folder Face-Local expects this layout::

    <project root>/
      database/
          faces.db
          faces.db.lock        # heartbeat-protected lock file
      metadata/
          project.json         # project descriptor (created on first open)
      <images …>               # arbitrary picture files, possibly nested

This module owns the full lifecycle:

* ``open()`` — verify connectivity, acquire the lock, download faces.db
  to the OS cache dir (or initialise an empty one if none exists yet),
  start a background heartbeat thread.
* ``mark_dirty()`` — called by services after any DB mutation so the
  session knows the local copy diverged from the Drive copy.
* ``sync()`` — checkpoint the SQLite WAL, upload faces.db to Drive,
  refresh the heartbeat.
* ``close()`` — best-effort final sync if dirty, release the lock,
  release the local DB reference (the cache manager then deletes it).

Concurrency
-----------

Drive cannot prevent two clients from opening the same project, so we
implement a soft heartbeat-based lock.  A lock file is fresh while its
``last_heartbeat`` is within :data:`_STALE_AFTER_SECONDS`; older locks
are considered stale and silently overridden.  Genuine conflicts surface
through the :class:`ProjectLocked` exception with all metadata so the UI
can warn the user.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import socket
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app import __version__
from app.gdrive.cache import GDriveCacheManager, get_cache_manager
from app.gdrive.connectivity import require_online
from app.gdrive.protocols import DriveFile, GDriveClient

log = logging.getLogger(__name__)

# Names on Drive — kept as constants so both reader and writer agree.
_DB_SUBFOLDER = "database"
_METADATA_SUBFOLDER = "metadata"
_DB_FILENAME = "faces.db"
_LOCK_FILENAME = "faces.db.lock"
_PROJECT_DESCRIPTOR = "project.json"

# Lock heartbeat cadence and staleness threshold.
_HEARTBEAT_SECONDS = 60
_STALE_AFTER_SECONDS = 300  # 5 min — a lock older than this is considered abandoned


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProjectFolders:
    """Resolved Drive file_ids for the standard project sub-folders."""

    root_id: str
    root_name: str
    database_folder_id: str
    metadata_folder_id: str
    db_file_id: Optional[str]
    lock_file_id: Optional[str]


@dataclass
class LockInfo:
    """Contents of ``faces.db.lock`` on Drive."""

    lock_id: str
    device_name: str
    user_email: str
    app_version: str
    started_at: str
    last_heartbeat: str

    @classmethod
    def from_json(cls, raw: str) -> "LockInfo":
        return cls(**json.loads(raw))

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    def is_stale(self, threshold_seconds: int = _STALE_AFTER_SECONDS) -> bool:
        try:
            hb = datetime.fromisoformat(self.last_heartbeat)
        except ValueError:
            return True
        if hb.tzinfo is None:
            hb = hb.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - hb).total_seconds()
        return age > threshold_seconds


@dataclass
class SyncStatus:
    """Summary of the session's sync state — shown in the UI."""

    last_sync_at: Optional[str] = None
    last_sync_succeeded: bool = True
    last_error: Optional[str] = None
    last_uploaded_bytes: int = 0
    dirty: bool = False
    current_op: Optional[str] = None  # e.g. "Uploading database", "Downloading image"

    def snapshot(self) -> "SyncStatus":
        return SyncStatus(**asdict(self))


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ProjectLocked(RuntimeError):
    """Raised when another active session already holds the project lock."""

    def __init__(self, info: LockInfo) -> None:
        super().__init__(
            f"Ezt a Google Drive projektet másik gép használja: "
            f"{info.device_name} ({info.user_email}), utolsó heartbeat: "
            f"{info.last_heartbeat}"
        )
        self.info = info


class ProjectFolderInvalid(RuntimeError):
    """Raised when the selected Drive folder is unsuitable as a project root."""


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class GDriveProjectSession:
    """Owns the lifecycle of an opened Google Drive project."""

    def __init__(
        self,
        client: GDriveClient,
        project_root_id: str,
        *,
        cache: Optional[GDriveCacheManager] = None,
        heartbeat_interval: int = _HEARTBEAT_SECONDS,
        force_unlock: bool = False,
    ) -> None:
        self._client = client
        self._root_id = project_root_id
        self._cache = cache or get_cache_manager()
        self._heartbeat_interval = heartbeat_interval
        self._force_unlock = force_unlock

        self._folders: Optional[ProjectFolders] = None
        self._lock_info: Optional[LockInfo] = None
        self._local_db_path: Optional[Path] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._status = SyncStatus()
        self._lock = threading.RLock()  # protects status + folders + lock_info

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def folders(self) -> Optional[ProjectFolders]:
        return self._folders

    @property
    def local_db_path(self) -> Optional[Path]:
        return self._local_db_path

    @property
    def status(self) -> SyncStatus:
        with self._lock:
            return self._status.snapshot()

    def mark_dirty(self) -> None:
        """Flag the local DB as having diverged from Drive."""
        with self._lock:
            self._status.dirty = True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> Path:
        """Acquire the lock, download the DB, return the local DB path.

        Raises:
            GDriveOfflineError:    No internet connectivity.
            ProjectLocked:         Another fresh lock is in place.
            ProjectFolderInvalid:  The folder structure cannot be created.
        """
        require_online()

        log.info("Opening Drive project: root_id=%s", self._root_id)
        self._set_op("Listing project folder")
        folders = self._resolve_folders()
        with self._lock:
            self._folders = folders

        self._set_op("Acquiring lock")
        self._acquire_lock(folders)

        self._set_op("Downloading database")
        local_db = self._download_or_init_db(folders)
        self._local_db_path = local_db

        self._start_heartbeat()
        self._set_op(None)
        log.info("Drive project opened — local DB: %s", local_db)
        return local_db

    def sync(self) -> None:
        """Upload faces.db to Drive (if dirty) and refresh the heartbeat."""
        if self._folders is None:
            raise RuntimeError("Session not opened.")
        require_online()

        with self._lock:
            dirty = self._status.dirty
            local = self._local_db_path

        if not dirty:
            log.debug("Sync skipped — DB not dirty.")
            self._refresh_heartbeat()
            return
        if local is None or not local.exists():
            log.warning("Sync requested but local DB missing: %s", local)
            return

        self._set_op("Uploading database")
        try:
            self._checkpoint_sqlite(local)
            new_id = self._client.upload_file(
                str(local),
                parent_folder_id=self._folders.database_folder_id,
                remote_name=_DB_FILENAME,
                existing_file_id=self._folders.db_file_id,
                mime_type="application/x-sqlite3",
            )
            with self._lock:
                self._folders = ProjectFolders(
                    **{**asdict(self._folders), "db_file_id": new_id}
                )
                self._status.last_sync_at = _utc_now_iso()
                self._status.last_sync_succeeded = True
                self._status.last_error = None
                self._status.last_uploaded_bytes = local.stat().st_size
                self._status.dirty = False
            log.info("DB sync successful — %d bytes uploaded.", local.stat().st_size)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._status.last_sync_succeeded = False
                self._status.last_error = str(exc)
            log.error("DB sync failed: %s", exc)
            raise
        finally:
            self._set_op(None)

        self._refresh_heartbeat()

    def close(self, *, upload_pending: bool = True) -> None:
        """Stop the heartbeat, optionally sync, release the lock & temp files.

        ``upload_pending=True`` is the default safe behaviour — the local
        DB is only deleted *after* the final upload returns success.
        """
        log.info("Closing Drive project session")
        self._stop_event.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)
            self._heartbeat_thread = None

        final_sync_failed = False
        if upload_pending and self._folders is not None and self._status.dirty:
            try:
                self.sync()
            except Exception as exc:  # noqa: BLE001
                final_sync_failed = True
                log.error(
                    "Final sync on close failed — local DB will be kept: %s", exc
                )

        try:
            self._release_lock()
        except Exception as exc:  # noqa: BLE001
            log.warning("Lock release failed: %s", exc)

        if self._local_db_path is not None:
            # Only delete when we are confident the data is safe on Drive.
            delete = (not final_sync_failed) and (not self._status.dirty)
            self._cache.release(self._local_db_path, delete=delete)
            if not delete:
                log.warning(
                    "Local DB %s NOT deleted (dirty / upload failed) — "
                    "will be cleaned up on next startup unless re-opened.",
                    self._local_db_path,
                )
            self._local_db_path = None

    # ------------------------------------------------------------------
    # Folder discovery
    # ------------------------------------------------------------------

    def _resolve_folders(self) -> ProjectFolders:
        try:
            root = self._client.get_metadata(self._root_id)
        except Exception as exc:  # noqa: BLE001
            raise ProjectFolderInvalid(
                f"Cannot access Drive folder {self._root_id}: {exc}"
            ) from exc
        if not root.is_folder:
            raise ProjectFolderInvalid(
                f"Drive object {self._root_id} is not a folder."
            )

        db_folder = self._ensure_child_folder(root.file_id, _DB_SUBFOLDER)
        meta_folder = self._ensure_child_folder(root.file_id, _METADATA_SUBFOLDER)

        db_file = self._client.find_child(db_folder.file_id, _DB_FILENAME)
        lock_file = self._client.find_child(db_folder.file_id, _LOCK_FILENAME)

        self._ensure_project_descriptor(meta_folder.file_id)

        return ProjectFolders(
            root_id=root.file_id,
            root_name=root.name,
            database_folder_id=db_folder.file_id,
            metadata_folder_id=meta_folder.file_id,
            db_file_id=db_file.file_id if db_file else None,
            lock_file_id=lock_file.file_id if lock_file else None,
        )

    def _ensure_child_folder(self, parent_id: str, name: str) -> DriveFile:
        existing = self._client.find_child(parent_id, name)
        if existing is not None and existing.is_folder:
            return existing
        if existing is not None and not existing.is_folder:
            raise ProjectFolderInvalid(
                f"Drive item {name!r} exists in project but is not a folder."
            )
        try:
            return self._client.create_folder(parent_id, name)
        except Exception as exc:  # noqa: BLE001
            # 403 = no write permission on the shared folder.
            try:
                from googleapiclient.errors import HttpError as _HttpError
                if isinstance(exc, _HttpError) and exc.resp.status in (403, 404):
                    raise ProjectFolderInvalid(
                        "Nincs szerkesztési jog a Google Drive mappához.\n\n"
                        "A Face-Local projekt megnyitásához szerkesztői hozzáférés szükséges "
                        f"(a '{name}' almappa létrehozásához).\n"
                        "Kérd a mappa tulajdonosától, hogy adjon szerkesztési jogot, "
                        "vagy válassz olyan mappát, amelyhez teljes hozzáférésed van."
                    ) from exc
            except ImportError:
                pass
            raise

    def _ensure_project_descriptor(self, metadata_folder_id: str) -> None:
        if self._client.find_child(metadata_folder_id, _PROJECT_DESCRIPTOR) is not None:
            return
        descriptor = {
            "app": "face-local",
            "version": __version__,
            "created_at": _utc_now_iso(),
            "schema_version": 1,
        }
        descriptor_path = self._cache.allocate_path("project.json")
        descriptor_path.write_text(
            json.dumps(descriptor, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        try:
            self._client.upload_file(
                str(descriptor_path),
                parent_folder_id=metadata_folder_id,
                remote_name=_PROJECT_DESCRIPTOR,
                mime_type="application/json",
            )
        finally:
            self._cache.release(descriptor_path, delete=True)

    # ------------------------------------------------------------------
    # Lock acquisition / release / heartbeat
    # ------------------------------------------------------------------

    def _acquire_lock(self, folders: ProjectFolders) -> None:
        if folders.lock_file_id is not None:
            # An existing lock — read it and decide.
            try:
                existing = self._download_lock(folders.lock_file_id)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "Could not read existing lock file (%s) — assuming stale.", exc
                )
                existing = None

            if existing is not None and not existing.is_stale() and not self._force_unlock:
                raise ProjectLocked(existing)
            if existing is not None and existing.is_stale():
                log.warning(
                    "Overriding stale lock from %s (last heartbeat %s)",
                    existing.device_name, existing.last_heartbeat,
                )

        info = LockInfo(
            lock_id=str(uuid.uuid4()),
            device_name=_device_name(),
            user_email=self._client.account.email,
            app_version=__version__,
            started_at=_utc_now_iso(),
            last_heartbeat=_utc_now_iso(),
        )
        new_id = self._upload_lock(folders, info)
        with self._lock:
            self._folders = ProjectFolders(
                **{**asdict(folders), "lock_file_id": new_id}
            )
            self._lock_info = info
        log.info("Lock acquired: id=%s device=%s", info.lock_id, info.device_name)

    def _upload_lock(self, folders: ProjectFolders, info: LockInfo) -> str:
        lock_local = self._cache.allocate_path(_LOCK_FILENAME)
        lock_local.write_text(info.to_json(), encoding="utf-8")
        try:
            return self._client.upload_file(
                str(lock_local),
                parent_folder_id=folders.database_folder_id,
                remote_name=_LOCK_FILENAME,
                existing_file_id=folders.lock_file_id,
                mime_type="application/json",
            )
        finally:
            self._cache.release(lock_local, delete=True)

    def _download_lock(self, lock_file_id: str) -> LockInfo:
        local = self._cache.allocate_path(f"{_LOCK_FILENAME}.peek")
        self._cache.mark_active(local)
        try:
            self._client.download_file(lock_file_id, str(local))
            return LockInfo.from_json(local.read_text(encoding="utf-8"))
        finally:
            self._cache.release(local, delete=True)

    def _refresh_heartbeat(self) -> None:
        with self._lock:
            info = self._lock_info
            folders = self._folders
        if info is None or folders is None or folders.lock_file_id is None:
            return
        new_info = LockInfo(
            lock_id=info.lock_id,
            device_name=info.device_name,
            user_email=info.user_email,
            app_version=info.app_version,
            started_at=info.started_at,
            last_heartbeat=_utc_now_iso(),
        )
        try:
            self._upload_lock(folders, new_info)
            with self._lock:
                self._lock_info = new_info
            log.debug("Lock heartbeat refreshed (%s)", new_info.last_heartbeat)
        except Exception as exc:  # noqa: BLE001
            log.warning("Lock heartbeat refresh failed: %s", exc)

    def _release_lock(self) -> None:
        with self._lock:
            folders = self._folders
            info = self._lock_info
        if folders is None or folders.lock_file_id is None:
            return
        # Only delete if this process owns the lock.
        if info is None:
            return
        try:
            current = self._download_lock(folders.lock_file_id)
            if current.lock_id != info.lock_id:
                log.warning(
                    "Lock owned by another lock_id (%s) — not releasing.",
                    current.lock_id,
                )
                return
        except Exception:  # noqa: BLE001
            log.debug("Could not read lock before release; deleting anyway.")
        try:
            self._client.delete_file(folders.lock_file_id)
            log.info("Lock released")
        finally:
            with self._lock:
                self._folders = ProjectFolders(
                    **{**asdict(folders), "lock_file_id": None}
                )
                self._lock_info = None

    def _start_heartbeat(self) -> None:
        if self._heartbeat_thread is not None:
            return
        self._stop_event.clear()

        def _run() -> None:
            log.debug("Heartbeat thread started (interval=%ds)", self._heartbeat_interval)
            while not self._stop_event.wait(self._heartbeat_interval):
                try:
                    self._refresh_heartbeat()
                except Exception as exc:  # noqa: BLE001
                    log.warning("Heartbeat tick failed: %s", exc)
            log.debug("Heartbeat thread exiting.")

        self._heartbeat_thread = threading.Thread(
            target=_run, name="gdrive-heartbeat", daemon=True
        )
        self._heartbeat_thread.start()

    # ------------------------------------------------------------------
    # Database download / init
    # ------------------------------------------------------------------

    def _download_or_init_db(self, folders: ProjectFolders) -> Path:
        local_db = self._cache.allocate_path(_DB_FILENAME)
        self._cache.mark_active(local_db)

        if folders.db_file_id is not None:
            log.info("Downloading existing faces.db from Drive …")
            try:
                self._client.download_file(folders.db_file_id, str(local_db))
            except Exception:
                self._cache.release(local_db, delete=True)
                raise
            return local_db

        # No DB on Drive yet — create empty file; the app's init_db() will
        # set up the schema, then sync() uploads it.
        log.info("No faces.db on Drive — initialising a fresh database.")
        try:
            local_db.parent.mkdir(parents=True, exist_ok=True)
            # Touch the file so init_db can open it.
            local_db.write_bytes(b"")
        except Exception:
            self._cache.release(local_db, delete=True)
            raise

        with self._lock:
            self._status.dirty = True  # first sync uploads the new schema
        return local_db

    def _checkpoint_sqlite(self, db_path: Path) -> None:
        """Run a WAL checkpoint so the uploaded .db file is fully flushed."""
        try:
            conn = sqlite3.connect(str(db_path))
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001
            log.warning("SQLite checkpoint failed for %s: %s", db_path, exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_op(self, label: Optional[str]) -> None:
        with self._lock:
            self._status.current_op = label
        if label:
            log.debug("Drive op: %s", label)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _device_name() -> str:
    host = socket.gethostname() or "unknown-host"
    sys = platform.system() or "?"
    return f"{host} ({sys})"


# Suppress unused-import warning — present for future use.
_ = field
_ = os
