"""Drive-aware image scan service.

Downloads image files from a Google Drive project folder to a persistent
local mirror directory, then records them in the database just like the
local :class:`~app.services.scan_service.ScanService` does.

On the first scan every image is downloaded.  On subsequent scans the
MD5 checksum reported by Drive is compared against the stored value in
:class:`~app.db.models.RemoteImage`; unchanged files are skipped.

Folder layout on Drive::

    <project root>/
      database/        ← skipped (contains DB + lock)
      metadata/        ← skipped (contains project.json)
      <album 1>/
        photo.jpg
        ...
      photo2.jpg       ← direct children are included too

Local mirror layout::

    <mirror_dir>/
      <drive_file_id>.jpg   ← named by Drive file_id to avoid collisions
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Set

from sqlalchemy.orm import Session

from app.config import ScanConfig
from app.db.models import Image, RemoteImage
from app.gdrive.protocols import DriveFile, GDriveClient

log = logging.getLogger(__name__)

# Subfolders at the project root level that are reserved for app use.
_SKIP_FOLDER_NAMES: Set[str] = {"database", "metadata"}

# Google Drive MIME type for folders.
_FOLDER_MIME = "application/vnd.google-apps.folder"


class DriveScanService:
    """Scan a Google Drive project folder and populate the local database.

    Images are downloaded to *local_mirror_dir* (a persistent directory that
    survives between sessions) with the Drive ``file_id`` as the filename stem
    so that name collisions across sub-albums are impossible.

    Args:
        session:          SQLAlchemy session for the Drive project's database.
        client:           Authenticated Drive client.
        root_folder_id:   Drive file_id of the project root folder.
        local_mirror_dir: Persistent local directory that receives downloaded
                          images.  Created automatically if absent.
        config:           Scan configuration (extensions, etc.).
        progress_cb:      Optional ``(current, total, drive_file_name)`` callback.
    """

    def __init__(
        self,
        session: Session,
        client: GDriveClient,
        root_folder_id: str,
        local_mirror_dir: Path,
        config: ScanConfig,
        progress_cb: Optional[Callable[[int, Optional[int], str], None]] = None,
    ) -> None:
        self._session = session
        self._client = client
        self._root_id = root_folder_id
        self._mirror = local_mirror_dir
        self._config = config
        self._progress_cb = progress_cb or (lambda *_: None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self) -> List[int]:
        """List Drive images, download new/changed ones, return IDs for detection.

        Returns:
            List of :class:`~app.db.models.Image` primary keys that are new
            or have changed content since the last scan.
        """
        self._mirror.mkdir(parents=True, exist_ok=True)
        ext_set = {e.lower() for e in self._config.image_extensions}

        log.info(
            "DriveScan: listing images in Drive folder %s …", self._root_id
        )
        drive_files = self._list_images_recursive(
            self._root_id, skip_names=_SKIP_FOLDER_NAMES, ext_set=ext_set
        )
        total = len(drive_files)
        log.info("DriveScan: found %d image file(s) on Drive", total)

        new_or_changed: List[int] = []
        for idx, df in enumerate(drive_files, start=1):
            self._progress_cb(idx, total, df.name)
            try:
                image_id = self._index_drive_file(df)
                if image_id is not None:
                    new_or_changed.append(image_id)
            except Exception as exc:  # noqa: BLE001
                log.warning("DriveScan: error indexing %s: %s", df.name, exc)

        self._session.commit()
        log.info(
            "DriveScan: %d new/changed file(s) out of %d total",
            len(new_or_changed), total,
        )
        return new_or_changed

    # ------------------------------------------------------------------
    # Drive traversal
    # ------------------------------------------------------------------

    def _list_images_recursive(
        self,
        folder_id: str,
        *,
        skip_names: Set[str],
        ext_set: Set[str],
    ) -> List[DriveFile]:
        """Recursively collect image files from *folder_id*, skipping *skip_names*."""
        results: List[DriveFile] = []
        children = self._client.list_folder(folder_id, include_subfolders=True)
        for child in children:
            if child.is_folder:
                if child.name in skip_names:
                    log.debug("DriveScan: skipping reserved folder %r", child.name)
                    continue
                # Recurse — subfolders of subfolders don't need to be skipped.
                results.extend(
                    self._list_images_recursive(
                        child.file_id, skip_names=set(), ext_set=ext_set
                    )
                )
            else:
                if Path(child.name).suffix.lower() in ext_set:
                    results.append(child)
        return results

    # ------------------------------------------------------------------
    # Per-file indexing
    # ------------------------------------------------------------------

    def _index_drive_file(self, df: DriveFile) -> Optional[int]:
        """Download (if needed) and upsert *df* into the database.

        Returns the Image primary key when the record is new or changed,
        ``None`` when the file is unchanged.
        """
        # Derive a stable local filename from the Drive file_id.
        suffix = Path(df.name).suffix.lower()
        local_name = f"{df.file_id}{suffix}"
        local_path = self._mirror / local_name

        # ------------------------------------------------------------------
        # 1. Look up existing RemoteImage record by Drive file_id.
        # ------------------------------------------------------------------
        remote_rec: Optional[RemoteImage] = (
            self._session.query(RemoteImage)
            .filter(RemoteImage.drive_file_id == df.file_id)
            .first()
        )

        now = datetime.now(timezone.utc).replace(tzinfo=None)

        if remote_rec is not None:
            # File already known — check if it changed on Drive.
            if self._is_unchanged(df, remote_rec):
                log.debug("DriveScan: unchanged, skipping %s", df.name)
                # Update last_seen_at so we know it still exists.
                remote_rec.last_seen_at = now
                remote_rec.deleted_remote = False
                return None
            # Changed on Drive — re-download.
            log.debug("DriveScan: changed on Drive, re-downloading %s", df.name)
            self._download(df, local_path)
            file_hash = _hash_file(local_path)
            image = self._session.get(Image, remote_rec.image_id)
            if image is not None:
                image.file_hash = file_hash
                image.file_mtime = local_path.stat().st_mtime
                image.detection_done = False
                image.embedding_done = False
            remote_rec.modified_time = df.modified_time
            remote_rec.checksum = df.md5_checksum
            remote_rec.last_seen_at = now
            remote_rec.deleted_remote = False
            return remote_rec.image_id if image else None

        # ------------------------------------------------------------------
        # 2. New Drive file — download and create Image + RemoteImage.
        # ------------------------------------------------------------------
        log.debug("DriveScan: new file, downloading %s", df.name)
        self._download(df, local_path)
        file_hash = _hash_file(local_path)
        mtime = local_path.stat().st_mtime

        image = Image(
            file_path=str(local_path),
            file_hash=file_hash,
            file_mtime=mtime,
        )
        self._session.add(image)
        self._session.flush()  # populate image.id

        remote = RemoteImage(
            image_id=image.id,
            provider="google_drive",
            drive_file_id=df.file_id,
            drive_folder_id=df.parents[0] if df.parents else None,
            remote_name=df.name,
            modified_time=df.modified_time,
            checksum=df.md5_checksum,
            last_seen_at=now,
            deleted_remote=False,
        )
        self._session.add(remote)
        self._session.flush()

        log.debug(
            "DriveScan: indexed new file %s → local %s (image.id=%d)",
            df.name, local_name, image.id,
        )
        return image.id

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_unchanged(self, df: DriveFile, rec: RemoteImage) -> bool:
        """Return True when the Drive file is identical to the stored record.

        Uses MD5 checksum when available; falls back to modifiedTime string.
        """
        if df.md5_checksum and rec.checksum:
            return df.md5_checksum == rec.checksum
        if df.modified_time and rec.modified_time:
            return df.modified_time == rec.modified_time
        # No metadata to compare — assume changed to be safe.
        return False

    def _download(self, df: DriveFile, local_path: Path) -> None:
        """Download *df* from Drive to *local_path*, creating parent dirs."""
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(df.file_id, str(local_path))


# ---------------------------------------------------------------------------
# File hash helper
# ---------------------------------------------------------------------------

_HASH_CHUNK = 4 * 1024 * 1024  # 4 MB


def _hash_file(path: Path) -> str:
    """SHA-256 hex digest of *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()
