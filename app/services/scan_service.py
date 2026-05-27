"""Image discovery and indexing service.

Responsibilities
----------------
* Recursively enumerate image files under a root folder.
* Compute a SHA-256 hash for each file.
* Insert new records into the ``images`` table.
* Skip files whose path + hash + mtime haven't changed (resume support).
* Return a list of :class:`~app.db.models.Image` IDs ready for detection.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Callable, Generator, List, Optional

from sqlalchemy.orm import Session

from app.config import ScanConfig
from app.db.models import Image
from app.services.image_library_service import ImageLibraryService
from app.services.place_service import PlaceService
from app.utils.exif import read_exif_gps

log = logging.getLogger(__name__)

# How many bytes to hash at a time (4 MB chunks)
_HASH_CHUNK = 4 * 1024 * 1024


# ---------------------------------------------------------------------------
# File hash
# ---------------------------------------------------------------------------

def hash_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of *path*.

    Reads in 4 MB chunks to stay memory-efficient on large images.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_images(
    root: Path,
    extensions: List[str],
) -> Generator[Path, None, None]:
    """Yield image files under *root* recursively.

    Args:
        root:       Directory to scan.
        extensions: Lower-case extensions to include (e.g. ``[".jpg"]``).

    Yields:
        Absolute :class:`Path` objects for each matching file.
    """
    ext_set = {e.lower() for e in extensions}
    for dirpath, _, filenames in os.walk(root):
        for filename in sorted(filenames):
            if Path(filename).suffix.lower() in ext_set:
                yield Path(dirpath) / filename


# ---------------------------------------------------------------------------
# Scan service
# ---------------------------------------------------------------------------

class ScanService:
    """Indexes image files into the database, skipping unchanged files.

    Args:
        session:    SQLAlchemy session.
        config:     Scan configuration block.
        progress_cb: Optional callable ``(current, total, path)`` called for
                     each file processed.  ``total`` may be ``None`` if
                     the file count is unknown.
    """

    def __init__(
        self,
        session: Session,
        config: ScanConfig,
        progress_cb: Optional[Callable[[int, Optional[int], str], None]] = None,
        image_library_svc: Optional[ImageLibraryService] = None,
    ) -> None:
        self._session = session
        self._config = config
        self._progress_cb = progress_cb or (lambda *_: None)
        self._library_svc = image_library_svc

    def scan(self, root_folder: str) -> List[int]:
        """Scan *root_folder* and return IDs of images that need processing.

        "Need processing" means either new or changed since last scan.

        Args:
            root_folder: Path to the root image directory.

        Returns:
            List of :class:`~app.db.models.Image` primary keys.
        """
        root = Path(root_folder)
        if not root.exists():
            raise FileNotFoundError(f"Root folder not found: {root}")

        log.info("Scanning folder: %s", root)

        # Collect all candidate paths first so we can report progress
        paths = list(discover_images(root, self._config.image_extensions))
        total = len(paths)
        log.info("Found %d candidate image file(s)", total)

        new_or_changed: List[int] = []

        for idx, path in enumerate(paths, start=1):
            self._progress_cb(idx, total, str(path))

            try:
                image_id = self._index_file(path)
                if image_id is not None:
                    new_or_changed.append(image_id)
            except Exception as exc:  # noqa: BLE001
                log.warning("Error indexing %s: %s", path, exc)

        self._session.commit()
        log.info(
            "Scan complete: %d new/changed file(s) out of %d total",
            len(new_or_changed), total,
        )
        return new_or_changed

    def _index_file(self, path: Path) -> Optional[int]:
        """Insert or update the DB record for *path*.

        When an :class:`~app.services.image_library_service.ImageLibraryService`
        is configured, the method:
        * Stores a portable ``relative_path`` on every new record.
        * Recognises images by ``relative_path`` when ``file_path`` no longer
          matches (e.g. after moving the project to a new machine).
        * Back-fills ``relative_path`` on existing records that lack it.

        Returns:
            The image ID if the record is new or changed; ``None`` if
            the file is unchanged and already fully processed.
        """
        try:
            mtime = path.stat().st_mtime
        except OSError as exc:
            log.warning("Cannot stat %s: %s", path, exc)
            return None

        # Compute portable relative path if library service is available.
        relative_path: Optional[str] = None
        if self._library_svc is not None:
            relative_path = self._library_svc.make_relative(path)

        # Primary lookup: absolute file_path (fast, covers same-machine case).
        existing: Optional[Image] = (
            self._session.query(Image)
            .filter(Image.file_path == str(path))
            .first()
        )

        # Secondary lookup: relative_path (cross-machine portability).
        # This finds an existing record even when file_path has changed.
        if existing is None and relative_path is not None:
            existing = (
                self._session.query(Image)
                .filter(Image.relative_path == relative_path)
                .first()
            )
            if existing is not None:
                # Update the stored absolute path to the current machine's path.
                log.info(
                    "Re-linked image by relative path: %s → %s",
                    existing.file_path,
                    path,
                )
                existing.file_path = str(path)

        if existing is not None:
            # Back-fill relative_path on older records that lack it.
            if existing.relative_path is None and relative_path is not None:
                existing.relative_path = relative_path
                log.debug("Back-filled relative_path for: %s", path.name)

            # Quick check: if mtime matches, skip hashing.
            if existing.file_mtime == mtime and existing.detection_done:
                log.debug("Skipping unchanged file: %s", path.name)
                return None

        # Hash is relatively expensive — only compute when mtime changed.
        file_hash = hash_file(path)

        if existing is not None:
            if existing.file_hash == file_hash and existing.detection_done:
                # Update mtime in DB (file touched but content unchanged).
                existing.file_mtime = mtime
                log.debug("Content unchanged (mtime updated): %s", path.name)
                return None

            # Content changed — reset processing flags.
            existing.file_hash = file_hash
            existing.file_mtime = mtime
            existing.detection_done = False
            existing.embedding_done = False
            self._apply_exif_gps(existing, path)
            log.debug("Content changed — requeued: %s", path.name)
            return existing.id
        else:
            # New file.
            image = Image(
                file_path=str(path),
                file_hash=file_hash,
                file_mtime=mtime,
                relative_path=relative_path,
            )
            self._session.add(image)
            self._session.flush()  # populate image.id before returning
            self._apply_exif_gps(image, path)
            log.debug("Indexed new file: %s (id=%d)", path.name, image.id)
            return image.id

    def _apply_exif_gps(self, image: Image, path: Path) -> None:
        """Read EXIF GPS and link an anonymous/nearby Place without aborting scan."""
        try:
            gps = read_exif_gps(path)
            if gps is None:
                log.debug("No EXIF GPS data for: %s", path.name)
                return
            latitude, longitude = gps
            if image.place_id is None:
                place = PlaceService(self._session).link_exif_gps_to_image(
                    image.id,
                    latitude,
                    longitude,
                )
                log.info(
                    "EXIF GPS linked image %d to place %d (%s): %.6f, %.6f",
                    image.id,
                    place.id,
                    place.name,
                    latitude,
                    longitude,
                )
            else:
                image.exif_latitude = latitude
                image.exif_longitude = longitude
        except Exception as exc:  # noqa: BLE001
            log.info("EXIF GPS processing skipped for %s: %s", path, exc)
