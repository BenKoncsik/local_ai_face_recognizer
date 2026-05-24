"""Portable image library path management.

The ``ImageLibraryService`` decouples absolute machine-specific file paths
from the portable relative paths stored in the database.  It solves the
cross-machine portability problem: the same database can be opened on any
machine as long as the user points the service at the local copy of the
image folder.

Architecture
------------
* ``Image.relative_path``  – POSIX-style path relative to the library root,
  stored in the database and portable across machines.
* ``Image.file_path``       – The original absolute path recorded at index
  time on the original machine; kept for backward compatibility with old
  databases that have no relative_path.
* ``project.local.json``   – A machine-specific JSON file that lives next to
  the database and stores the local library root path.  It is NOT part of
  the portable project and should be excluded from version control.

Path resolution priority (``resolve_path``)
-------------------------------------------
1. ``relative_path`` + library root  → portable, cross-machine
2. ``file_path`` as absolute          → legacy / same-machine fallback
3. ``None``                           → broken / unknown

Module-level singleton
----------------------
Follow the same pattern as ``app.db.database``:

    init_image_library(db_path)        – call once after init_db()
    get_image_library()                – raises if not initialised
    get_image_library_optional()       – returns None if not initialised
    resolve_image_path(image)          – convenience resolver
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional

if TYPE_CHECKING:
    from app.db.models import Image

log = logging.getLogger(__name__)

LOCAL_CONFIG_FILENAME = "project.local.json"
_LIBRARY_ROOT_KEY = "image_library_root"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_service: Optional["ImageLibraryService"] = None


def init_image_library(db_path: Path | str) -> "ImageLibraryService":
    """Create (or recreate) the global :class:`ImageLibraryService`.

    Call once after :func:`app.db.database.init_db`.
    """
    global _service
    _service = ImageLibraryService(db_path)
    log.debug("ImageLibraryService initialised for db: %s", db_path)
    return _service


def get_image_library() -> "ImageLibraryService":
    """Return the global service; raises if not yet initialised."""
    if _service is None:
        raise RuntimeError(
            "Image library not initialised — call init_image_library() first."
        )
    return _service


def get_image_library_optional() -> Optional["ImageLibraryService"]:
    """Return the global service, or ``None`` if not yet initialised."""
    return _service


def resolve_image_path(image: "Image") -> Optional[Path]:
    """Return the absolute :class:`~pathlib.Path` for *image*.

    Uses the global library service when available; falls back to
    ``image.file_path``.  Returns ``None`` only when neither source
    can produce a path.
    """
    if _service is not None:
        return _service.resolve_path(image)
    if getattr(image, "file_path", None):
        return Path(image.file_path)
    return None


# ---------------------------------------------------------------------------
# Migration result
# ---------------------------------------------------------------------------

@dataclass
class MigrationResult:
    """Summary of a ``migrate_to_relative_paths`` run."""

    migrated: int = 0
    skipped: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def total_processed(self) -> int:
        return self.migrated + self.skipped + len(self.errors)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class ImageLibraryService:
    """Manages the image library root and portable path resolution.

    Args:
        db_path: Path to the SQLite database file.  The local config file
                 (``project.local.json``) is stored in the same directory.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path).resolve()
        self._local_config_path = self._db_path.parent / LOCAL_CONFIG_FILENAME
        self._library_root: Optional[Path] = None
        self._config_loaded: bool = False

    # ------------------------------------------------------------------
    # Library root access
    # ------------------------------------------------------------------

    @property
    def library_root(self) -> Optional[Path]:
        """Return the configured library root, or ``None`` if not set."""
        if not self._config_loaded:
            self._library_root = self._load_library_root()
            self._config_loaded = True
        return self._library_root

    def is_configured(self) -> bool:
        """True if a library root has been set (regardless of existence)."""
        return self.library_root is not None

    def is_available(self) -> bool:
        """True if the library root is set *and* exists on disk."""
        root = self.library_root
        return root is not None and root.is_dir()

    def set_library_root(self, root: Path | str) -> None:
        """Set and persist a new library root.

        Saves to ``project.local.json`` and updates the in-memory cache.

        Args:
            root: Absolute path to the image library directory.

        Raises:
            NotADirectoryError: If *root* does not exist as a directory.
        """
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Library root is not a directory: {root}")
        self._persist_library_root(root)
        self._library_root = root
        self._config_loaded = True
        log.info("Image library root set: %s", root)

    def local_config_path(self) -> Path:
        """Return the path to ``project.local.json``."""
        return self._local_config_path

    # ------------------------------------------------------------------
    # Path conversion
    # ------------------------------------------------------------------

    def make_relative(self, absolute_path: Path | str) -> Optional[str]:
        """Compute the POSIX-style relative path for *absolute_path*.

        Returns ``None`` when no library root is configured or the path
        is not under the library root.

        The returned value always uses forward slashes and is suitable for
        cross-platform storage in the database.
        """
        root = self.library_root
        if root is None:
            return None
        try:
            resolved = Path(absolute_path).resolve()
            rel = resolved.relative_to(root)
            return rel.as_posix()  # Forward slashes for cross-platform portability
        except ValueError:
            return None  # Path is not under the library root

    def resolve_path(self, image: "Image") -> Optional[Path]:
        """Return the absolute path for *image* on the current machine.

        Resolution order:
        1. ``relative_path`` + library root  (portable)
        2. ``file_path`` as-is               (legacy / same-machine)
        3. ``None``

        The returned :class:`~pathlib.Path` may not exist on disk if the
        file has been moved or an external drive is disconnected.
        """
        rel = getattr(image, "relative_path", None)
        if rel:
            root = self.library_root
            if root is not None:
                return root / _from_posix(rel)
            # Root not configured — do NOT silently fall back to file_path when
            # relative_path exists; the caller should detect the missing root.
            log.debug(
                "Image %s has relative_path but library root is not configured",
                getattr(image, "id", "?"),
            )

        fp = getattr(image, "file_path", None)
        if fp:
            return Path(fp)

        return None

    def is_path_under_root(self, absolute_path: Path | str) -> bool:
        """Return ``True`` if *absolute_path* is inside the library root."""
        root = self.library_root
        if root is None:
            return False
        try:
            Path(absolute_path).resolve().relative_to(root)
            return True
        except ValueError:
            return False

    # ------------------------------------------------------------------
    # Auto-detection
    # ------------------------------------------------------------------

    def detect_common_root(
        self, session: "Session"  # noqa: F821
    ) -> Optional[str]:
        """Heuristically detect a common path prefix from existing records.

        Useful for pre-filling the library root during initial setup.
        Only considers images that do *not* yet have a ``relative_path``.

        Returns the detected root path as a string, or ``None``.
        """
        from app.db.models import Image

        rows = (
            session.query(Image.file_path)
            .filter(
                Image.file_path.isnot(None),
                Image.relative_path.is_(None),
            )
            .limit(500)
            .all()
        )
        paths = [Path(r[0]) for r in rows if r[0]]
        if not paths:
            return None

        # Find the common ancestor of all parent directories.
        parents = [p.parent for p in paths]
        common = parents[0]
        for p in parents[1:]:
            while True:
                try:
                    p.relative_to(common)
                    break
                except ValueError:
                    parent = common.parent
                    if parent == common:
                        return None
                    common = parent
        return str(common)

    # ------------------------------------------------------------------
    # Migration
    # ------------------------------------------------------------------

    def migrate_to_relative_paths(
        self,
        session: "Session",  # noqa: F821
        library_root: Path | str,
        validate_files: bool = True,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> MigrationResult:
        """Convert ``file_path`` → ``relative_path`` for existing records.

        This is a non-destructive operation: ``file_path`` is kept as-is.
        Only the ``relative_path`` column is written.

        After migrating, the library root is saved to ``project.local.json``
        and the in-memory cache is updated.

        Args:
            session:      Active SQLAlchemy session.
            library_root: Root directory of the image library.
            validate_files: When ``True``, skip records whose resolved path
                            does not exist on disk and report them as errors.
            progress_cb:  Optional ``(current, total, path)`` callback.

        Returns:
            A :class:`MigrationResult` with counts and error list.
        """
        from app.db.models import Image

        library_root = Path(library_root).expanduser().resolve()
        result = MigrationResult()

        images = (
            session.query(Image)
            .filter(Image.relative_path.is_(None))
            .order_by(Image.id)
            .all()
        )
        total = len(images)

        for idx, image in enumerate(images):
            if progress_cb:
                progress_cb(idx + 1, total, image.file_path or "")

            if not image.file_path:
                result.skipped += 1
                continue

            try:
                abs_path = Path(image.file_path)
                try:
                    rel = abs_path.relative_to(library_root)
                except ValueError:
                    log.warning(
                        "Path not under library root, skipping: %s", image.file_path
                    )
                    result.skipped += 1
                    continue

                if validate_files:
                    resolved = library_root / rel
                    if not resolved.exists():
                        log.warning("File not found during migration: %s", resolved)
                        result.errors.append(str(abs_path))
                        result.skipped += 1
                        continue

                image.relative_path = rel.as_posix()
                result.migrated += 1

            except Exception as exc:  # noqa: BLE001
                log.error("Migration error for %s: %s", image.file_path, exc)
                result.errors.append(f"{image.file_path}: {exc}")
                result.skipped += 1

        # Always persist the provided root, even when nothing was migrated,
        # so the user's choice is saved for future scans.
        self._persist_library_root(library_root)
        self._library_root = library_root
        self._config_loaded = True

        log.info(
            "Path migration complete: %d migrated, %d skipped, %d errors",
            result.migrated,
            result.skipped,
            len(result.errors),
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_library_root(self) -> Optional[Path]:
        if not self._local_config_path.exists():
            return None
        try:
            data = json.loads(
                self._local_config_path.read_text(encoding="utf-8")
            )
            raw = data.get(_LIBRARY_ROOT_KEY)
            if raw:
                return Path(raw).expanduser()
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Could not read local config %s: %s", self._local_config_path, exc
            )
        return None

    def _persist_library_root(self, root: Path) -> None:
        data: dict = {}
        if self._local_config_path.exists():
            try:
                data = json.loads(
                    self._local_config_path.read_text(encoding="utf-8")
                )
            except Exception:  # noqa: BLE001
                data = {}
        data[_LIBRARY_ROOT_KEY] = str(root)
        self._local_config_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.debug("Saved library root to %s", self._local_config_path)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _from_posix(posix_path: str) -> Path:
    """Convert a stored POSIX-style relative path to a native :class:`~pathlib.Path`.

    Handles both ``/`` and ``\\`` separators to be resilient against paths
    that were accidentally stored with Windows separators.
    """
    normalized = posix_path.replace("\\", "/")
    parts = [p for p in normalized.split("/") if p]
    if not parts:
        return Path(".")
    return Path(*parts)
