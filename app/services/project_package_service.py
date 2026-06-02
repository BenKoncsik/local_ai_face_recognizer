"""Portable full-project export / import.

A ``.facepack`` file is a ZIP-compatible archive that bundles an entire
Face-Local project so it can be moved to another machine and restored:

    project.facepack
      manifest.json
      database/
        faces.db
      crops/
        img000001_face000001.jpg
        ...
      images/
        relative/path/to/original/image.jpg
      config/
        project.local.json

Design goals
------------
* The database is copied with the SQLite *backup API* (never a raw file copy)
  so a WAL-mode database is captured in a consistent state.
* Images are stored under portable, library-root-relative paths so the same
  archive restores on any machine.
* A missing source image never aborts the whole export — it is logged and
  reported in the result (and recorded in the manifest).
* Google-Drive-only images are exported only when a local/cached copy exists;
  otherwise they are reported as ``missing`` and flagged in the manifest.
* Import is hardened against ZIP path-traversal: any member with an absolute
  path or a ``..`` component is rejected and the archive is not extracted.

This module is intentionally self-contained and does not modify the existing
:class:`~app.services.export_service.ExportService`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

log = logging.getLogger(__name__)

# Public package metadata
PACKAGE_EXTENSION = ".facepack"
PACKAGE_FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"

# Archive layout (POSIX separators — ZIP convention)
ARCHIVE_DB_DIR = "database"
ARCHIVE_DB_NAME = "faces.db"
ARCHIVE_DB_PATH = f"{ARCHIVE_DB_DIR}/{ARCHIVE_DB_NAME}"
ARCHIVE_CROPS_DIR = "crops"
ARCHIVE_IMAGES_DIR = "images"
ARCHIVE_CONFIG_DIR = "config"
ARCHIVE_LOCAL_CONFIG = f"{ARCHIVE_CONFIG_DIR}/project.local.json"

# Images that are not under the library root land here, keyed by image id, so
# they remain restorable even without a configured library root.
_EXTERNAL_IMAGE_PREFIX = "_external"

ProgressCb = Callable[[int, int, str], None]


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------

@dataclass
class ExportResult:
    """Summary of an :meth:`ProjectPackageService.export_package` run."""

    package_path: Path
    image_count: int = 0
    crop_count: int = 0
    missing_images: List[str] = field(default_factory=list)
    drive_only_images: List[str] = field(default_factory=list)

    @property
    def has_warnings(self) -> bool:
        return bool(self.missing_images or self.drive_only_images)


@dataclass
class ValidationResult:
    """Outcome of a structural validation of a ``.facepack`` file."""

    ok: bool
    manifest: Optional[dict] = None
    errors: List[str] = field(default_factory=list)


@dataclass
class ImportResult:
    """Summary of an :meth:`ProjectPackageService.import_package` run."""

    project_dir: Path
    db_path: Path
    images_dir: Path
    crops_dir: Path
    manifest: dict = field(default_factory=dict)
    remapped_crops: int = 0


class ProjectPackageError(Exception):
    """Raised on unrecoverable export/import failures."""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class ProjectPackageService:
    """Build and restore portable ``.facepack`` project archives.

    Args:
        db_path:       Path to the live SQLite database file.
        crops_dir:     Directory holding face-crop thumbnails.
        library_root:  Configured image-library root (used to compute portable
                       relative paths for images that lack ``relative_path``).
                       May be ``None`` when no root is configured.
        local_config_path: Path to ``project.local.json`` (stored in the
                       archive's ``config/`` folder for reference).
    """

    def __init__(
        self,
        db_path: Path | str,
        crops_dir: Path | str,
        library_root: Optional[Path | str] = None,
        local_config_path: Optional[Path | str] = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._crops_dir = Path(crops_dir)
        self._library_root = Path(library_root) if library_root else None
        self._local_config_path = (
            Path(local_config_path) if local_config_path else None
        )

    # ==================================================================
    # Export
    # ==================================================================

    def export_package(
        self,
        target_path: Path | str,
        *,
        app_version: str = "",
        session=None,
        progress_cb: Optional[ProgressCb] = None,
    ) -> ExportResult:
        """Write a ``.facepack`` archive to *target_path*.

        Args:
            target_path:  Destination file.  ``.facepack`` is appended when the
                          caller passes a path without that extension.
            app_version:  Human-readable application version recorded in the
                          manifest.
            session:      Optional SQLAlchemy session for image enumeration.
                          When ``None`` a transient session is opened.
            progress_cb:  Optional ``(current, total, message)`` callback.

        Returns:
            An :class:`ExportResult`.
        """
        target = Path(target_path)
        if target.suffix.lower() != PACKAGE_EXTENSION:
            target = target.with_name(target.name + PACKAGE_EXTENSION)
        target.parent.mkdir(parents=True, exist_ok=True)

        if not self._db_path.exists():
            raise ProjectPackageError(f"Database not found: {self._db_path}")

        if session is not None:
            return self._export_with_session(
                target, session, app_version, progress_cb
            )
        from app.db.database import session_scope

        with session_scope() as scoped:
            return self._export_with_session(
                target, scoped, app_version, progress_cb
            )

    def _export_with_session(
        self,
        target: Path,
        session,
        app_version: str,
        progress_cb: Optional[ProgressCb],
    ) -> ExportResult:
        from app.db.models import Image

        result = ExportResult(package_path=target)
        images = session.query(Image).order_by(Image.id).all()
        crop_files = self._collect_crop_files()
        total = len(images) + len(crop_files) + 2  # +db +manifest

        # Write to a temporary file first so a crash never leaves a half-written
        # archive in place of an existing valid one.
        tmp_fd, tmp_name = tempfile.mkstemp(
            suffix=PACKAGE_EXTENSION, dir=str(target.parent)
        )
        os.close(tmp_fd)
        tmp_path = Path(tmp_name)

        try:
            with zipfile.ZipFile(
                tmp_path, "w", compression=zipfile.ZIP_DEFLATED
            ) as zf:
                step = 0

                # --- database (consistent SQLite backup) ---
                step += 1
                _emit(progress_cb, step, total, "Adatbázis mentése…")
                db_sha, db_size = self._add_database(zf)

                # --- crops ---
                for crop in crop_files:
                    step += 1
                    _emit(progress_cb, step, total, f"Kivágás: {crop.name}")
                    arc = f"{ARCHIVE_CROPS_DIR}/{crop.name}"
                    zf.write(crop, arc)
                    result.crop_count += 1

                # --- images ---
                image_manifest: List[dict] = []
                for image in images:
                    step += 1
                    entry = self._add_image(zf, image, result)
                    _emit(
                        progress_cb, step, total,
                        f"Kép: {Path(entry['archive_path'] or entry['source']).name}",
                    )
                    image_manifest.append(entry)

                # --- local config (informational) ---
                self._add_local_config(zf)

                # --- manifest ---
                step += 1
                _emit(progress_cb, step, total, "Manifest írása…")
                manifest = self._build_manifest(
                    app_version=app_version,
                    db_sha=db_sha,
                    db_size=db_size,
                    images=image_manifest,
                    result=result,
                )
                zf.writestr(
                    MANIFEST_NAME,
                    json.dumps(manifest, indent=2, ensure_ascii=False),
                )

            # Atomically move the finished archive into place.
            os.replace(tmp_path, target)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        log.info(
            "Project package written: %s (%d images, %d crops, %d missing)",
            target, result.image_count, result.crop_count, len(result.missing_images),
        )
        return result

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def _collect_crop_files(self) -> List[Path]:
        if not self._crops_dir.is_dir():
            return []
        return sorted(
            p for p in self._crops_dir.iterdir()
            if p.is_file() and not p.name.startswith(".")
        )

    def _add_database(self, zf: zipfile.ZipFile) -> tuple[str, int]:
        """Back up the SQLite database and add it to *zf*.

        Uses the SQLite online-backup API so a WAL-mode database is captured
        consistently without copying stray ``-wal`` / ``-shm`` files.
        """
        tmp_fd, tmp_name = tempfile.mkstemp(suffix=".db")
        os.close(tmp_fd)
        tmp_db = Path(tmp_name)
        try:
            src = sqlite3.connect(str(self._db_path))
            try:
                dst = sqlite3.connect(str(tmp_db))
                try:
                    src.backup(dst)
                    dst.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                finally:
                    dst.close()
            finally:
                src.close()

            sha, size = _file_digest(tmp_db)
            zf.write(tmp_db, ARCHIVE_DB_PATH)
            return sha, size
        finally:
            tmp_db.unlink(missing_ok=True)

    def _archive_image_path(self, image) -> str:
        """Return the portable archive path for *image* (POSIX separators)."""
        rel = getattr(image, "relative_path", None)
        if rel:
            return f"{ARCHIVE_IMAGES_DIR}/{_normalize_posix(rel)}"

        # No stored relative path — try to derive one from the library root.
        file_path = getattr(image, "file_path", None)
        if file_path and self._library_root is not None:
            try:
                rel2 = Path(file_path).resolve().relative_to(
                    self._library_root.resolve()
                )
                return f"{ARCHIVE_IMAGES_DIR}/{rel2.as_posix()}"
            except (ValueError, OSError):
                pass

        # Fall back to an id-keyed external bucket so the file is still bundled.
        name = Path(file_path).name if file_path else f"image_{image.id}"
        return (
            f"{ARCHIVE_IMAGES_DIR}/{_EXTERNAL_IMAGE_PREFIX}/"
            f"{image.id}/{name}"
        )

    def _resolve_image_source(self, image) -> Optional[Path]:
        """Resolve the on-disk source path for *image*, if any."""
        try:
            from app.services.image_library_service import resolve_image_path

            resolved = resolve_image_path(image)
            if resolved is not None:
                return resolved
        except Exception:  # noqa: BLE001 — never let resolution abort export
            pass
        fp = getattr(image, "file_path", None)
        return Path(fp) if fp else None

    def _add_image(self, zf: zipfile.ZipFile, image, result: ExportResult) -> dict:
        archive_path = self._archive_image_path(image)
        source = self._resolve_image_source(image)
        entry: dict = {
            "id": image.id,
            "source": str(source) if source else "",
            "archive_path": archive_path,
            "relative_path": getattr(image, "relative_path", None),
            "present": False,
            "size": None,
        }

        if source is None or not source.exists():
            # Drive-only / cache-miss images have a relative path but no local
            # file; everything else is simply missing.
            label = str(source) if source else f"image_{image.id}"
            if getattr(image, "relative_path", None):
                result.drive_only_images.append(label)
            result.missing_images.append(label)
            entry["archive_path"] = None
            log.warning("Export: image source missing, skipping: %s", label)
            return entry

        try:
            zf.write(source, archive_path)
            entry["present"] = True
            entry["size"] = source.stat().st_size
            result.image_count += 1
        except OSError as exc:
            result.missing_images.append(str(source))
            entry["archive_path"] = None
            log.warning("Export: failed to add image %s: %s", source, exc)
        return entry

    def _add_local_config(self, zf: zipfile.ZipFile) -> None:
        if self._local_config_path and self._local_config_path.exists():
            try:
                zf.write(self._local_config_path, ARCHIVE_LOCAL_CONFIG)
                return
            except OSError as exc:
                log.warning("Export: could not add local config: %s", exc)
        # Always include a config file so the archive layout is predictable.
        payload = {
            "image_library_root": str(self._library_root)
            if self._library_root else None,
        }
        zf.writestr(
            ARCHIVE_LOCAL_CONFIG, json.dumps(payload, indent=2, ensure_ascii=False)
        )

    def _build_manifest(
        self,
        *,
        app_version: str,
        db_sha: str,
        db_size: int,
        images: List[dict],
        result: ExportResult,
    ) -> dict:
        return {
            "package_format_version": PACKAGE_FORMAT_VERSION,
            "app_version": app_version,
            "exported_at": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
            "database": {
                "archive_path": ARCHIVE_DB_PATH,
                "filename": ARCHIVE_DB_NAME,
                "sha256": db_sha,
                "size": db_size,
            },
            "crops": {
                "archive_dir": ARCHIVE_CROPS_DIR,
                "count": result.crop_count,
            },
            "images": {
                "archive_dir": ARCHIVE_IMAGES_DIR,
                "count": result.image_count,
                "missing_count": len(result.missing_images),
                "drive_only_count": len(result.drive_only_images),
                "entries": images,
            },
            "config": {
                "archive_path": ARCHIVE_LOCAL_CONFIG,
            },
            "source_library_root": str(self._library_root)
            if self._library_root else None,
        }

    # ==================================================================
    # Validation
    # ==================================================================

    @staticmethod
    def validate_package(package_path: Path | str) -> ValidationResult:
        """Structurally validate a ``.facepack`` file.

        Checks the extension, that the file is a readable ZIP, that a
        ``manifest.json`` exists and parses, that the declared database member
        is present, and that the mandatory archive folders exist.
        """
        path = Path(package_path)
        errors: List[str] = []

        if path.suffix.lower() != PACKAGE_EXTENSION:
            errors.append(
                f"Érvénytelen kiterjesztés: {path.suffix!r} "
                f"(várt: {PACKAGE_EXTENSION})"
            )
            return ValidationResult(ok=False, errors=errors)

        if not path.is_file():
            errors.append(f"A fájl nem található: {path}")
            return ValidationResult(ok=False, errors=errors)

        if not zipfile.is_zipfile(path):
            errors.append("A fájl nem olvasható ZIP-archívumként.")
            return ValidationResult(ok=False, errors=errors)

        try:
            with zipfile.ZipFile(path, "r") as zf:
                names = set(zf.namelist())

                if MANIFEST_NAME not in names:
                    errors.append("Hiányzik a manifest.json.")
                    return ValidationResult(ok=False, errors=errors)

                try:
                    manifest = json.loads(zf.read(MANIFEST_NAME).decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    errors.append(f"A manifest.json nem értelmezhető: {exc}")
                    return ValidationResult(ok=False, errors=errors)

                db_member = (
                    manifest.get("database", {}).get("archive_path")
                    or ARCHIVE_DB_PATH
                )
                if db_member not in names:
                    errors.append(f"Hiányzik az adatbázis: {db_member}")

                has_crops = any(n.startswith(ARCHIVE_CROPS_DIR + "/") for n in names)
                has_images = any(
                    n.startswith(ARCHIVE_IMAGES_DIR + "/") for n in names
                )
                if not has_crops:
                    errors.append("Hiányzik a crops/ mappa.")
                if not has_images:
                    errors.append("Hiányzik az images/ mappa.")

                # Path-traversal scan — refuse to even open a hostile archive.
                for name in names:
                    if _is_unsafe_member(name):
                        errors.append(f"Nem biztonságos útvonal az archívumban: {name}")
                        break

        except zipfile.BadZipFile as exc:
            errors.append(f"Sérült ZIP-archívum: {exc}")
            return ValidationResult(ok=False, errors=errors)

        if errors:
            return ValidationResult(ok=False, manifest=None, errors=errors)
        return ValidationResult(ok=True, manifest=manifest, errors=[])

    # ==================================================================
    # Import
    # ==================================================================

    @staticmethod
    def import_package(
        package_path: Path | str,
        dest_dir: Path | str,
        *,
        progress_cb: Optional[ProgressCb] = None,
    ) -> ImportResult:
        """Extract a ``.facepack`` into a fresh project directory.

        The archive is validated first; extraction is path-traversal safe.
        After extraction the local image-library root is rewritten to point at
        the freshly extracted ``images/`` folder so the project is immediately
        usable on this machine.

        Args:
            package_path:  Source ``.facepack`` file.
            dest_dir:      Target project directory (created if absent; must be
                           empty or not yet exist to avoid clobbering data).
            progress_cb:   Optional ``(current, total, message)`` callback.

        Returns:
            An :class:`ImportResult`.
        """
        package_path = Path(package_path)
        dest = Path(dest_dir)

        validation = ProjectPackageService.validate_package(package_path)
        if not validation.ok:
            raise ProjectPackageError(
                "Érvénytelen .facepack: " + "; ".join(validation.errors)
            )

        if dest.exists() and any(dest.iterdir()):
            raise ProjectPackageError(
                f"A célmappa nem üres: {dest}. Válassz üres vagy új mappát."
            )
        dest.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(package_path, "r") as zf:
            members = zf.infolist()
            total = len(members)
            for idx, member in enumerate(members, start=1):
                if _is_unsafe_member(member.filename):
                    raise ProjectPackageError(
                        f"Path traversal kísérlet az archívumban: {member.filename}"
                    )
                _emit(progress_cb, idx, total, f"Kibontás: {member.filename}")
                _safe_extract_member(zf, member, dest)

        db_path = dest / ARCHIVE_DB_DIR / ARCHIVE_DB_NAME
        images_dir = dest / ARCHIVE_IMAGES_DIR
        crops_dir = dest / ARCHIVE_CROPS_DIR
        if not db_path.exists():
            raise ProjectPackageError(
                f"A kibontott archívumból hiányzik az adatbázis: {db_path}"
            )
        images_dir.mkdir(exist_ok=True)
        crops_dir.mkdir(exist_ok=True)

        manifest = validation.manifest or {}

        # Rewrite the machine-local config so the library root points at the
        # extracted images directory (ignoring any stale absolute root that the
        # archive's config/project.local.json may have carried over).
        _write_local_config(db_path.parent / "project.local.json", images_dir)

        log.info("Project package imported into %s", dest)
        return ImportResult(
            project_dir=dest,
            db_path=db_path,
            images_dir=images_dir,
            crops_dir=crops_dir,
            manifest=manifest,
        )

    @staticmethod
    def remap_crop_paths(session, crops_dir: Path | str) -> int:
        """Repoint every ``Face.crop_path`` at the imported crops directory.

        Crop files are bundled by their canonical basename
        (``imgNNNNNN_faceNNNNNN.jpg``); after import the database still holds the
        *original machine's* absolute crop paths.  This rewrites each path by
        basename to the new location so previews resolve without a re-crop.

        Returns the number of rewritten rows.
        """
        from app.db.models import Face

        crops_dir = Path(crops_dir)
        remapped = 0
        faces = session.query(Face).filter(Face.crop_path.isnot(None)).all()
        for face in faces:
            name = Path(face.crop_path).name
            candidate = crops_dir / name
            if candidate.exists():
                new_path = str(candidate)
                if face.crop_path != new_path:
                    face.crop_path = new_path
                    remapped += 1
        if remapped:
            session.flush()
            log.info("Remapped %d crop path(s) to %s", remapped, crops_dir)
        return remapped


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _emit(cb: Optional[ProgressCb], current: int, total: int, message: str) -> None:
    if cb is not None:
        try:
            cb(current, total, message)
        except Exception:  # noqa: BLE001 — progress must never break the job
            log.debug("progress callback raised", exc_info=True)


def _normalize_posix(rel: str) -> str:
    """Normalize a stored relative path to forward slashes, no leading slash."""
    parts = [p for p in rel.replace("\\", "/").split("/") if p and p != "."]
    return "/".join(parts)


def _is_unsafe_member(name: str) -> bool:
    """Return ``True`` if a ZIP member name is absolute or escapes the root."""
    if not name:
        return False
    normalized = name.replace("\\", "/")
    if normalized.startswith("/"):
        return True
    # Windows drive-letter absolute path (e.g. ``C:/...``)
    if len(normalized) >= 2 and normalized[1] == ":":
        return True
    parts = normalized.split("/")
    return any(part == ".." for part in parts)


def _safe_extract_member(
    zf: zipfile.ZipFile, member: zipfile.ZipInfo, dest: Path
) -> None:
    """Extract a single member into *dest*, guaranteeing it stays inside it."""
    if member.is_dir():
        (dest / member.filename).mkdir(parents=True, exist_ok=True)
        return

    target = (dest / member.filename).resolve()
    dest_resolved = dest.resolve()
    if not _is_within(target, dest_resolved):
        raise ProjectPackageError(
            f"Path traversal blokkolva: {member.filename}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    # Stream the member so large images never load fully into memory.
    with zf.open(member, "r") as src, open(target, "wb") as out:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _file_digest(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            h.update(chunk)
    return h.hexdigest(), size


def _write_local_config(config_path: Path, images_dir: Path) -> None:
    data: dict = {}
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            data = {}
    data["image_library_root"] = str(images_dir.resolve())
    config_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
