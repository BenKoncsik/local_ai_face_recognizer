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
from pathlib import Path, PureWindowsPath
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

# The ZIP format cannot represent timestamps before 1980-01-01.  Files older
# than that (or with unreadable mtimes) get this fixed date in the archive so
# the export never aborts.  The real mtime is preserved in the manifest.
ZIP_MIN_DATE_TIME = (1980, 1, 1, 0, 0, 0)

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
    # Images that could not be bundled (missing / unreadable / foreign path).
    missing_images: List[str] = field(default_factory=list)
    # Same set, exposed under the name the UI/callers consume for "skipped".
    skipped_images: List[str] = field(default_factory=list)
    # Subset with a relative path but no local file (Google-Drive-only / cache miss).
    drive_only_images: List[str] = field(default_factory=list)

    @property
    def warning_count(self) -> int:
        """Number of images that triggered a warning (were not bundled)."""
        return len(self.skipped_images)

    @property
    def has_warnings(self) -> bool:
        return bool(self.skipped_images or self.missing_images)


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
    # Archive members that could not be extracted on this machine (e.g. a name
    # the target filesystem rejects even after sanitisation).  Reported to the
    # user so a missing original is never silently swallowed.
    skipped_members: List[str] = field(default_factory=list)


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
                    try:
                        _write_file_to_zip_safe(zf, crop, arc)
                        result.crop_count += 1
                    except OSError as exc:
                        log.warning("Export: failed to add crop %s: %s", crop, exc)

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
            _write_file_to_zip_safe(zf, tmp_db, ARCHIVE_DB_PATH)
            return sha, size
        finally:
            tmp_db.unlink(missing_ok=True)

    def _archive_image_path(
        self, image, resolved_source: Optional[Path] = None
    ) -> str:
        """Return the portable archive path for *image* (POSIX separators)."""
        rel = getattr(image, "relative_path", None)
        if rel:
            return f"{ARCHIVE_IMAGES_DIR}/{_normalize_posix(rel)}"

        # No stored relative path — try to derive one from the library root.
        # Check both the stored file_path and the actually resolved source so
        # a stale file_path doesn't block us from computing a portable path.
        file_path = getattr(image, "file_path", None)
        if self._library_root is not None:
            lib_resolved = self._library_root.resolve()
            for candidate in filter(None, [
                file_path,
                str(resolved_source) if resolved_source else None,
            ]):
                try:
                    rel2 = Path(candidate).resolve().relative_to(lib_resolved)
                    return f"{ARCHIVE_IMAGES_DIR}/{rel2.as_posix()}"
                except (ValueError, OSError):
                    pass

        # Fall back: preserve the original directory structure by stripping
        # the drive letter / leading slash from the absolute path.  This keeps
        # images that lived in the same folder together in the archive instead
        # of scattering each one into its own numbered sub-folder.
        path_str = file_path or (str(resolved_source) if resolved_source else None)
        if path_str:
            arc_rel = _abs_path_to_archive_rel(path_str)
            if arc_rel:
                return f"{ARCHIVE_IMAGES_DIR}/{_EXTERNAL_IMAGE_PREFIX}/{arc_rel}"

        # Last resort: id-keyed bucket (guarantees uniqueness when the path
        # cannot be sanitised to a usable relative form).
        name = _safe_component(_basename_any(file_path)) if file_path else f"image_{image.id}"
        return f"{ARCHIVE_IMAGES_DIR}/{_EXTERNAL_IMAGE_PREFIX}/{image.id}/{name}"

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
        # Resolve the source first so the path computation can use it as a
        # fallback when image.file_path is stale or missing.
        source = self._resolve_image_source(image)
        archive_path = self._archive_image_path(image, source)
        src_str = str(source) if source else ""
        entry: dict = {
            "id": image.id,
            "source": src_str,
            "archive_path": archive_path,
            "relative_path": getattr(image, "relative_path", None),
            "present": False,
            "size": None,
            "mtime": None,
            "mtime_epoch": None,
            "skipped_reason": None,
        }

        # A Windows-only absolute path (``D:\…`` / ``\\server\share\…``) cannot
        # exist on macOS/Linux; checking it with ``Path.exists()`` would raise
        # ``OSError: [Errno 63] File name too long`` because it is treated as one
        # huge local file name.  Detect and skip it *without* touching the
        # filesystem.
        foreign = bool(src_str) and _is_foreign_windows_path(src_str)
        available = (
            source is not None
            and not foreign
            and _safe_file_exists(source)
        )

        if not available:
            label = src_str or f"image_{image.id}"
            reason = "windows_path_on_non_windows" if foreign else "unavailable"
            self._record_skipped_image(result, image, label, reason, foreign)
            entry["archive_path"] = None
            entry["skipped_reason"] = reason
            log.warning(
                "Export: skipping unavailable original image (%s): %s",
                reason, label,
            )
            return entry

        try:
            entry["mtime"] = _iso_mtime(source)
            try:
                entry["mtime_epoch"] = source.stat().st_mtime
            except OSError:
                entry["mtime_epoch"] = None
            _write_file_to_zip_safe(zf, source, archive_path)
            entry["present"] = True
            entry["size"] = source.stat().st_size
            result.image_count += 1
        except OSError as exc:
            self._record_skipped_image(result, image, src_str, "io_error", False)
            entry["archive_path"] = None
            entry["skipped_reason"] = "io_error"
            log.warning("Export: failed to add image %s: %s", source, exc)
        return entry

    @staticmethod
    def _record_skipped_image(
        result: ExportResult,
        image,
        label: str,
        reason: str,
        foreign: bool,
    ) -> None:
        """Record one un-exportable image into the result's warning lists."""
        result.missing_images.append(label)
        result.skipped_images.append(label)
        # Images that have a portable relative path but no local file are the
        # classic Google-Drive-only / cache-miss case.  A foreign Windows path
        # is a different problem, so it is *not* flagged as drive-only.
        if getattr(image, "relative_path", None) and not foreign:
            result.drive_only_images.append(label)

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
                "skipped_count": len(result.skipped_images),
                "drive_only_count": len(result.drive_only_images),
                "skipped": list(result.skipped_images),
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

        skipped: List[str] = []
        with zipfile.ZipFile(package_path, "r") as zf:
            members = zf.infolist()
            total = len(members)
            for idx, member in enumerate(members, start=1):
                if _is_unsafe_member(member.filename):
                    # Security: a traversal / absolute-path member always aborts.
                    raise ProjectPackageError(
                        f"Path traversal kísérlet az archívumban: {member.filename}"
                    )
                _emit(progress_cb, idx, total, f"Kibontás: {member.filename}")
                try:
                    _safe_extract_member(zf, member, dest)
                except OSError as exc:
                    # A single un-writable file (e.g. a name too long for the
                    # target filesystem) must not abort the whole import; log it
                    # and carry on.  Mandatory members are re-checked below.
                    skipped.append(member.filename)
                    log.warning(
                        "Import: skipped unextractable member %r: %s",
                        member.filename, exc,
                    )

        if skipped:
            log.warning("Import: %d member(s) could not be extracted", len(skipped))

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

        # Restore each image's original modification time from the manifest.
        # The ZIP entry's own date was clamped to 1980 for pre-1980 files, but
        # the real timestamp was preserved in the manifest so it survives the
        # round-trip.
        restored = _restore_image_mtimes(dest, manifest)
        if restored:
            log.info("Restored original mtime for %d image(s)", restored)

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
            skipped_members=skipped,
        )

    @staticmethod
    def remap_image_paths(
        session, manifest: dict, images_dir: Path | str
    ) -> int:
        """Repoint every :class:`Image` at its bundled copy under *images_dir*.

        The archive bundles each image under ``images/<archive-relative>`` (see
        :meth:`_archive_image_path`).  After import the database still holds the
        *original machine's* absolute ``file_path``; legacy records additionally
        have no ``relative_path`` at all, so :meth:`ImageLibraryService.resolve_path`
        would fall back to that stale absolute path and fail to find the file.

        Two columns are rewritten so the project is fully usable on the new
        machine, keeping the original folder structure under the extracted
        ``images/`` directory:

        * ``relative_path`` → the path the image was bundled under (relative to
          ``images/``), so :meth:`ImageLibraryService.resolve_path` resolves it
          against the new library root regardless of the original layout.
        * ``file_path``     → the new *absolute* location on this machine.  Many
          callers read ``image.file_path`` directly (previews, the image
          browser, exports…); without this they would keep pointing at the
          original machine's path and report the image as missing.  This mirrors
          how :meth:`remap_crop_paths` rewrites ``Face.crop_path``.

        Images that were not bundled (missing / foreign path at export time) are
        left untouched.

        Returns the number of rewritten rows.
        """
        from app.db.models import Image, Place, PlaceAlias

        images_dir = Path(images_dir)
        entries = (manifest.get("images") or {}).get("entries") or []
        remapped = 0
        # Build old → new file_path mapping so Place/PlaceAlias thumbnail_path
        # (which stores image.file_path verbatim) can also be rewritten in one pass.
        old_to_new: dict[str, str] = {}
        # Disable autoflush so the per-row writes are committed in one batch at
        # the end: ``file_path`` is UNIQUE, and a mid-loop flush could transiently
        # collide a freshly-rewritten path with an as-yet-unremapped original.
        with session.no_autoflush:
            for entry in entries:
                archive_path = entry.get("archive_path")
                image_id = entry.get("id")
                if not archive_path or image_id is None:
                    # Not bundled (missing/foreign at export) — keep existing paths.
                    continue
                # Extraction sanitises every path component (Windows-illegal
                # chars, trailing dots, long paths…); mirror that here so the
                # rewritten paths point at the file that actually landed on disk.
                rel = _archive_path_to_image_rel(
                    _sanitize_archive_path(archive_path)
                )
                if rel is None:
                    continue
                on_disk = images_dir / _from_archive_rel(rel)
                # Only repoint when the bundled file actually landed on disk, so a
                # member that failed to extract never hides behind a broken path.
                if not _os_exists(on_disk):
                    continue
                image = session.get(Image, image_id)
                if image is None:
                    continue
                new_file_path = str(on_disk.resolve())
                changed = False
                if image.relative_path != rel:
                    image.relative_path = rel
                    changed = True
                if image.file_path != new_file_path:
                    old_to_new[image.file_path] = new_file_path
                    image.file_path = new_file_path
                    changed = True
                if changed:
                    remapped += 1

            # Place.thumbnail_path and PlaceAlias.thumbnail_path store a raw
            # image.file_path value; remap them to match the new on-disk location.
            if old_to_new:
                for place in session.query(Place).filter(
                    Place.thumbnail_path.isnot(None)
                ).all():
                    new = old_to_new.get(place.thumbnail_path)
                    if new:
                        place.thumbnail_path = new
                        remapped += 1
                for alias in session.query(PlaceAlias).filter(
                    PlaceAlias.thumbnail_path.isnot(None)
                ).all():
                    new = old_to_new.get(alias.thumbnail_path)
                    if new:
                        alias.thumbnail_path = new
                        remapped += 1

        if remapped:
            session.flush()
            log.info("Remapped %d image/thumbnail path(s) to %s", remapped, images_dir)
        return remapped

    @staticmethod
    def remap_crop_paths(session, crops_dir: Path | str) -> int:
        """Repoint every crop/thumbnail path at the imported crops directory.

        Crop files are bundled by their canonical basename
        (``imgNNNNNN_faceNNNNNN.jpg``); after import the database still holds the
        *original machine's* absolute crop paths.  This rewrites each path by
        basename to the new location so previews resolve without a re-crop.

        Covers: ``Face.crop_path``, ``Person.thumbnail_path``,
        ``IgnoredFace.thumbnail_path`` — all of which store absolute paths to
        files in the crops directory.

        Returns the number of rewritten rows.
        """
        from app.db.models import Face, IgnoredFace, Person

        crops_dir = Path(crops_dir)
        remapped = 0

        def _remap_path(old_path: str) -> Optional[str]:
            candidate = crops_dir / Path(old_path).name
            if candidate.exists():
                new = str(candidate)
                return new if new != old_path else None
            return None

        for face in session.query(Face).filter(Face.crop_path.isnot(None)).all():
            new = _remap_path(face.crop_path)
            if new:
                face.crop_path = new
                remapped += 1

        for person in session.query(Person).filter(Person.thumbnail_path.isnot(None)).all():
            new = _remap_path(person.thumbnail_path)
            if new:
                person.thumbnail_path = new
                remapped += 1

        for iface in session.query(IgnoredFace).filter(IgnoredFace.thumbnail_path.isnot(None)).all():
            new = _remap_path(iface.thumbnail_path)
            if new:
                iface.thumbnail_path = new
                remapped += 1

        if remapped:
            session.flush()
            log.info("Remapped %d crop/thumbnail path(s) to %s", remapped, crops_dir)
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


def _abs_path_to_archive_rel(path_str: str) -> str:
    """Strip the drive / absolute root from *path_str* → sanitised POSIX rel.

    Preserves the full directory structure so images that were originally in the
    same folder stay together in the archive's ``_external/`` bucket rather than
    each landing in its own numbered sub-folder.

    ``/Users/alice/Photos/vacation/img.jpg``  →  ``Users/alice/Photos/vacation/img.jpg``
    ``D:\\Photos\\vacation\\img.jpg``          →  ``Photos/vacation/img.jpg``
    """
    norm = path_str.replace("\\", "/")
    # Strip Windows drive letter (e.g. "C:/")
    if len(norm) >= 2 and norm[1] == ":":
        norm = norm[2:]
    norm = norm.lstrip("/")
    parts = [p for p in norm.split("/") if p]
    if not parts:
        return ""
    safe = [_sanitize_component(p) for p in parts]
    return "/".join(p for p in safe if p)


def _archive_path_to_image_rel(archive_path: str) -> Optional[str]:
    """Strip the leading ``images/`` from an archive path → library-relative path.

    Returns the portion of *archive_path* below :data:`ARCHIVE_IMAGES_DIR`
    (e.g. ``images/sub/a.jpg`` → ``sub/a.jpg``), or ``None`` when the path is
    not under the images directory.
    """
    norm = _normalize_posix(archive_path)
    prefix = ARCHIVE_IMAGES_DIR + "/"
    if not norm.startswith(prefix):
        return None
    rel = norm[len(prefix):]
    return rel or None


# Most filesystems cap a single path component at 255 bytes; stay well under it
# to leave room for any future suffixing and avoid edge cases.
_MAX_COMPONENT_BYTES = 200


def _safe_file_exists(path: Path | str) -> bool:
    """Return whether *path* exists, swallowing filesystem errors.

    ``Path.exists()`` can raise ``OSError`` (e.g. ``[Errno 63] File name too
    long`` when a Windows absolute path is mis-handled as one giant local file
    name on macOS/Linux).  Such failures must never abort the export, so they
    are logged and treated as "does not exist".
    """
    try:
        return Path(path).exists()
    except OSError as exc:
        log.warning("File existence check failed for %r: %s", str(path), exc)
        return False


def _is_foreign_windows_path(path_str: str) -> bool:
    """True if *path_str* is a Windows-only absolute path on a non-Windows host.

    Recognises drive-letter paths (``C:\\…`` / ``C:/…``) and UNC paths
    (``\\\\server\\share\\…``) via :class:`pathlib.PureWindowsPath`.  Always
    ``False`` on Windows, where such paths are genuinely local and checkable.
    """
    if os.name == "nt" or not path_str:
        return False
    try:
        pw = PureWindowsPath(path_str)
    except (TypeError, ValueError):
        return False
    # A drive letter or UNC root makes it an absolute Windows-only location.
    return bool(pw.drive) and pw.is_absolute()


def _basename_any(path_str: str) -> str:
    """Return the file name from *path_str*, treating ``/`` and ``\\`` as separators.

    ``pathlib.Path`` only honours the *host* OS separator, so a Windows path
    such as ``D:\\dir\\file.jpg`` keeps its backslashes (and stays one giant
    component) on POSIX.  This splits on both so the bare file name is always
    recovered, regardless of which OS produced the path.
    """
    normalized = path_str.replace("\\", "/")
    name = normalized.rsplit("/", 1)[-1].strip()
    return name or "file"


def _safe_component(name: str, max_bytes: int = _MAX_COMPONENT_BYTES) -> str:
    """Clamp a single file-name component to *max_bytes* UTF-8 bytes.

    The extension is preserved and multi-byte characters are never split.  Used
    only for id-keyed external image names, where the surrounding ``<id>/``
    folder already guarantees uniqueness, so truncation cannot collide.
    """
    name = name.strip() or "file"
    if len(name.encode("utf-8")) <= max_bytes:
        return name

    stem, dot, ext = name.rpartition(".")
    if not dot or len(ext) > 16:  # no usable extension
        stem, ext = name, ""
    ext_suffix = ("." + ext) if ext else ""
    budget = max(max_bytes - len(ext_suffix.encode("utf-8")), 1)
    truncated = stem.encode("utf-8")[:budget].decode("utf-8", "ignore").strip()
    return (truncated or "file") + ext_suffix


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


# Characters that are illegal in a Windows path component, plus control chars.
_WINDOWS_ILLEGAL_CHARS = set('<>:"/\\|?*') | {chr(c) for c in range(0, 32)}
# Reserved DOS device names (case-insensitive, with or without an extension).
_WINDOWS_RESERVED_NAMES = (
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


def _sanitize_component(comp: str) -> str:
    """Make a single path component safe on every supported filesystem.

    Replaces characters Windows rejects with ``_``, strips trailing dots and
    spaces (also forbidden on Windows), avoids reserved device names, and clamps
    over-long names.  Ordinary photo names (letters, digits, spaces, ``-_.`` and
    unicode) pass through unchanged, so same-OS round-trips are unaffected and
    only pathological cross-OS names are rewritten.
    """
    if not comp or comp in (".", ".."):
        return comp
    out = "".join("_" if ch in _WINDOWS_ILLEGAL_CHARS else ch for ch in comp)
    out = out.rstrip(" .")
    if not out:
        out = "_"
    stem = out.split(".", 1)[0].upper()
    if stem in _WINDOWS_RESERVED_NAMES:
        out = "_" + out
    return _safe_component(out)


def _sanitize_archive_path(name: str) -> str:
    """Sanitise every component of an archive member path (POSIX separators).

    Honours both ``/`` and ``\\`` as separators so a path written by either OS
    is split correctly.  Returns a forward-slash relative path with each
    component made filesystem-safe.
    """
    norm = name.replace("\\", "/")
    parts = [_sanitize_component(p) for p in norm.split("/") if p and p != "."]
    return "/".join(p for p in parts if p)


def _from_archive_rel(rel: str) -> Path:
    """Turn a sanitised POSIX archive-relative path into a native ``Path``."""
    parts = [p for p in rel.split("/") if p]
    return Path(*parts) if parts else Path(".")


def _os_path(path: Path) -> str:
    r"""Return a string path that bypasses the Windows ``MAX_PATH`` (260) limit.

    On Windows an absolute target is prefixed with the extended-length marker
    (``\\?\`` for local paths, ``\\?\UNC\`` for UNC shares) so deeply nested
    ``images/`` trees still open.  A no-op on other platforms.
    """
    if os.name != "nt":
        return str(path)
    s = str(path)
    if s.startswith("\\\\?\\"):
        return s
    if s.startswith("\\\\"):  # UNC path: \\server\share\...
        return "\\\\?\\UNC\\" + s[2:]
    return "\\\\?\\" + s


def _os_mkdir(path: Path) -> None:
    """``mkdir -p`` that tolerates the Windows long-path prefix."""
    os.makedirs(_os_path(path), exist_ok=True)


def _os_exists(path: Path) -> bool:
    """Existence check that survives the Windows long-path limit, never raises."""
    try:
        return os.path.exists(_os_path(path))
    except OSError as exc:
        log.warning("File existence check failed for %r: %s", str(path), exc)
        return False


def _safe_extract_member(
    zf: zipfile.ZipFile, member: zipfile.ZipInfo, dest: Path
) -> None:
    """Extract a single member into *dest*, guaranteeing it stays inside it.

    The archive path is sanitised per component (see
    :func:`_sanitize_archive_path`) so a name produced on one OS — e.g. a macOS
    photo path containing characters Windows forbids (``: ? * " | < >``), a
    trailing dot/space, or a reserved device name — does not make extraction
    fail on the importing machine.  On Windows the on-disk target is opened
    through an extended-length (``\\\\?\\``) path so a deep ``images/`` tree
    that exceeds the legacy 260-character ``MAX_PATH`` limit still extracts.
    :func:`remap_image_paths` applies the exact same sanitiser, so the rewritten
    ``relative_path`` always matches the file that actually landed on disk.
    """
    safe_rel = _sanitize_archive_path(member.filename)
    if member.is_dir():
        if safe_rel:
            _os_mkdir(dest / _from_archive_rel(safe_rel))
        return

    target = (dest / _from_archive_rel(safe_rel)).resolve()
    dest_resolved = dest.resolve()
    if not _is_within(target, dest_resolved):
        raise ProjectPackageError(
            f"Path traversal blokkolva: {member.filename}"
        )
    _os_mkdir(target.parent)
    # Stream the member so large images never load fully into memory.
    with zf.open(member, "r") as src, open(_os_path(target), "wb") as out:
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


def _safe_zip_date_time(source_path: Path) -> tuple[tuple[int, int, int, int, int, int], bool]:
    """Return a ZIP-compatible ``date_time`` tuple for *source_path*.

    The ZIP format cannot encode timestamps before 1980.  When the file's
    modification time is older than that — or cannot be read at all — the date
    is clamped to :data:`ZIP_MIN_DATE_TIME` and the second tuple element is
    ``True`` to signal the caller that a warning should be logged.
    """
    try:
        dt = datetime.fromtimestamp(source_path.stat().st_mtime)
        if dt.year < 1980:
            return ZIP_MIN_DATE_TIME, True
        return (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second), False
    except (OSError, OverflowError, ValueError):
        # Negative / out-of-range timestamps raise on some platforms (notably
        # Windows); fall back to the minimum supported date.
        return ZIP_MIN_DATE_TIME, True


def _restore_image_mtimes(dest: Path, manifest: dict) -> int:
    """Re-apply each image's original modification time after extraction.

    Uses ``mtime_epoch`` (a raw POSIX timestamp) when present — it round-trips
    pre-1980 / pre-1970 dates without timezone or parsing pitfalls — and falls
    back to parsing the ISO ``mtime`` string.  Best effort: a single failure
    never aborts the import.

    Returns the number of files whose timestamp was restored.
    """
    entries = (manifest.get("images") or {}).get("entries") or []
    restored = 0
    for entry in entries:
        archive_path = entry.get("archive_path")
        if not archive_path:
            continue
        epoch = _entry_mtime_epoch(entry)
        if epoch is None:
            continue
        target = dest / _from_archive_rel(_sanitize_archive_path(archive_path))
        if not _os_exists(target):
            continue
        try:
            os.utime(_os_path(target), (epoch, epoch))
            restored += 1
        except (OSError, OverflowError, ValueError):
            # e.g. Windows cannot set negative (pre-1970) times — skip quietly.
            log.debug("Could not restore mtime for %s", target, exc_info=True)
    return restored


def _entry_mtime_epoch(entry: dict) -> Optional[float]:
    """Resolve a POSIX timestamp from a manifest image entry, if recorded."""
    epoch = entry.get("mtime_epoch")
    if isinstance(epoch, (int, float)):
        return float(epoch)
    iso = entry.get("mtime")
    if isinstance(iso, str) and iso:
        try:
            return datetime.fromisoformat(iso).timestamp()
        except (ValueError, OverflowError, OSError):
            return None
    return None


def _iso_mtime(source_path: Path) -> Optional[str]:
    """Return the file's modification time as an ISO-8601 string, or ``None``."""
    try:
        return datetime.fromtimestamp(source_path.stat().st_mtime).isoformat()
    except (OSError, OverflowError, ValueError):
        return None


def _write_file_to_zip_safe(
    zf: zipfile.ZipFile, source_path: Path | str, archive_path: str
) -> bool:
    """Stream *source_path* into *zf* with a ZIP-safe timestamp.

    A :class:`zipfile.ZipInfo` is built by hand so a pre-1980 (or invalid)
    modification time can be clamped to :data:`ZIP_MIN_DATE_TIME` instead of
    raising ``ValueError: ZIP does not support timestamps before 1980``.  The
    file content is copied in chunks so large images never load fully into
    memory, and the archive stays a valid, ZIP-compatible ``.facepack``.

    Returns ``True`` when the timestamp had to be clamped.
    """
    source_path = Path(source_path)
    st = source_path.stat()
    date_time, clamped = _safe_zip_date_time(source_path)

    zinfo = zipfile.ZipInfo(filename=archive_path, date_time=date_time)
    zinfo.compress_type = zipfile.ZIP_DEFLATED
    # Preserve the unix permission bits in the high word of external_attr.
    zinfo.external_attr = (st.st_mode & 0xFFFF) << 16

    with source_path.open("rb") as src, zf.open(zinfo, "w") as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)

    if clamped:
        log.warning(
            "ZIP timestamp clamped to 1980 for %s (original mtime unsupported)",
            source_path,
        )
    return clamped


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
