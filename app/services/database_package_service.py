"""Database-only project export / import.

A ``.facedb`` file is a small ZIP archive that contains **only** the SQLite
database (no images, no face crops)::

    project.facedb
      manifest.json
      database/
        faces.db

Why a database-only package
---------------------------
Two machines often hold the *same* photo tree under a *different* base folder::

    machine A:  C:\\local\\kepek\\1998\\nyar\\img_001.jpg
    machine B:  D:\\kepek\\1998\\nyar\\img_001.jpg

Everything below the library root (``1998/nyar/img_001.jpg``) is identical, so
the absolute prefix is the only machine-specific part — and it does not need to
travel with the package.  Export therefore records each image by its
**library-root-relative path** and stores the source root only as information;
import re-bases every path onto the *target* machine's own root.

Face crops are intentionally left out: they are derived data and the app's
startup crop-repair pass regenerates them from the originals
(:func:`app.services.face_crop_service.ensure_unique_face_crops`).  Crop
references that do not exist locally are cleared on import so that pass picks
them up.

When an image cannot be found at the re-based location (the folder layout below
the root differs, files were renamed/moved), the record is reported as
*unresolved* — :mod:`app.services.image_path_matcher` can then scan a folder and
match those records by file name / content hash.

Import is hardened against ZIP path-traversal: any member with an absolute path
or a ``..`` component is rejected and the archive is not extracted.
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
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Callable, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# Public package metadata
PACKAGE_EXTENSION = ".facedb"
PACKAGE_FORMAT_VERSION = 1
PACKAGE_KIND = "database"
MANIFEST_NAME = "manifest.json"

# Archive layout (POSIX separators — ZIP convention)
ARCHIVE_DB_DIR = "database"
ARCHIVE_DB_NAME = "faces.db"
ARCHIVE_DB_PATH = f"{ARCHIVE_DB_DIR}/{ARCHIVE_DB_NAME}"

#: Local config file written next to the imported database.
LOCAL_CONFIG_NAME = "project.local.json"

#: ``(table, column)`` pairs holding an absolute path to a *face crop* file.
#: Only the file name survives an import; the rest is machine-specific.
_CROP_PATH_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("faces", "crop_path"),
    ("persons", "thumbnail_path"),
    ("ignored_faces", "thumbnail_path"),
    ("tagged_objects", "thumbnail_path"),
    ("merge_decisions", "candidate_crop_path"),
    ("merge_decisions", "target_crop_path"),
)

#: ``(table, column)`` pairs holding a copy of an ``images.file_path`` value.
_IMAGE_PATH_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("places", "thumbnail_path"),
    ("place_aliases", "thumbnail_path"),
)

ProgressCb = Callable[[int, int, str], None]


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------

@dataclass
class DbExportResult:
    """Summary of a :meth:`DatabasePackageService.export_database` run."""

    package_path: Path
    image_count: int = 0
    #: Images that carry a library-root-relative path (fully portable).
    portable_count: int = 0
    #: Images stored outside the library root — only their absolute path is known.
    outside_root_count: int = 0
    library_root: Optional[str] = None
    size_bytes: int = 0

    @property
    def has_warnings(self) -> bool:
        return self.outside_root_count > 0


@dataclass
class DbValidationResult:
    """Outcome of a structural validation of a ``.facedb`` file."""

    ok: bool
    manifest: Optional[dict] = None
    errors: List[str] = field(default_factory=list)


@dataclass
class DbImportResult:
    """Summary of a :meth:`DatabasePackageService.import_database` run."""

    project_dir: Path
    db_path: Path
    crops_dir: Path
    library_root: Path
    manifest: dict = field(default_factory=dict)
    #: Image rows whose path was re-based onto the target library root.
    rebased: int = 0
    #: Re-based rows whose file actually exists on this machine.
    resolved: int = 0
    #: Rows whose file could not be found (candidates for the path matcher).
    unresolved: int = 0
    #: Crop references cleared because the file is not present locally.
    cleared_crops: int = 0

    @property
    def needs_matching(self) -> bool:
        """True when some originals were not found at the re-based location."""
        return self.unresolved > 0


class DatabasePackageError(Exception):
    """Raised on unrecoverable database export/import failures."""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class DatabasePackageService:
    """Build and restore database-only ``.facedb`` packages.

    Args:
        db_path:      Path to the live SQLite database file.
        library_root: Configured image-library root on *this* machine.  Used to
                      derive the portable relative path of images that do not
                      have one yet.  May be ``None``.
    """

    def __init__(
        self,
        db_path: Path | str,
        library_root: Optional[Path | str] = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._library_root = Path(library_root) if library_root else None

    # ==================================================================
    # Export
    # ==================================================================

    def export_database(
        self,
        target_path: Path | str,
        *,
        app_version: str = "",
        progress_cb: Optional[ProgressCb] = None,
    ) -> DbExportResult:
        """Write a ``.facedb`` archive containing only the database.

        The database is copied with the SQLite *backup API* (never a raw file
        copy) so a WAL-mode database is captured in a consistent state.  In the
        copy — never in the live database — every image that lives under the
        library root gets its ``relative_path`` filled in, so the package is
        portable even for records indexed before the portable-library feature
        existed.

        Args:
            target_path: Destination file (``.facedb`` is appended when absent).
            app_version: Application version recorded in the manifest.
            progress_cb: Optional ``(current, total, message)`` callback.

        Returns:
            A :class:`DbExportResult`.
        """
        target = Path(target_path)
        if target.suffix.lower() != PACKAGE_EXTENSION:
            target = target.with_name(target.name + PACKAGE_EXTENSION)
        target.parent.mkdir(parents=True, exist_ok=True)

        if not self._db_path.exists():
            raise DatabasePackageError(f"Database not found: {self._db_path}")

        result = DbExportResult(
            package_path=target,
            library_root=str(self._library_root) if self._library_root else None,
        )

        tmp_fd, tmp_name = tempfile.mkstemp(suffix=".db")
        os.close(tmp_fd)
        tmp_db = Path(tmp_name)
        # Write the archive to a temporary file first so a crash never leaves a
        # half-written package in place of an existing valid one.
        pkg_fd, pkg_name = tempfile.mkstemp(
            suffix=PACKAGE_EXTENSION, dir=str(target.parent)
        )
        os.close(pkg_fd)
        tmp_pkg = Path(pkg_name)

        try:
            _emit(progress_cb, 1, 4, "Adatbázis mentése…")
            self._backup_database(tmp_db)

            _emit(progress_cb, 2, 4, "Útvonalak hordozhatóvá tétele…")
            self._portabilize(tmp_db, result)

            _emit(progress_cb, 3, 4, "Csomag írása…")
            db_sha, db_size = _file_digest(tmp_db)
            with zipfile.ZipFile(
                tmp_pkg, "w", compression=zipfile.ZIP_DEFLATED
            ) as zf:
                zf.write(tmp_db, ARCHIVE_DB_PATH)
                manifest = self._build_manifest(
                    app_version=app_version,
                    db_sha=db_sha,
                    db_size=db_size,
                    result=result,
                )
                zf.writestr(
                    MANIFEST_NAME,
                    json.dumps(manifest, indent=2, ensure_ascii=False),
                )

            os.replace(tmp_pkg, target)
            _emit(progress_cb, 4, 4, "Kész")
        except Exception:
            tmp_pkg.unlink(missing_ok=True)
            raise
        finally:
            tmp_db.unlink(missing_ok=True)

        try:
            result.size_bytes = target.stat().st_size
        except OSError:
            result.size_bytes = 0

        log.info(
            "Database package written: %s (%d images, %d portable, %d outside root)",
            target, result.image_count, result.portable_count,
            result.outside_root_count,
        )
        return result

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def _backup_database(self, target: Path) -> None:
        """Copy the live database into *target* via the SQLite backup API."""
        src = sqlite3.connect(str(self._db_path))
        try:
            dst = sqlite3.connect(str(target))
            try:
                src.backup(dst)
                dst.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            finally:
                dst.close()
        finally:
            src.close()

    def _portabilize(self, db_file: Path, result: DbExportResult) -> None:
        """Fill in missing ``relative_path`` values in the exported copy."""
        conn = sqlite3.connect(str(db_file))
        try:
            rows = conn.execute(
                "SELECT id, file_path, relative_path FROM images"
            ).fetchall()
            result.image_count = len(rows)
            updates: List[Tuple[str, int]] = []
            for image_id, file_path, relative_path in rows:
                if relative_path:
                    result.portable_count += 1
                    continue
                rel = _relative_to_root(file_path, self._library_root)
                if rel is None:
                    result.outside_root_count += 1
                    continue
                updates.append((rel, image_id))
                result.portable_count += 1
            if updates:
                conn.executemany(
                    "UPDATE images SET relative_path = ? WHERE id = ?", updates
                )
                conn.commit()
                log.info(
                    "Export: derived relative_path for %d image(s)", len(updates)
                )
        finally:
            conn.close()

    def _build_manifest(
        self,
        *,
        app_version: str,
        db_sha: str,
        db_size: int,
        result: DbExportResult,
    ) -> dict:
        return {
            "package_format_version": PACKAGE_FORMAT_VERSION,
            "kind": PACKAGE_KIND,
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
            "source_library_root": result.library_root,
            "images": {
                "total": result.image_count,
                "with_relative_path": result.portable_count,
                "outside_root": result.outside_root_count,
            },
            "contains": ["database"],
        }

    # ==================================================================
    # Validation
    # ==================================================================

    @staticmethod
    def validate_package(package_path: Path | str) -> DbValidationResult:
        """Structurally validate a ``.facedb`` file.

        Checks the extension, that the file is a readable ZIP, that a
        ``manifest.json`` exists and parses, and that the declared database
        member is present and free of path-traversal tricks.
        """
        path = Path(package_path)
        errors: List[str] = []

        if path.suffix.lower() != PACKAGE_EXTENSION:
            errors.append(
                f"Érvénytelen kiterjesztés: {path.suffix!r} "
                f"(várt: {PACKAGE_EXTENSION})"
            )
            return DbValidationResult(ok=False, errors=errors)

        if not path.is_file():
            errors.append(f"A fájl nem található: {path}")
            return DbValidationResult(ok=False, errors=errors)

        if not zipfile.is_zipfile(path):
            errors.append("A fájl nem olvasható ZIP-archívumként.")
            return DbValidationResult(ok=False, errors=errors)

        try:
            with zipfile.ZipFile(path, "r") as zf:
                names = set(zf.namelist())

                if MANIFEST_NAME not in names:
                    errors.append("Hiányzik a manifest.json.")
                    return DbValidationResult(ok=False, errors=errors)

                try:
                    manifest = json.loads(zf.read(MANIFEST_NAME).decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    errors.append(f"A manifest.json nem értelmezhető: {exc}")
                    return DbValidationResult(ok=False, errors=errors)

                db_member = (
                    manifest.get("database", {}).get("archive_path")
                    or ARCHIVE_DB_PATH
                )
                if db_member not in names:
                    errors.append(f"Hiányzik az adatbázis: {db_member}")

                for name in names:
                    if _is_unsafe_member(name):
                        errors.append(
                            f"Nem biztonságos útvonal az archívumban: {name}"
                        )
                        break
        except zipfile.BadZipFile as exc:
            errors.append(f"Sérült ZIP-archívum: {exc}")
            return DbValidationResult(ok=False, errors=errors)

        if errors:
            return DbValidationResult(ok=False, manifest=None, errors=errors)
        return DbValidationResult(ok=True, manifest=manifest, errors=[])

    # ==================================================================
    # Import
    # ==================================================================

    @staticmethod
    def import_database(
        package_path: Path | str,
        dest_dir: Path | str,
        library_root: Path | str,
        *,
        progress_cb: Optional[ProgressCb] = None,
    ) -> DbImportResult:
        """Extract a ``.facedb`` and re-base every path onto *library_root*.

        Args:
            package_path: Source ``.facedb`` file.
            dest_dir:     Target project directory (created if absent; must be
                          empty or not yet exist so no data is clobbered).
            library_root: The base image folder **on this machine** — the local
                          counterpart of the exporting machine's root.
            progress_cb:  Optional ``(current, total, message)`` callback.

        Returns:
            A :class:`DbImportResult`.
        """
        package_path = Path(package_path)
        dest = Path(dest_dir)
        root = Path(library_root).expanduser()

        validation = DatabasePackageService.validate_package(package_path)
        if not validation.ok:
            raise DatabasePackageError(
                "Érvénytelen .facedb: " + "; ".join(validation.errors)
            )

        if dest.exists() and any(dest.iterdir()):
            raise DatabasePackageError(
                f"A célmappa nem üres: {dest}. Válassz üres vagy új mappát."
            )
        dest.mkdir(parents=True, exist_ok=True)

        manifest = validation.manifest or {}
        db_member = (
            manifest.get("database", {}).get("archive_path") or ARCHIVE_DB_PATH
        )
        db_path = dest / ARCHIVE_DB_NAME
        crops_dir = dest / "crops"
        crops_dir.mkdir(exist_ok=True)

        _emit(progress_cb, 1, 3, "Adatbázis kibontása…")
        with zipfile.ZipFile(package_path, "r") as zf:
            if _is_unsafe_member(db_member):
                raise DatabasePackageError(
                    f"Path traversal kísérlet az archívumban: {db_member}"
                )
            with zf.open(db_member, "r") as src, open(db_path, "wb") as out:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)

        result = DbImportResult(
            project_dir=dest,
            db_path=db_path,
            crops_dir=crops_dir,
            library_root=root,
            manifest=manifest,
        )

        _emit(progress_cb, 2, 3, "Útvonalak igazítása…")
        DatabasePackageService.rebase_paths(
            db_path,
            target_root=root,
            source_root=manifest.get("source_library_root"),
            crops_dir=crops_dir,
            result=result,
        )

        _emit(progress_cb, 3, 3, "Kész")
        _write_local_config(dest / LOCAL_CONFIG_NAME, root)

        log.info(
            "Database package imported into %s (%d re-based, %d resolved, "
            "%d unresolved)",
            dest, result.rebased, result.resolved, result.unresolved,
        )
        return result

    # ------------------------------------------------------------------
    # Path re-basing
    # ------------------------------------------------------------------

    @staticmethod
    def rebase_paths(
        db_path: Path | str,
        *,
        target_root: Path | str,
        source_root: Optional[str],
        crops_dir: Path | str,
        result: Optional[DbImportResult] = None,
    ) -> DbImportResult:
        """Rewrite machine-specific paths in *db_path* for this machine.

        ``images.file_path`` becomes ``target_root / relative_path``.  Rows
        without a stored ``relative_path`` fall back to stripping *source_root*
        from their absolute path (handles databases exported before the
        portable-library feature).  Place/alias thumbnails — which store a copy
        of ``images.file_path`` — follow the same mapping.

        Crop references are rewritten to *crops_dir* by file name, and cleared
        when the file is not present, so the app's crop-repair pass regenerates
        them from the originals.

        Works on a plain SQLite connection (no ORM, no global engine), so it can
        safely run against a database that is not the active project.
        """
        db_path = Path(db_path)
        target = Path(target_root)
        crops_dir = Path(crops_dir)
        res = result or DbImportResult(
            project_dir=db_path.parent,
            db_path=db_path,
            crops_dir=crops_dir,
            library_root=target,
        )

        conn = sqlite3.connect(str(db_path))
        try:
            _rebase_images(conn, target, source_root, res)
            _rebase_crop_columns(conn, crops_dir, res)
            conn.commit()
        finally:
            conn.close()
        return res


# ---------------------------------------------------------------------------
# Re-basing internals
# ---------------------------------------------------------------------------

def _rebase_images(
    conn: sqlite3.Connection,
    target_root: Path,
    source_root: Optional[str],
    result: DbImportResult,
) -> None:
    """Point every image row at its location under *target_root*."""
    rows = conn.execute(
        "SELECT id, file_path, relative_path FROM images"
    ).fetchall()

    # old absolute path -> new absolute path (for the thumbnail columns below)
    mapping: Dict[str, str] = {}
    updates: List[Tuple[str, str, int]] = []

    for image_id, file_path, relative_path in rows:
        rel = relative_path or _relative_to_root(file_path, source_root)
        if not rel:
            # Nothing portable to work with — leave the original path alone and
            # let the path matcher deal with it.
            result.unresolved += 1
            continue
        rel = _normalize_posix(rel)
        new_path = str(target_root / _from_posix(rel))
        updates.append((new_path, rel, image_id))
        if file_path and file_path != new_path:
            mapping[file_path] = new_path
        if os.path.exists(new_path):
            result.resolved += 1
        else:
            result.unresolved += 1

    if not updates:
        return

    # ``images.file_path`` is UNIQUE, so a direct rewrite can transiently
    # collide with a row that has not been re-based yet.  Park every affected
    # row on a guaranteed-unique placeholder first, then write the final value.
    conn.executemany(
        "UPDATE images SET file_path = ? WHERE id = ?",
        [(f"\x00pending:{image_id}", image_id) for _, _, image_id in updates],
    )
    for new_path, rel, image_id in updates:
        try:
            conn.execute(
                "UPDATE images SET file_path = ?, relative_path = ? WHERE id = ?",
                (new_path, rel, image_id),
            )
            result.rebased += 1
        except sqlite3.IntegrityError:
            # Two records would land on the same file — keep the first and drop
            # this one back to something unique so the database stays valid.
            log.warning(
                "Import: duplicate target path for image %s: %s", image_id, new_path
            )
            conn.execute(
                "UPDATE images SET file_path = ?, relative_path = ? WHERE id = ?",
                (f"{new_path}#dup{image_id}", rel, image_id),
            )

    if mapping:
        _remap_columns(conn, _IMAGE_PATH_COLUMNS, mapping)


def _rebase_crop_columns(
    conn: sqlite3.Connection, crops_dir: Path, result: DbImportResult
) -> None:
    """Repoint crop references at *crops_dir*; clear the ones without a file."""
    for table, column in _CROP_PATH_COLUMNS:
        if not _has_column(conn, table, column):
            continue
        rows = conn.execute(
            f"SELECT rowid, {column} FROM {table} WHERE {column} IS NOT NULL"
        ).fetchall()
        updates: List[Tuple[Optional[str], int]] = []
        for rowid, value in rows:
            if not value:
                continue
            candidate = crops_dir / _basename_any(value)
            if candidate.exists():
                new_value: Optional[str] = str(candidate)
            else:
                # No local crop file: NULL it so the crop-repair pass rebuilds
                # it from the original image instead of showing a dead path.
                new_value = None
                result.cleared_crops += 1
            if new_value != value:
                updates.append((new_value, rowid))
        if updates:
            conn.executemany(
                f"UPDATE {table} SET {column} = ? WHERE rowid = ?", updates
            )


def _remap_columns(
    conn: sqlite3.Connection,
    columns: Tuple[Tuple[str, str], ...],
    mapping: Dict[str, str],
) -> None:
    """Apply an old→new path *mapping* to each ``(table, column)`` pair."""
    for table, column in columns:
        if not _has_column(conn, table, column):
            continue
        rows = conn.execute(
            f"SELECT rowid, {column} FROM {table} WHERE {column} IS NOT NULL"
        ).fetchall()
        updates = [
            (mapping[value], rowid)
            for rowid, value in rows
            if value in mapping
        ]
        if updates:
            conn.executemany(
                f"UPDATE {table} SET {column} = ? WHERE rowid = ?", updates
            )


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """True when *table* exists in the database and has *column*.

    Older databases predate some tables/columns; a missing one is skipped
    rather than aborting the whole import.
    """
    try:
        info = conn.execute(f"PRAGMA table_info({table})").fetchall()
    except sqlite3.DatabaseError:
        return False
    return any(row[1] == column for row in info)


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
    """Normalize a relative path to forward slashes without a leading slash."""
    parts = [p for p in rel.replace("\\", "/").split("/") if p and p != "."]
    return "/".join(parts)


def _from_posix(rel: str) -> Path:
    """Turn a POSIX-style relative path into a native :class:`Path`."""
    parts = [p for p in rel.split("/") if p]
    return Path(*parts) if parts else Path(".")


def _basename_any(path_str: str) -> str:
    """Return the file name of *path_str*, honouring ``/`` and ``\\``.

    ``pathlib`` only splits on the *host* separator, so a Windows path read on
    macOS/Linux would stay one giant component.  Splitting on both recovers the
    bare name no matter which OS wrote it.
    """
    return path_str.replace("\\", "/").rsplit("/", 1)[-1].strip() or "file"


def _relative_to_root(
    file_path: Optional[str], root: Optional[Path | str]
) -> Optional[str]:
    """Return *file_path* relative to *root* as a POSIX string, or ``None``.

    Pure string/`PurePath` arithmetic — the paths may well come from another
    operating system, so nothing is touched on disk and no ``resolve()`` is
    attempted.  Comparison is case-insensitive for Windows-style roots, which
    is what those filesystems do.
    """
    if not file_path or not root:
        return None
    root_str = str(root)
    windows = "\\" in root_str or "\\" in file_path or _looks_like_windows(root_str)
    pure = PureWindowsPath if windows else PurePosixPath
    try:
        rel = pure(file_path).relative_to(pure(root_str))
    except ValueError:
        return None
    parts = [p for p in rel.as_posix().split("/") if p and p != "."]
    return "/".join(parts) or None


def _looks_like_windows(path_str: str) -> bool:
    """True for a drive-letter or UNC path, regardless of the host OS."""
    return bool(PureWindowsPath(path_str).drive)


def _is_unsafe_member(name: str) -> bool:
    """Return ``True`` if a ZIP member name is absolute or escapes the root."""
    if not name:
        return False
    normalized = name.replace("\\", "/")
    if normalized.startswith("/"):
        return True
    if len(normalized) >= 2 and normalized[1] == ":":
        return True
    return any(part == ".." for part in normalized.split("/"))


def _file_digest(path: Path) -> Tuple[str, int]:
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


def _write_local_config(config_path: Path, library_root: Path) -> None:
    """Write/merge ``project.local.json`` so the app finds the images."""
    data: dict = {}
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            data = {}
    data["image_library_root"] = str(library_root)
    config_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
