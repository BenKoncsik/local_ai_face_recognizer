---
name: project-portable-image-library
description: Portable Image Library feature — relative path storage for cross-machine database portability
metadata:
  type: project
---

Implemented a portable image library system where image paths are stored as relative POSIX paths instead of (only) absolute paths.

**Why:** The original system stored absolute paths in `Image.file_path`. Moving the DB to another machine or renaming folders broke all image links.

**How to apply:** This is production code, not a workaround. Any path-loading code should use `resolve_image_path(image)` from `app.services.image_library_service` instead of `image.file_path` directly, for cross-machine portability.

## Architecture

- `Image.relative_path` (new DB column, nullable) — POSIX-style path relative to the library root, e.g. `family/IMG_001.jpg`
- `Image.file_path` — still kept for backward compatibility with old DBs
- `project.local.json` (next to the .db file) — machine-specific JSON with `image_library_root`
- `ImageLibraryService` (`app/services/image_library_service.py`) — central service
- Module-level singleton: `init_image_library(db_path)`, `get_image_library_optional()`, `resolve_image_path(image)`

## Key behaviors

- `ScanService` automatically back-fills `relative_path` on every scan when library root is configured
- On a new machine: `ScanService._index_file()` finds existing records via `relative_path` match when `file_path` changes, and updates `file_path` to the current machine's absolute path (re-linking)
- Migration: Settings dialog → Image Library → "Migrate to Relative Paths" button
- Startup check: If root is configured but not found on disk, shows `ImageLibraryMissingDialog`

## Files changed

- `app/services/image_library_service.py` (new)
- `app/ui/dialogs/image_library_dialog.py` (new)
- `tests/test_image_library_service.py` (new — 42 tests)
- `app/db/models.py` — added `relative_path` column to `Image`
- `app/db/database.py` — migration + `init_image_library()` call
- `app/services/scan_service.py` — optional `image_library_svc` param, relative path handling
- `app/services/detection_service.py` — uses `resolve_image_path()` instead of `image.file_path`
- `app/workers/pipeline_worker.py` — passes library svc to ScanService
- `app/ui/i18n.py` — translation strings (`img_lib_*` keys)
- `app/ui/dialogs/settings_dialog.py` — Image Library group
- `app/ui/main_window.py` — startup check, re-init on DB switch
