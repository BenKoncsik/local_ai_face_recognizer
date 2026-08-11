# Face-Local — Agent / Developer Guide

**Face-Local** is an offline, privacy-first desktop application for grouping photos by the people in them. It uses computer vision to detect and embed faces, then clusters them into identity groups — all locally, with no data sent to any server.

---

## TL;DR — Quick Commands

```bash
bash scripts/build_and_run.sh        # Run app (handles all setup)
pytest tests/                         # Run all tests
pytest -q tests/test_post_buffer_release.py  # Test release posting
python -m app.main                   # Run app directly (after setup)
```

**Languages**: English / Hungarian (EN / HU) via [`app/ui/i18n.py`](app/ui/i18n.py) — ALL UI strings in both.

**⚠️ Git Workflow**: You **cannot commit directly**. Stage changes with `git add`, then request human approval. The human reviews and commits with proper GitHub issue references (`#13 #18 #10`).

---

## Core Principles

1. **Zero-Research, One-Click Fixes** (most important) — users never research; app always shows exact fix commands or one-click button
2. **Bilingual UI** — every string in EN + HU via `t("key")`
3. **Graceful Fallbacks** — missing TPU/models → CPU/HOG stub, not crashes
4. **Terse Responses** — max 1–2 short paragraphs, minimal tokens, be direct

---

## Common Agent Tasks

| Task | File(s) | Quick Command |
|------|---------|---------------|
| Add UI string (EN + HU) | [`app/ui/i18n.py`](app/ui/i18n.py) | Add to `_STRINGS`, use `t("key")` in widget |
| Fix TPU issue | [`app/ui/dialogs/tpu_status_dialog.py`](app/ui/dialogs/tpu_status_dialog.py) | Show fix dialog with symlink command |
| Add detection model | [`app/detectors/factory.py`](app/detectors/factory.py) | Update `create_detector()`, add fallback |
| Adjust embedding params | [`config.yaml`](config.yaml) → [`app/config.py`](app/config.py) | Update threshold, dimensions |
| Database schema change | [`app/db/models.py`](app/db/models.py) | Update ORM, handle migration |
| Debug face clustering | [`app/clustering/clusterer.py`](app/clustering/clusterer.py) | Adjust DBSCAN epsilon in config |
| Add export format | [`app/services/export_service.py`](app/services/export_service.py) | Extend `export_by_person()` |
| Move a project to another machine | [`app/services/database_package_service.py`](app/services/database_package_service.py) | `.facedb` = DB only; import re-bases paths onto the local library root |
| Re-attach moved / renamed originals | [`app/services/image_path_matcher.py`](app/services/image_path_matcher.py) | Scan folders, match by name + SHA-256, review in `ImagePathRepairDialog` |
| Trace pipeline steps | [`app/workers/pipeline_worker.py`](app/workers/pipeline_worker.py) | Follow scan → detect → embed → cluster |
| Add background task | [`app/tasks/manager.py`](app/tasks/manager.py) | Use `get_task_manager().submit(name, work_fn, priority=TaskPriority.HIGH)` |

---

## Zero-Research, One-Click Fixes

Users never research; app always shows exact fix commands or one-click button. Examples:
- Build script auto-downloads models, installs `libedgetpu` with `sudo`
- TPU Status dialog shows exact symlink command + "Auto-fix" button
- Missing `mobilefacenet.tflite` → exact GitHub URL shown
- Missing `libedgetpu` on Apple Silicon → exact symlink shown in dialog

---

## Key APIs

| Module | Key Functions/Classes | Purpose |
|--------|----------------------|---------|
| [`app/config.py`](app/config.py) | `AppConfig`, `DetectionConfig`, `EmbeddingConfig` | Load/validate all settings |
| [`app/db/database.py`](app/db/database.py) | `session_scope()`, `init_db()` | DB access (always use context manager) |
| [`app/detectors/factory.py`](app/detectors/factory.py) | `create_detector()` | Coral → CPU fallback |
| [`app/embeddings/tflite_embedder.py`](app/embeddings/tflite_embedder.py) | `TfliteEmbedder.embed()` | Face → 192/512-dim embedding |
| [`app/clustering/clusterer.py`](app/clustering/clusterer.py) | `Clusterer.cluster()` | DBSCAN cosine distance |
| [`app/services/scan_service.py`](app/services/scan_service.py) | `ScanService.scan()` | Find new image files |
| [`app/services/detection_service.py`](app/services/detection_service.py) | `DetectionService.detect_faces()` | Run detector on image |
| [`app/ui/i18n.py`](app/ui/i18n.py) | `t("key")` | Translate string to EN/HU |
| [`app/tasks/manager.py`](app/tasks/manager.py) | `get_task_manager().submit()`, `TaskPriority`, `BackgroundTask` | Priority-scheduled background tasks with UI visibility + auto-cleanup |

---

## Project Structure

```
local_ai_face_recognizer/
├── app/
│   ├── main.py                    # Entry point: CLI args, QApplication, MainWindow
│   ├── config.py                  # YAML/dataclass config loading and path resolution
│   ├── app_settings.py            # QSettings-backed app preferences
│   ├── diagnostics.py             # Environment and runtime diagnostics helpers
│   ├── logging_setup.py           # File/GUI logging setup
│   ├── paths.py                   # Frozen/dev path helpers
│   ├── db/
│   │   ├── database.py            # session_scope(), init_db()
│   │   └── models.py              # SQLAlchemy ORM: images, faces, people, metadata, places, groups
│   ├── detectors/
│   │   ├── base.py                # FaceDetector ABC, Detection dataclass
│   │   ├── factory.py             # create_detector(): Coral/YuNet/CPU fallback
│   │   ├── coral_detector.py      # EdgeTPU detector
│   │   ├── cpu_detector.py        # OpenCV DNN / Haar fallback
│   │   └── yunet_detector.py      # OpenCV YuNet detector with landmarks
│   ├── embeddings/
│   │   ├── base.py                # FaceEmbedder ABC
│   │   ├── alignment.py           # Face crop alignment helpers
│   │   ├── sface_embedder.py      # OpenCV SFace backend
│   │   └── tflite_embedder.py     # MobileFaceNet/ArcFace-style TFLite (+ fallback)
│   ├── deep/
│   │   ├── classifier.py          # Optional learned classifier layer
│   │   ├── dataset.py             # Training/inference dataset helpers
│   │   └── debug_info.py          # Deep recognition diagnostics
│   ├── clustering/
│   │   └── clusterer.py           # DBSCAN over cosine distance
│   ├── gdrive/
│   │   ├── drive_client.py        # Google Drive API adapter
│   │   ├── oauth_flow.py          # OAuth flow
│   │   ├── credential_store.py    # Token/keyring storage
│   │   ├── project_session.py     # Lock/heartbeat project session
│   │   ├── storage_provider.py    # Local/Drive storage abstraction
│   │   ├── drive_scan_service.py  # Drive-backed scan integration
│   │   └── cache.py               # Ephemeral Drive cache
│   ├── jobs/
│   │   ├── cancellation.py        # Shared cancellation tokens
│   │   └── job_types.py           # Background job result/progress types
│   ├── services/
│   │   ├── scan_service.py        # Local image discovery, hashing, metadata
│   │   ├── detection_service.py   # Run detector, save Face records/crops
│   │   ├── embedding_service.py   # Run embedder, save embeddings
│   │   ├── recognition_service.py # Known-person matching and assignment
│   │   ├── clustering_service.py  # Unknown-person clustering and recluster
│   │   ├── suggestion_service.py  # Unknown -> known-person suggestions
│   │   ├── identity_service.py    # Rename / merge / delete / reassign people
│   │   ├── person_service.py      # Person profile and detail operations
│   │   ├── person_group_service.py # Person groups and group membership
│   │   ├── image_library_service.py # Library/project state and image sources
│   │   ├── image_browser_service.py # Browser tab queries and image details
│   │   ├── face_*_service.py      # Face crop/date/quality/diagnostics/export/verification helpers
│   │   ├── family*_*.py           # Family search and family-code parsing/schemes
│   │   ├── place_service.py       # Places, geocoding links, map/search data
│   │   ├── object_service.py      # Object tags, occurrences, and merges
│   │   ├── collage_*.py           # Collage parser and persistence
│   │   ├── *_merge*_service.py    # Unknown/auto/merge suggestion workflows
│   │   ├── *_duplicate*_service.py # Duplicate/overlap/intra-image cleanup
│   │   ├── screen_recorder_service.py # ffmpeg screen/audio recording
│   │   ├── project_package_service.py # Full .facepack export/import (db+crops+images)
│   │   ├── database_package_service.py # DB-only .facedb export/import, path re-basing
│   │   ├── image_path_matcher.py  # Find missing originals, match by name/content hash
│   │   ├── export_service.py      # CSV/JSON/image/Astro static export
│   │   ├── update_service.py      # Sparkle/appcast update checks
│   │   └── geocoding/             # Provider interface + Nominatim provider
│   ├── workers/
│   │   ├── pipeline_worker.py     # QThread: scan -> detect -> embed -> recognize -> cluster -> suggest
│   │   ├── deep_pipeline_worker.py # Deep recognition background run
│   │   ├── match_job_worker.py    # Batch match jobs
│   │   ├── thumbnail_worker.py    # Thumbnail loading/generation
│   │   ├── drive_image_worker.py  # Drive thumbnail/image fetch
│   │   └── *_worker.py            # Export, geocoding, rerecognition, Astro jobs
│   ├── updater/
│   │   └── sparkle_bridge.py      # macOS Sparkle helper bridge
│   ├── utils/
│   │   ├── exif.py                # EXIF read/write helpers
│   │   ├── image_metadata.py      # Image metadata extraction
│   │   ├── image_utils.py         # Image/path utilities
│   │   ├── person_search.py       # Person search helpers
│   │   └── fuzzy_date.py          # Fuzzy/partial date handling
│   └── ui/
│       ├── i18n.py                # t("key") - EN + HU strings
│       ├── main_window.py         # Main QMainWindow
│       ├── theme.py               # Application theme
│       ├── display_utils.py       # UI formatting helpers
│       ├── panels/
│       │   ├── cluster_panel.py   # Face recognition/person face grid
│       │   ├── image_browser_panel.py # Photo browser and metadata workflows
│       │   ├── persons_panel.py   # Person detail/list tab
│       │   ├── family_search_panel.py # Family search tab
│       │   ├── locations_panel.py # Places/map tab
│       │   ├── objects_panel.py   # Object tag tab
│       │   ├── collage_panel.py   # Collage workflow tab
│       │   ├── groups_panel.py    # Person groups tab
│       │   ├── preview_panel.py   # Image preview with bbox overlay
│       │   ├── sidebar_panel.py   # Legacy/person sidebar
│       │   └── log_panel.py       # Activity log dock
│       ├── dialogs/               # Settings, Drive, export, merge, place/object, repair, update dialogs
│       ├── widgets/               # Shared Qt widgets (search selects, maps, grids, recording controls)
│       ├── helpers/               # UI helper algorithms
│       └── workers/               # UI-local async helpers
├── assets/
│   ├── entitlements.plist         # macOS signing/packaging entitlements
│   └── icons/                     # App icons (.png, .ico, .icns, iconset)
├── models/                        # Versioned detector/embedder assets plus optional local models
│   ├── deploy.prototxt
│   ├── res10_300x300_ssd_iter_140000.caffemodel
│   ├── face_detection_yunet_2023mar.onnx
│   ├── sface.onnx
│   ├── ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite
│   └── mobilefacenet.tflite       # Optional/manual local embedding model
├── data/                          # Runtime DB, crops, caches, trained model data (gitignored)
│   ├── faces.db                   # SQLite with WAL, local runtime only
│   ├── crops/                     # Face crop thumbnails
│   └── deep_model/                # Optional trained classifier artifacts
├── web/
│   └── astro/                     # Static HTML gallery export (Astro SSG)
│       ├── src/                   # Astro pages/components/lib/data loaders
│       ├── public/                # Generated export assets (gitignored)
│       └── dist/                  # Built static site (gitignored)
├── sparkle/
│   └── Sources/SparkleHelper/     # Swift helper used by Sparkle update flow
├── scripts/
│   ├── build_and_run.sh           # macOS / Linux setup + launch
│   ├── build_and_run.bat          # Windows setup + launch
│   ├── build_and_run.ps1          # PowerShell setup + launch
│   ├── run_tests.sh / run_tests.ps1 # Test runners
│   ├── package_app.py             # Desktop packaging
│   ├── build_linux_deb.sh         # Linux .deb packaging
│   ├── build_windows_installer.iss # Inno Setup installer script
│   ├── generate_appcast.py        # Sparkle appcast generation
│   ├── github_release.py          # GitHub release automation
│   ├── post_buffer_release.py / post_x_release.py # Social release posting helpers
│   └── detect_*_probe.py          # Detector diagnostics
├── docs/                          # Blueprint, packaging docs, appcast/wiki artifacts
├── memory/                        # Agent memory/project notes
├── tests/                         # Pytest suite across DB, services, UI, Drive, export, release
├── .github/
│   ├── workflows/                 # Release and test dashboard workflows
│   └── scripts/                   # GitHub project automation
├── config.yaml                    # Checked-in default/local config
├── config.example.yaml            # Example config template
├── requirements.txt               # Runtime dependencies
├── requirements-windows.txt       # Windows-specific dependency set
├── pyproject.toml                 # Package metadata, deps, pytest/ruff/mypy config
├── README.md
└── AGENT.md
```

---

## Test & Debug Commands

```bash
# Run app (one command; handles all setup)
bash scripts/build_and_run.sh

# Run tests
pytest tests/
pytest tests/test_post_buffer_release.py -v

# Direct launch (after setup)
python -m app.main

# Check detection (no GUI)
python -c "from app.detectors.factory import create_detector; d = create_detector(); print(type(d))"

# Validate config
python -c "from app.config import load_config; cfg = load_config(); print(cfg)"

# Database check
python -c "from app.db.database import init_db; init_db(); print('DB ready')"
```

**Build script does:**
- Finds Python 3.11+ (tries 3.13, 3.12, 3.11)
- Removes stale venv, installs deps
- Tries TPU packages; continues if unavailable
- Installs `libedgetpu` system driver
- Downloads missing model files
- Auto-generates `config.yaml`
- Launches app

---

## Architecture Decisions (Why?)

| Decision | Why | File |
|----------|-----|------|
| **Coral → CPU fallback** | Graceful degradation; app works offline without hardware | [`detectors/factory.py`](app/detectors/factory.py) |
| **Use `ai-edge-litert`, not `tflite-runtime`** | No Python 3.12 wheels for `tflite-runtime` | [`detectors/coral_detector.py`](app/detectors/coral_detector.py) |
| **SQLite + WAL** | No external DB dependency; offline-first; fast concurrent reads | [`db/database.py`](app/db/database.py) |
| **Session context manager** | Auto-commit/rollback; prevents transaction leaks | All services |
| **MobileFaceNet fallback** | Manual download (no canonical URL); fallback HOG keeps app running | [`embeddings/tflite_embedder.py`](app/embeddings/tflite_embedder.py) |
| **i18n via `t("key")`** | Bilingual requirement; single source of truth | [`ui/i18n.py`](app/ui/i18n.py) |
| **Multiple DB support** | Let users switch databases; path saved in `~/.face_local_prefs.json` | [`ui/dialogs/settings_dialog.py`](app/ui/dialogs/settings_dialog.py) |
| **TaskManager for all background work** | Unified scheduling, priority queue, pause/resume, UI visibility; auto-cleanup after 5 min | [`tasks/manager.py`](app/tasks/manager.py) |

---

## MobileFaceNet Model

**Not auto-downloaded** (no canonical URL). Without it, clustering is poor (HOG fallback). Place at `models/mobilefacenet.tflite` (must be shape `[1, 192]` or `[1, 512]`).

**Get from**: [Hucao90/MobileFaceNet](https://github.com/Hucao90/MobileFaceNet)

Build script validates shape on every run.

---

## Config Reference

```yaml
detection:
  confidence_threshold: 0.65     # Min confidence (0–1)
  min_face_size: 50              # Min bbox width/height
  coral_model_path: models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite
  cpu_model_path: models/res10_300x300_ssd_iter_140000.caffemodel

embedding:
  model_path: models/mobilefacenet.tflite
  input_size: [112, 112]
  embedding_dim: 192

clustering:
  epsilon: 0.4                   # DBSCAN cosine distance (adjust for loose/tight clusters)
  min_samples: 2                 # Min faces per cluster

suggestions:
  similarity_threshold: 0.5      # Min cosine similarity to suggest
  max_suggestions_per_person: 3

storage:
  db_path: data/faces.db
  crops_dir: data/crops

scan:
  image_extensions: [.jpg, .jpeg, .png, .webp]
  worker_threads: 2
  thumbnail_size: [128, 128]
```

---

## UI Feature Checklist

1. **Strings**: Add to [`app/ui/i18n.py`](app/ui/i18n.py) (EN + HU)
2. **Widget code**: Use `t("key")`, never hardcode
3. **Persistent UI**: Add to `_retranslate()` in [`main_window.py`](app/ui/main_window.py)
4. **New dependency**: Update `pyproject.toml`, add graceful fallback + clear error message with install command
5. **Dialogs**: Both OK/Cancel and Close buttons translated

---

## Troubleshooting Tree

**App won't start?**
- → `ModuleNotFoundError` → `bash scripts/build_and_run.sh` (re-installs all deps)
- → `libedgetpu` error on Apple Silicon → symlink: `sudo ln -sf /usr/local/lib/libedgetpu.1.dylib /opt/homebrew/lib/libedgetpu.1.dylib`
- → Missing model files → build script auto-downloads; check `models/` directory

**Detector issues?**
- → 0 faces detected → images marked `detection_done=True`; click "Force Full Rescan"
- → TPU device not working → auto-falls back to CPU; check TPU Status dialog for driver issues

**Clustering issues?**
- → All faces in one cluster → MobileFaceNet missing; place at `models/mobilefacenet.tflite`
- → Poor clustering → adjust `clustering.epsilon` in `config.yaml` (default 0.4)

**UI strings not translated?**
- → Check [`app/ui/i18n.py`](app/ui/i18n.py): missing EN or HU entry? Add both.
- → Widget not using `t("key")`? Update widget code.

**Database issues?**
- → Check `data/faces.db` exists; uses SQLAlchemy + WAL mode
- → All access via `session_scope()` context manager

---

## Release Social Posting (Buffer)

This repository has release-to-social automation via Buffer.

Current implementation:
- Workflow file: `.github/workflows/build-release.yml`
- Posting script: `scripts/post_buffer_release.py`
- Tests: `tests/test_post_buffer_release.py`

Behavior:
- After the `release` job succeeds, a separate `post-to-buffer` job runs.
- It runs only for `push` events and only on the first workflow attempt:
  `needs.release.result == 'success' && github.event_name == 'push' && github.run_attempt == 1`
- The posting step uses `continue-on-error: true`, so social posting must never fail the release.
- Default posting mode is `shareNow`, so the post should publish immediately.
- If `BUFFER_POST_MODE` is empty, the script falls back to `shareNow`.

Required org/repo secret:
- `BUFFER_API_KEY`

Optional org/repo secrets:
- `BUFFER_CHANNEL_ID` — recommended if multiple Buffer channels exist
- `BUFFER_CHANNEL_NAME` — optional fallback selector when channel ID is not set
- `BUFFER_ORGANIZATION_ID` — optional, otherwise the script auto-discovers organizations
- `BUFFER_CHANNEL_SERVICE` — optional, defaults to `twitter`
- `BUFFER_POST_MODE` — optional, defaults to `shareNow`
- `BUFFER_POST_TEMPLATE` — optional custom post template

Buffer posting details:
- The script creates a text post through Buffer's GraphQL API at `https://api.buffer.com`
- It auto-selects the first connected `twitter` channel when no explicit channel secret is set
- If multiple matching channels exist, the script picks the first sorted match and prints a warning
- The template supports:
  `{app_name}`, `{tag}`, `{version}`, `{platforms}`, `{release_url}`
- Successful platform names are derived from the build job results
- The generated text is capped at 280 characters

Recommended setup for this repo:
- Set `BUFFER_API_KEY`
- Leave `BUFFER_POST_MODE` empty or set it explicitly to `shareNow`
- If more than one Buffer/X account is connected, set `BUFFER_CHANNEL_ID`

Useful local checks:

```bash
pytest -q tests/test_post_buffer_release.py

BUFFER_POST_MODE='' python3 scripts/post_buffer_release.py \
  --dry-run \
  --app-name "Face-Local" \
  --tag "v1.2.3" \
  --release-url "https://github.com/example/repo/releases/tag/v1.2.3" \
  --platform "macOS=success" \
  --platform "Windows=success"
```
