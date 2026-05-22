# Face-Local — Agent / Developer Guide

## Project Overview

**Face-Local** is an offline, privacy-first desktop application for grouping photos by the people in them. It uses computer vision to detect and embed faces, then clusters them into identity groups — all locally, with no data sent to any server.

**Languages supported: English / Hungarian (EN / HU)** — every user-facing string must be available in both languages via the `app/ui/i18n.py` module.

---

## RESPONSE STYLE

- max 1–2 short paragraphs
- minimal tokens
- no unnecessary explanation
- be direct and compressed

---

## UX Principle: Zero-Research, One-Click Fixes

**This is the most important design constraint of the project.**

Users are not expected to know how to install native libraries, download ML models, or debug Python package conflicts. Whenever a component is missing or broken, the app must:

1. **Tell the user exactly what is wrong** — in plain language, in both EN and HU.
2. **Provide a one-click fix** (button in UI that runs the install automatically), or
3. **Provide a copy-pasteable terminal command** that they can run without modification.

Never tell the user to "check the documentation" or "install it manually" without also showing them the exact command. Never require the user to research what package to install or where to find a model file.

Examples already implemented:
- Build script (`scripts/build_and_run.sh`) auto-downloads all model files and installs `libedgetpu` with `sudo`.
- TPU Status dialog shows the exact fix commands and has an "Auto-fix" button that runs them.
- If `mobilefacenet.tflite` is missing, the build script prints the exact download URL.
- If `libedgetpu` is installed to `/usr/local/lib` but not found at `/opt/homebrew/lib` (Apple Silicon), the fix dialog shows the exact symlink command.

---

## Project Structure

```
local_ai_face_recognizer/
├── app/
│   ├── config.py                  # Pydantic config (AppConfig, DetectionConfig, …)
│   ├── main.py                    # Entry point: QApplication, MainWindow
│   ├── logging_setup.py
│   ├── db/
│   │   ├── database.py            # SQLAlchemy engine, session_scope(), init_db()
│   │   └── models.py              # ORM: Image, Face, Person, FaceCorrection
│   ├── detectors/
│   │   ├── base.py                # FaceDetector ABC, Detection dataclass
│   │   ├── factory.py             # create_detector() — Coral probe + CPU fallback
│   │   ├── coral_detector.py      # EdgeTPU detector (ai-edge-litert)
│   │   └── cpu_detector.py        # OpenCV DNN (Caffe SSD res10)
│   ├── embeddings/
│   │   ├── base.py                # FaceEmbedder ABC
│   │   └── tflite_embedder.py     # MobileFaceNet TFLite embedder (+ HOG stub fallback)
│   ├── clustering/
│   │   └── clusterer.py           # DBSCAN over cosine distance
│   ├── services/
│   │   ├── scan_service.py        # Discovers new image files
│   │   ├── detection_service.py   # Runs detector, saves Face records + crop thumbnails
│   │   ├── embedding_service.py   # Runs embedder, saves embeddings
│   │   ├── clustering_service.py  # Runs DBSCAN, assigns Person IDs
│   │   ├── identity_service.py    # Rename / merge / delete person, reassign face
│   │   ├── suggestion_service.py  # Matches Unknown persons to named-person profiles
│   │   └── export_service.py      # CSV export, image export by person
│   ├── workers/
│   │   └── pipeline_worker.py     # QThread: scan → detect → embed → cluster → suggest
│   └── ui/
│       ├── i18n.py                # All UI strings (EN + HU), t(key) helper
│       ├── main_window.py         # Main QMainWindow
│       ├── panels/
│       │   ├── sidebar_panel.py   # Person list with search
│       │   ├── cluster_panel.py   # Face grid for selected person
│       │   ├── preview_panel.py   # Full image preview with bbox overlay
│       │   └── log_panel.py       # Activity log dock
│       └── dialogs/
│           ├── settings_dialog.py # Language, database, TPU status
│           ├── tpu_status_dialog.py # TPU probe + auto-fix
│           ├── rename_dialog.py
│           ├── merge_dialog.py
│           └── suggestion_dialog.py # Review/approve Unknown → Known name suggestions
├── models/                        # Downloaded model files (gitignored)
│   ├── deploy.prototxt
│   ├── res10_300x300_ssd_iter_140000.caffemodel
│   ├── ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite
│   └── mobilefacenet.tflite       # Must be placed manually (see below)
├── data/                          # Runtime data (gitignored)
│   ├── faces.db                   # SQLite database
│   └── crops/                     # Face crop thumbnails
├── scripts/
│   ├── build_and_run.sh           # Linux / macOS: venv + deps + models + launch
│   └── build_and_run.bat          # Windows: same
├── tests/
├── config.yaml                    # Auto-generated on first run
├── config.example.yaml
└── pyproject.toml
```

---

## How to Run

```bash
bash scripts/build_and_run.sh        # macOS / Linux
scripts\build_and_run.bat            # Windows
```

The build script handles everything automatically:
- Finds Python 3.11+ (tries 3.13, 3.12, 3.11 in order)
- Detects and removes stale venv (wrong Python version)
- Installs / upgrades pip, setuptools, wheel
- Installs all Python dependencies (`pip install -e ".[dev]"`)
- Tries to install TPU packages (`pip install -e ".[tflite]"`) — warns but continues if unavailable
- Checks for / installs `libedgetpu` system driver (macOS: downloads from GitHub; Linux: apt)
- Downloads missing model files (Caffe SSD models, Coral edgetpu model)
- Auto-generates `config.yaml` if missing
- Launches the app (`python -m app.main`)

---

## Key Architecture Decisions

### Detector stack (factory.py)
`create_detector()` tries Coral first, falls back to CPU:
1. `probe_coral()` — attempts to load the EdgeTPU delegate via `ai_edge_litert`
2. If probe succeeds → `CoralDetector` (uses EdgeTPU delegate for inference)
3. If probe fails OR inference throws `RuntimeError` (device disconnected mid-run) → `CpuDetector`

`CpuDetector` uses OpenCV's DNN module with the Caffe SSD res10 model. Includes NMS (`cv2.dnn.NMSBoxes`, IoU 0.4) and an aspect ratio filter (0.4–2.5) to reduce false positives.

### EdgeTPU / Python 3.12 compatibility
`tflite-runtime` does not have Python 3.12 wheels. Use `ai-edge-litert` (Google's replacement).  
`pycoral` is similarly Python ≤3.9 only — all pycoral functionality is reimplemented directly in `coral_detector.py` using raw TFLite tensor API.

### libedgetpu path (Apple Silicon)
The official installer places the library at `/usr/local/lib/libedgetpu.1.dylib`, but Python on Apple Silicon searches `/opt/homebrew/lib/`. The fix is a symlink:
```bash
sudo ln -sf /usr/local/lib/libedgetpu.1.dylib /opt/homebrew/lib/libedgetpu.1.dylib
```
This is shown in the TPU Status dialog with a one-click auto-fix button.

### Embeddings
`TfliteEmbedder` loads `models/mobilefacenet.tflite`. If the model is missing it falls back to a HOG stub (fast but poor — all faces may cluster into one group). The build script warns and shows download instructions if the model is absent.

### i18n
All UI strings live in `app/ui/i18n.py`. Use `t("key")` everywhere in the UI. Never hardcode English text in UI widgets. To add a new string:
1. Add it to `_STRINGS` dict with both `"en"` and `"hu"` values.
2. Call `t("your_key")` in the widget.
3. Add it to `_retranslate()` in `main_window.py` if it's in the toolbar/sidebar.

### Database
SQLite with WAL mode. All access goes through `session_scope()` context manager (auto-commit/rollback).  
Multiple databases are supported — the Settings dialog lets the user create a new DB or open an existing one. The path is stored in `~/.face_local_prefs.json`.

---

## MobileFaceNet Model

This model is **not** auto-downloaded because there is no single canonical redistribution URL. Without it, face grouping quality is poor (HOG fallback).

Options:
- **Recommended**: Get a pre-converted TFLite file from [Hucao90/MobileFaceNet](https://github.com/Hucao90/MobileFaceNet) — place at `models/mobilefacenet.tflite`
- The model must have exactly 1 output of shape `[1, 192]` or `[1, 512]`

The build script validates the model shape on every run and removes incompatible files.

---

## Config Reference (`config.yaml`)

```yaml
detection:
  confidence_threshold: 0.65   # Min detection confidence (0–1)
  min_face_size: 50             # Min face bbox width/height in pixels
  coral_model_path: models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite
  cpu_model_path: models/res10_300x300_ssd_iter_140000.caffemodel

embedding:
  model_path: models/mobilefacenet.tflite
  input_size: [112, 112]
  embedding_dim: 192

clustering:
  epsilon: 0.4       # DBSCAN cosine distance threshold
  min_samples: 2     # Min faces to form a cluster

suggestions:
  similarity_threshold: 0.5       # Min cosine similarity to suggest a match
  max_suggestions_per_person: 3   # Ranked targets proposed per unknown person

storage:
  db_path: data/faces.db
  crops_dir: data/crops

scan:
  image_extensions: [.jpg, .jpeg, .png, .webp]
  worker_threads: 2
  thumbnail_size: [128, 128]
```

---

## Adding New UI Features — Checklist

1. Add string(s) to `app/ui/i18n.py` (both EN and HU)
2. Use `t("key")` in the widget — never hardcode text
3. If the string appears in a persistent widget (toolbar, sidebar), add it to `_retranslate()` in `main_window.py`
4. If the feature requires a new dependency: update `pyproject.toml`, make it optional with a graceful fallback, and handle the missing-dependency case with a clear user message + install command
5. All dialogs must have both OK/Cancel and Close buttons translated

---

## Common Issues

| Symptom | Cause | Fix |
|---|---|---|
| `No module named 'tflite_runtime'` | tflite-runtime has no Python 3.12 wheel | Use `ai-edge-litert` instead |
| `dlopen(libedgetpu.1.dylib, ...)` | Library not in Python's search path | Symlink `/usr/local/lib/` → `/opt/homebrew/lib/` (Apple Silicon) |
| `EdgeTpuDelegateForCustomOp failed to invoke` | USB device not recognized at inference time | App auto-falls back to CPU; re-plug device and restart |
| 0 faces detected on rescan | Images already have `detection_done=True` | Use "Force Full Rescan" button |
| All faces in one cluster | MobileFaceNet model missing, using HOG stub | Place `mobilefacenet.tflite` in `models/` |
| `setuptools.backends.legacy` not found | Old setuptools in stale venv | Delete `.venv/`, re-run build script |

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
