# Face-Local

**Offline face grouping and person labeling with Google Coral Edge TPU acceleration.**

Scans a folder tree of images, detects faces, groups identical people into clusters, and lets you label, merge, and browse identities — entirely local, no cloud dependency.

---

## What it does

1. **Scan** — recursively indexes images (`.jpg`, `.jpeg`, `.png`, `.webp`), hashes each file, skips unchanged ones on re-runs.
2. **Detect** — runs face detection using a Google Coral Edge TPU (if available) or CPU fallback (OpenCV DNN SSD or Haar cascade).
3. **Embed** — generates 192-dim face embeddings with MobileFaceNet TFLite on **CPU** (Coral is not used for this step — no practical Edge TPU ArcFace model is publicly available).
4. **Cluster** — groups faces by cosine similarity using DBSCAN.
5. **Label** — PySide6 GUI lets you rename clusters, merge wrong splits, reassign individual faces, and mark same/different pairs.
6. **Export** — CSV, JSON reports, or copy face images to a folder.

Everything is persisted in a local SQLite database. No network calls are made.

---

## Architecture

```
app/
├── main.py               Entry point (Qt app init, arg parsing)
├── config.py             AppConfig dataclass + YAML loader
├── logging_setup.py      Structured logging + QLogHandler for GUI
├── db/
│   ├── models.py         SQLAlchemy ORM: Image, Face, Person, FaceCorrection
│   └── database.py       Engine init, session_scope context manager
├── detectors/
│   ├── base.py           FaceDetector ABC + Detection dataclass
│   ├── coral_detector.py CoralDetector — real pycoral integration
│   ├── cpu_detector.py   CpuDetector  — OpenCV DNN SSD + Haar fallback
│   └── factory.py        probe_coral() + create_detector() factory
├── embeddings/
│   ├── base.py           FaceEmbedder ABC
│   └── tflite_embedder.py TFLiteEmbedder (CPU, MobileFaceNet) + HOG stub
├── clustering/
│   └── clusterer.py      cluster_embeddings() — DBSCAN + same-pair constraints
├── services/
│   ├── scan_service.py      ScanService — file discovery + hashing
│   ├── detection_service.py DetectionService — runs detector, saves crops
│   ├── embedding_service.py EmbeddingService — runs embedder, stores vectors
│   ├── clustering_service.py ClusteringService — DBSCAN → Person assignment
│   ├── identity_service.py  IdentityService — rename/merge/reassign ops
│   └── export_service.py    ExportService — CSV/JSON/image export
├── workers/
│   └── pipeline_worker.py PipelineWorker — QThread, runs all 4 stages
└── ui/
    ├── main_window.py     MainWindow — toolbar, splitter, dock
    ├── panels/
    │   ├── sidebar_panel.py  Person list + search
    │   ├── cluster_panel.py  Face thumbnail grid
    │   ├── log_panel.py      Coloured activity log
    │   └── preview_panel.py  Full image preview with bbox overlay
    └── dialogs/
        ├── rename_dialog.py  Rename person dialog
        └── merge_dialog.py   Merge into … dialog
```

---

## Requirements

- **Python 3.11+**
- **Linux** (primary target — tested on Ubuntu 22.04 and Raspberry Pi OS Bookworm)
- **Google Coral USB Accelerator** *(optional)* — for Edge TPU acceleration
- Display / X11 or Wayland for the GUI

macOS and Windows are secondary targets. PySide6 and OpenCV work on both, but pycoral only supports Linux officially.

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url> face-local
cd face-local
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note on tflite-runtime:** The package is available for Linux x86-64 and ARM.
> On macOS or Windows, install TensorFlow instead: `pip install tensorflow`

### 3. Download model files

#### Face detection (CPU) — Caffe SSD res10

```bash
mkdir -p models
wget -P models/ \
  https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt

wget -P models/ \
  https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
```

Without these files, the detector falls back to OpenCV's Haar cascade (lower quality but always available).

#### Face detection (CPU) — YuNet (recommended; enables aligned crops)

YuNet returns 5 facial landmarks per face, which the `aligned` embedding crop
mode needs to warp faces onto the ArcFace template.  When present it is used in
preference to the Caffe SSD / Haar CPU path.

```bash
wget -P models/ \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

Override the location with `detection.yunet_model_path`, or set
`detection.use_yunet: false` to force the Caffe/Haar path.

> **Changing `embedding.crop_mode`** (`legacy` → `square` → `aligned`) changes
> the geometry of every crop and therefore every embedding.  Existing
> embeddings are *not* comparable to new ones, so after switching you must
> re-run a full **re-detect + re-embed** of the library.

#### Face embedding — MobileFaceNet TFLite (CPU only)

The embedding model is NOT included in the repository.  You have three options:

**Option A — Community pre-converted model** (easiest):
```bash
# Search for "mobilefacenet tflite" on GitHub or HuggingFace.
# A commonly used one is from the sirius-ai/MobileFaceNet_TF project.
# Place the downloaded .tflite file at:
cp /path/to/mobilefacenet.tflite models/mobilefacenet.tflite
```

**Option B — Convert from ONNX yourself**:
```bash
pip install onnx onnx-tf tensorflow
# Download mobilefacenet.onnx from InsightFace model zoo
# (https://github.com/deepinsight/insightface/tree/master/model_zoo)
# Then convert — see docs/convert_model.md (TODO: write this guide)
```

**Option C — Use the HOG stub (development only)**:  
Leave `embedding.model_path` unset in `config.yaml`.  The app will warn you and use a low-quality HOG-based fallback.  Pipeline plumbing works, face recognition quality does not.

#### Face detection (Coral Edge TPU)

```bash
wget -P models/ \
  https://raw.githubusercontent.com/google-coral/test_data/master/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite
```

Then set in `config.yaml`:
```yaml
detection:
  coral_model_path: models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite
```

### 4. Install pycoral (Coral users only)

Follow the official guide at https://coral.ai/docs/accelerator/get-started/

```bash
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
  | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update
sudo apt install libedgetpu1-std   # or libedgetpu1-max for max clock speed
pip install pycoral
```

### 5. Configure

```bash
cp config.example.yaml config.yaml
# Edit config.yaml — set model paths, thresholds, etc.
```

---

## Running the application

```bash
# Default (auto-discovers config.yaml in cwd)
python -m app.main

# Explicit config
python -m app.main --config /path/to/config.yaml

# Debug logging
python -m app.main --debug

# Custom database location
python -m app.main --db /tmp/test.db
```

Or if installed via pip:

```bash
face-local --config config.yaml
```

---

## GUI workflow

1. **Select Folder** — choose the root directory containing your images.
2. **Scan & Index** — runs all 4 pipeline stages:
   - Scanning → hashing → DB insert
   - Detection → face bbox → crop thumbnails
   - Embedding → face vectors
   - Clustering → person groups
3. **Browse** — click a person in the left sidebar to see their face thumbnails.
4. **Click a thumbnail** — previews the original image with the face highlighted.
5. **Rename** — give a person a real name.
6. **Merge** — combine two clusters that represent the same person.
7. **Remove Face** — kick a wrong face out of a cluster.
8. **Reassign Face** — move a face to a different cluster.
9. **Re-cluster All** — re-run DBSCAN with current manual corrections applied.
10. **Export** — CSV/JSON report or copy images to a folder.

---

## Coral vs CPU — what's real, what's not

| Stage | Coral | CPU |
|-------|-------|-----|
| Face detection | ✅ Real pycoral integration via `CoralDetector` | ✅ OpenCV DNN SSD + Haar fallback |
| Face embedding | ❌ Not used — no practical public Edge TPU ArcFace model available | ✅ MobileFaceNet TFLite via `TFLiteEmbedder` |
| Clustering | ❌ Not applicable | ✅ scikit-learn DBSCAN |

**Performance expectations:**
- Coral: ~50–200 ms per image for detection (USB Accelerator speed)
- CPU DNN: ~100–500 ms per image depending on hardware
- Haar cascade: fast but misses ~30–40% of faces
- MobileFaceNet embedding: ~10–50 ms per face crop on modern x86 CPU

---

## Testing Coral fallback

To verify the CPU fallback path without Coral hardware:

```bash
# 1. Don't install pycoral — factory will auto-fallback:
python -m app.main

# 2. With pycoral installed but no USB Accelerator:
python -m app.main
# You will see in the log: "pycoral available but no Edge TPU devices found"

# 3. Set coral_model_path but unplug the USB stick:
python -m app.main
# You will see: "Coral init failed: ... — falling back to CPU"

# 4. Run tests:
pytest tests/test_detectors.py -v
# TestFactoryFallback verifies the CPU fallback path with monkeypatching
```

---

## Running tests

```bash
pytest -v

# Specific modules
pytest tests/test_clustering.py -v
pytest tests/test_database.py -v
pytest tests/test_detectors.py -v
pytest tests/test_scan_service.py -v
```

Tests do NOT require a real camera, Coral hardware, or model files.  They use in-memory SQLite databases and synthetic embeddings.

---

## Configuration reference

See [`config.example.yaml`](config.example.yaml) for all options with inline documentation.

Key thresholds:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `detection.confidence_threshold` | `0.5` | Lower → more detections, more false positives |
| `clustering.epsilon` | `0.4` | Lower → stricter matching, more clusters |
| `clustering.min_samples` | `2` | Higher → singletons become noise |
| `scan.thumbnail_size` | `[128, 128]` | Larger → better visual quality, more disk space |

---

## Project status

### Working in MVP
- Full pipeline: scan → detect → embed → cluster → GUI
- Coral detection with real pycoral integration
- CPU detection: OpenCV DNN SSD + Haar cascade fallback
- TFLite embedding with MobileFaceNet (or HOG stub)
- DBSCAN clustering with manual correction constraints
- SQLite persistence with SQLAlchemy ORM
- PySide6 GUI: sidebar, cluster grid, preview, rename, merge, remove, reassign
- Export: CSV, JSON, image folder
- Resumable processing (skips unchanged files)
- Structured logging with GUI log panel

### Placeholder / known limitations
- Embedding model file must be downloaded separately (see Setup)
- CPU model files (Caffe SSD) must be downloaded separately
- Face alignment available via `embedding.crop_mode: aligned` (5-point ArcFace
  alignment) when the YuNet detector supplies landmarks; otherwise crops are
  axis-aligned rectangles with no rotation correction
- Clustering is global re-run (not incremental)
- Re-clustering does not preserve cluster↔person mapping when many clusters change
- `worker_threads` config exists but pipeline is currently serial (QThread runs one thread)
- No HEIC support (extension point exists in `ScanConfig.image_extensions`)
- No split-cluster operation in GUI (remove faces one by one as workaround)

---

## License

[KOAI Personal Use License v1.0](LICENSE)

---

# Magyar változat

# Face-Local

**Offline arccsoportosítás és személycímkézés Google Coral Edge TPU gyorsítással.**

Beolvas egy képeket tartalmazó mappastruktúrát, arcokat érzékel, azonos személyeket klaszterekbe rendez, majd lehetővé teszi az identitások címkézését, egyesítését és böngészését teljesen helyben, felhőfüggőség nélkül.

---

## Mit csinál

1. **Beolvasás** — rekurzívan indexeli a képeket (`.jpg`, `.jpeg`, `.png`, `.webp`), minden fájlt hashel, majd újrafuttatáskor kihagyja a változatlanokat.
2. **Érzékelés** — arcfelismerést futtat Google Coral Edge TPU-val (ha elérhető), vagy CPU-s tartalékmegoldással (OpenCV DNN SSD vagy Haar cascade).
3. **Beágyazás** — 192 dimenziós arcbeágyazásokat generál MobileFaceNet TFLite segítségével **CPU-n** (ehhez a lépéshez a Coral nincs használatban, mert nyilvánosan nem érhető el gyakorlatban használható Edge TPU ArcFace modell).
4. **Klaszterezés** — az arcokat koszinusz-hasonlóság alapján DBSCAN-nel csoportosítja.
5. **Címkézés** — a PySide6 GUI-val átnevezheted a klasztereket, összevonhatod a hibás szétválásokat, újra hozzárendelhetsz egyedi arcokat, és megjelölhetsz azonos/különböző párokat.
6. **Exportálás** — CSV- és JSON-jelentések készítése, vagy az arcképek mappába másolása.

Minden adat egy helyi SQLite-adatbázisban tárolódik. Nem történik hálózati kommunikáció.

---

## Architektúra

```
app/
├── main.py               Belépési pont (Qt app inicializálás, argumentumfeldolgozás)
├── config.py             AppConfig dataclass + YAML betöltő
├── logging_setup.py      Strukturált naplózás + QLogHandler a GUI-hoz
├── db/
│   ├── models.py         SQLAlchemy ORM: Image, Face, Person, FaceCorrection
│   └── database.py       Engine inicializálás, session_scope context manager
├── detectors/
│   ├── base.py           FaceDetector ABC + Detection dataclass
│   ├── coral_detector.py CoralDetector — valódi pycoral integráció
│   ├── cpu_detector.py   CpuDetector — OpenCV DNN SSD + Haar tartalékmegoldás
│   └── factory.py        probe_coral() + create_detector() gyárfüggvény
├── embeddings/
│   ├── base.py           FaceEmbedder ABC
│   └── tflite_embedder.py TFLiteEmbedder (CPU, MobileFaceNet) + HOG csonk
├── clustering/
│   └── clusterer.py      cluster_embeddings() — DBSCAN + same-pair korlátozások
├── services/
│   ├── scan_service.py      ScanService — fájlfelderítés + hashelés
│   ├── detection_service.py DetectionService — detektor futtatása, kivágások mentése
│   ├── embedding_service.py EmbeddingService — beágyazó futtatása, vektorok tárolása
│   ├── clustering_service.py ClusteringService — DBSCAN → Person hozzárendelés
│   ├── identity_service.py  IdentityService — átnevezés/egyesítés/áthelyezés műveletek
│   └── export_service.py    ExportService — CSV/JSON/kép export
├── workers/
│   └── pipeline_worker.py PipelineWorker — QThread, mind a 4 lépcsőt futtatja
└── ui/
    ├── main_window.py     MainWindow — eszköztár, splitter, dock
    ├── panels/
    │   ├── sidebar_panel.py  Személylista + keresés
    │   ├── cluster_panel.py  Arcbélyegkép-rács
    │   ├── log_panel.py      Színezett aktivitási napló
    │   └── preview_panel.py  Teljes képelőnézet bbox átfedéssel
    └── dialogs/
        ├── rename_dialog.py  Személy átnevezése párbeszédablak
        └── merge_dialog.py   Beolvasztás ide: ... párbeszédablak
```

---

## Követelmények

- **Python 3.11+**
- **Linux** (elsődleges célplatform — Ubuntu 22.04-en és Raspberry Pi OS Bookwormön tesztelve)
- **Google Coral USB Accelerator** *(opcionális)* — Edge TPU gyorsításhoz
- Kijelző / X11 vagy Wayland a GUI-hoz

A macOS és a Windows másodlagos célplatformok. A PySide6 és az OpenCV mindkettőn működik, de a pycoral hivatalosan csak Linuxot támogat.

---

## Telepítés

### 1. Klónozás és virtuális környezet létrehozása

```bash
git clone <repo-url> face-local
cd face-local
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Python-függőségek telepítése

```bash
pip install -r requirements.txt
```

> **Megjegyzés a tflite-runtime-ról:** A csomag Linux x86-64 és ARM platformokra érhető el.
> macOS-en vagy Windowson inkább a TensorFlow telepítése ajánlott: `pip install tensorflow`

### 3. Modellfájlok letöltése

#### Arcérzékelés (CPU) — Caffe SSD res10

```bash
mkdir -p models
wget -P models/ \
  https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt

wget -P models/ \
  https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
```

E fájlok nélkül a detektor az OpenCV Haar cascade megoldására áll vissza (gyengébb minőség, de mindig elérhető).

#### Arcbeágyazás — MobileFaceNet TFLite (csak CPU)

A beágyazó modell nincs benne a repóban. Három lehetőséged van:

**A lehetőség — Közösségi, előre konvertált modell** (legegyszerűbb):
```bash
# Keress rá GitHubon vagy Hugging Face-en erre: "mobilefacenet tflite".
# Egy gyakran használt változat a sirius-ai/MobileFaceNet_TF projektből származik.
# A letöltött .tflite fájlt ide helyezd:
cp /path/to/mobilefacenet.tflite models/mobilefacenet.tflite
```

**B lehetőség — Konvertáld át magad ONNX-ből**:
```bash
pip install onnx onnx-tf tensorflow
# Töltsd le a mobilefacenet.onnx fájlt az InsightFace model zoo-ból
# (https://github.com/deepinsight/insightface/tree/master/model_zoo)
# Majd konvertáld át — lásd: docs/convert_model.md (TODO: ezt az útmutatót még meg kell írni)
```

**C lehetőség — Használd a HOG csonkot (csak fejlesztéshez)**:
Hagyd üresen az `embedding.model_path` értékét a `config.yaml` fájlban. Az alkalmazás figyelmeztetést ad, és egy gyenge minőségű, HOG-alapú tartalékmegoldást használ. A pipeline működni fog, az arcfelismerés minősége viszont nem lesz megfelelő.

#### Arcérzékelés (Coral Edge TPU)

```bash
wget -P models/ \
  https://raw.githubusercontent.com/google-coral/test_data/master/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite
```

Ezután ezt állítsd be a `config.yaml` fájlban:
```yaml
detection:
  coral_model_path: models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite
```

### 4. pycoral telepítése (csak Coral felhasználóknak)

Kövesd a hivatalos útmutatót: https://coral.ai/docs/accelerator/get-started/

```bash
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
  | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update
sudo apt install libedgetpu1-std   # vagy libedgetpu1-max a maximális órajelhez
pip install pycoral
```

### 5. Konfigurálás

```bash
cp config.example.yaml config.yaml
# Szerkeszd a config.yaml fájlt — állítsd be a modellútvonalakat, küszöböket stb.
```

---

## Az alkalmazás futtatása

```bash
# Alapértelmezett (automatikusan megkeresi a config.yaml fájlt az aktuális mappában)
python -m app.main

# Konfigurációs fájl megadása
python -m app.main --config /path/to/config.yaml

# Hibakeresési naplózás
python -m app.main --debug

# Egyéni adatbázis-hely
python -m app.main --db /tmp/test.db
```

Vagy ha `pip`-pel van telepítve:

```bash
face-local --config config.yaml
```

---

## GUI munkafolyamat

1. **Mappa kiválasztása** — válaszd ki a képeket tartalmazó gyökérkönyvtárat.
2. **Beolvasás és indexelés** — lefuttatja a pipeline mind a 4 szakaszát:
   - Beolvasás → hashelés → DB beszúrás
   - Érzékelés → arc bbox → bélyegképkivágások
   - Beágyazás → arcvetktorok
   - Klaszterezés → személycsoportok
3. **Böngészés** — kattints egy személyre a bal oldali sávban az arcbélyegképek megtekintéséhez.
4. **Kattintás a bélyegképre** — megnyitja az eredeti képet kiemelt arccal.
5. **Átnevezés** — adj egy valós nevet a személyhez.
6. **Egyesítés** — vonj össze két klasztert, ha ugyanazt a személyt jelölik.
7. **Arc eltávolítása** — vedd ki a rossz arcot a klaszterből.
8. **Arc újra hozzárendelése** — helyezd át az arcot egy másik klaszterbe.
9. **Összes újraklaszterezése** — futtasd újra a DBSCAN-t a jelenlegi kézi korrekciók figyelembevételével.
10. **Exportálás** — CSV/JSON jelentés vagy képek másolása egy mappába.

---

## Coral vs CPU — mi a valóság, és mi nem

| Szakasz | Coral | CPU |
|-------|-------|-----|
| Arcérzékelés | ✅ Valódi pycoral integráció a `CoralDetector` segítségével | ✅ OpenCV DNN SSD + Haar tartalékmegoldás |
| Arcbeágyazás | ❌ Nincs használatban — nem érhető el gyakorlatban használható nyilvános Edge TPU ArcFace modell | ✅ MobileFaceNet TFLite a `TFLiteEmbedder` segítségével |
| Klaszterezés | ❌ Nem alkalmazható | ✅ scikit-learn DBSCAN |

**Várható teljesítmény:**
- Coral: kb. 50–200 ms képenként az érzékeléshez (USB Accelerator sebesség)
- CPU DNN: kb. 100–500 ms képenként, hardvertől függően
- Haar cascade: gyors, de az arcok kb. 30–40%-át kihagyhatja
- MobileFaceNet beágyazás: kb. 10–50 ms arckivágásonként modern x86 CPU-n

---

## Coral fallback tesztelése

Így ellenőrizheted a CPU-s tartalék útvonalat Coral hardver nélkül:

```bash
# 1. Ne telepítsd a pycoral csomagot — a factory automatikusan visszaáll CPU-ra:
python -m app.main

# 2. Telepített pycoral mellett, de USB Accelerator nélkül:
python -m app.main
# A naplóban ezt látod majd: "pycoral available but no Edge TPU devices found"

# 3. Állíts be coral_model_path értéket, de húzd ki az USB eszközt:
python -m app.main
# Ezt fogod látni: "Coral init failed: ... — falling back to CPU"

# 4. Futtasd a teszteket:
pytest tests/test_detectors.py -v
# A TestFactoryFallback monkeypatch segítségével ellenőrzi a CPU fallback útvonalat
```

---

## Tesztek futtatása

```bash
pytest -v

# Konkrét modulok
pytest tests/test_clustering.py -v
pytest tests/test_database.py -v
pytest tests/test_detectors.py -v
pytest tests/test_scan_service.py -v
```

A tesztekhez NEM szükséges valódi kamera, Coral hardver vagy modellfájl. Memóriában futó SQLite-adatbázisokat és szintetikus beágyazásokat használnak.

---

## Konfigurációs referencia

Az összes opcióért és a beágyazott dokumentációért lásd: [`config.example.yaml`](config.example.yaml).

Fontos küszöbértékek:

| Paraméter | Alapértelmezett | Hatás |
|-----------|------------------|-------|
| `detection.confidence_threshold` | `0.5` | Alacsonyabb érték → több találat, több hamis pozitív |
| `clustering.epsilon` | `0.4` | Alacsonyabb érték → szigorúbb egyezés, több klaszter |
| `clustering.min_samples` | `2` | Magasabb érték → az egyedülálló elemek zajnak számítanak |
| `scan.thumbnail_size` | `[128, 128]` | Nagyobb érték → jobb vizuális minőség, több lemezhasználat |

---

## Projektállapot

### Működő MVP-elemek
- Teljes pipeline: beolvasás → érzékelés → beágyazás → klaszterezés → GUI
- Coral-alapú érzékelés valódi pycoral integrációval
- CPU-s érzékelés: OpenCV DNN SSD + Haar cascade fallback
- TFLite beágyazás MobileFaceNettel (vagy HOG csonkkal)
- DBSCAN klaszterezés kézi korrekciós megkötésekkel
- SQLite perzisztencia SQLAlchemy ORM-mel
- PySide6 GUI: oldalsáv, klaszterrács, előnézet, átnevezés, egyesítés, eltávolítás, áthelyezés
- Export: CSV, JSON, képmappa
- Folytatható feldolgozás (kihagyja a változatlan fájlokat)
- Strukturált naplózás GUI naplópanellel

### Helykitöltő elemek / ismert korlátok
- A beágyazó modellfájlt külön kell letölteni (lásd: Telepítés)
- A CPU-s modellfájlokat (Caffe SSD) külön kell letölteni
- Nincs arckiegyenesítési lépés (a kivágások tengelyhez igazított téglalapok, nincs forgatáskorrekció)
- A klaszterezés teljes újrafuttatással történik (nem inkrementális)
- Az újraklaszterezés nem őrzi meg a klaszter↔személy megfeleltetést, ha sok klaszter változik
- A `worker_threads` konfiguráció létezik, de a pipeline jelenleg soros (a QThread egy szálon fut)
- Nincs HEIC támogatás (kiterjesztési pont van a `ScanConfig.image_extensions` mezőben)
- Nincs klaszter-szétválasztó művelet a GUI-ban (kerülőmegoldásként az arcokat egyesével lehet eltávolítani)

---

## Licenc

[KOAI Personal Use License v1.0](LICENSE)
