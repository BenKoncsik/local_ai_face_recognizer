# Arc Felismerés Működése

## Összefoglalás

Az arc felismerési rendszer egy **6 fázisú pipeline**, amely képeket feldolgoz és az arcokat az ismert személyekhez rendeli hozzá:

1. **Detektálás** — arcok keresése képeken
2. **Ellenőrzés** — hamis pozitívok szűrése
3. **Igazítás & Crop** — arc normalizálása
4. **Embedding** — arc vektorizálása
5. **Felismerés** — MLP ensemble osztályozás
6. **Hozzárendelés** — review-re váró "Auto Assignment" javaslatok

---

## 1. Detektálás (Detection Service)

A rendszer **3 szintű fallback** detektort használ:

### Szintek sorrendje:

1. **Coral Edge TPU** (GPU gyorsítás, ha van)
   - Modell: `SSD MobileNet` (Coral-fordított)
   - Fájl: `models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite`
   - Sebességi előny: ~10× gyorsabb, mint CPU

2. **YuNet / InsightFace SCRFD** (CPU, de landmark-pontokkal)
   - Modell: YuNet ONNX (`face_detection_yunet_2023mar.onnx`)
   - Modell: InsightFace SCRFD (`detection_onnx.onnx`)
   - 5 landmark pont → arc igazítás lehetséges
   - Erősség: profilból nézett arcok, torz szögek

3. **Caffe SSD / Haar Cascade** (Alap CPU fallback)
   - Modell: `res10_300x300_ssd_iter_140000.caffemodel` (OpenCV bundled)
   - Modell: Haar Cascade (OpenCV beépített)
   - Mindig működik, nincs extra letöltés

### Paraméterek:

```yaml
detection:
  confidence_threshold: 0.65      # Alapértelmezett detektálás küszöb
  min_face_size: 50               # Minimum pixel szélesség & magasság
  verification_enabled: true      # Hamis pozitívok szűrése
  adaptive_escalation: true       # Lazított keresés, ha 0 arc találat
```

---

## 2. Ellenőrzés (Verification Gate)

Hamis pozitívok (fülek, nevek, kéz, fa-textúra) **szűrése** három szintű ellenőrzéssel:

### A. YuNet Verification (Landmark Geometry)
- A kinyert crop-ot **felnagyítjuk** (min 160px oldal)
- **YuNet újra-detektál** az arc-crop-ra
- Ha az arc **0.55+ pontszámmal** található → igaz arc
- **Landmark geometria validáció**: szem-szem fölött a nose, nose fölött a száj
- Interpupil distance (IOD) sane-e, szem-vonal nincs-e megdöntve

### B. Multi-stage Voting
- Több független detektor (Haar, Eye Cascade) szavaz az arc valódiságára
- Min **2 szavazat szükséges** az arc elfogadásához
- Egy technológia által talált arc ~97% hamis, de 2+ technológia = igaz arc

### C. Confidence Gates
```
1. Outlier gate      → Embedding hasonlít-e VALAMELYIK tanítási archoz?
2. Probability gate  → Ensemble probabilitás > person-specifikus küszöb?
3. Prototype gate    → Cosine hasonlóság > person-specifikus padló?
```

Ha **bármely kapu elutasít** → arc `Unknown` marad.

---

## 3. Igazítás & Crop-készítés (Alignment)

### Landmark-alapú Affine Transzformáció

1. **YuNet / InsightFace adja** az 5 facial landmark pontot (szem, orr, száj)
2. **Affine mátrix** számítása: a szem-pontok a **standardizált helyre** kerüljenek
3. A képet **forgatjuk, skálázzuk** az arcnak az igazított helyzetbe
4. **112×112 pixel standardizált crop** (az embedding modell inputja)

### Fallback Mód
Ha nincs landmark (Caffe/Haar fallback): **egyszerű center-crop**, némi padding-gel.

---

## 4. Embedding (Arc Vektorizálása)

Az igazított arc-crop **192 dimenziós vektor** (vagy 128D, az **L2-normalized**):

### Szintek:

1. **TFLite Embedder** (elsődleges)
   - Modell: `MobileFaceNet` (112×112 input)
   - Kimenet: 192D L2-normalized vektor
   - Méret: ~2 MB
   - Függőség: `ai-edge-litert` vagy `tflite-runtime`

2. **SFace Embedder** (fallback)
   - Modell: `sface.onnx` (~37 MB, OpenCV Zoo)
   - Kimenet: 128D L2-normalized vektor
   - Függőség: OpenCV + ONNX runtime
   - `cv2.FaceRecognizerSF` beépített OpenCV-ben

3. **HOG+PCA Stub** (noodstraat fallback)
   - **Determinisztikus**, lassú, alacsony minőség
   - Nem ajánlott produkció-hoz
   - Kimenet: 192D L2-norm
   - Semmilyen extra csomag nem szükséges

### Model Letöltés
```bash
# TFLite (MobileFaceNet) — automatikus letöltés az első futtatáskor
scripts/build_and_run.sh

# SFace (optional, ha TFLite nem elérhető)
curl -L https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx \
     -o models/sface.onnx
```

---

## 5. Felismerés (Deep Recognition)

### MLP Ensemble Classifier

Az arc **192D embedding** bementként egy **4-5 neural network ensemble**-nek:

1. **Tanítási adat**: Minden megjelölt arc (TRUSTED labels)
2. **Ensemble tagok**: 4-5 MLP (multi-layer perceptron)
   ```
   Rétegek: 192 → 256 → 192 → 128 → 64 → 48 → 32 → [num_persons]
   ```
3. **Outputs átlaga** → pontszám & confidence per személy

### 3 Független Elutasítási Kapu

| Kapu | Logika | Elutasítás oka |
|------|--------|---|
| **Outlier** | Embedding cosine-közel van-e VALAMELYIK tanítási archoz? | Stranger, non-face → Unknown |
| **Probability** | Ensemble valószínűség > person-specifikus küszöb (CV-ből)? | Bizonytalanság → Unknown |
| **Prototype** | Cosine szim > person-specifikus padló (hardcoded prototypes)? | Nem elég hasonlít az ismert arcokhoz → Unknown |

### Kimenet: `DeepPrediction`

```python
{
    person_id: int,           # None = Unknown (elutasított)
    person_name: str,
    score: float,             # Blended confidence [0, 1]
    probability: float,       # Raw ensemble valós
    similarity: float,        # Best cosine sim a nyerteshez
    margin: float,            # Gap a 2. helyhez
    reason: str,              # "assigned" | "rejected_outlier" | "rejected_threshold" | ...
}
```

### Tanítás (Continual Learning)

- **Minden futtatásnál** újratanítás az **összes megjelölt arcra**
- `FaceCorrection` táblázat: "más ember" veto → modell továbbfejlesztés
- Fingerprint ellenőrzés: tanítási adatok nem változtak → újrafelhasználás

---

## 6. Automatikus Hozzárendelés (AutoAssignment)

### Tábla: `auto_assignment`

```python
{
    face_id: int,
    person_id: int,           # Javasolt személy
    score: float,             # 0-1 confidence
    status: str,              # "auto" (új) → "confirmed" (elfogadott) → "corrected" (javított) → "reverted" (vissza)
    run_id: int,              # Melyik TrainingRun-ból
    decided_at: datetime,     # Felhasználói jóváhagyás időpontja
    correction_to_person_id: int,  # Ha "corrected": helyes személy
    decision: JSON,           # Teljes döntésgraf
}
```

### Felhasználói Actions

| Akció | Kimenet |
|-------|---------|
| ✓ **Confirm** | status=confirmed; új "trusted" tanítási adat a modellnek |
| ✓✓ **Confirm + Merge** | status=confirmed + merge unknown arc-ket |
| ✗ **Correct** | status=corrected; person_id javítva; FaceCorrection constraints |
| 🔄 **Revert** | status=reverted; arc = Unknown vissza |
| 🗑 **Delete** | Arc fizikailag törlve |

---

## Szükséges Python Csomagok

### Kötelező Függőségek

| Csomag | Verzió | Célja | Fallback | Megjegyzés |
|--------|--------|-------|---------|-----------|
| **numpy** | ≥1.24, <2.0 | Numerikus számítások | — | Alapkövetelmény |
| **opencv-python** | ≥4.8 | Képfeldolgozás, Haar/Caffe/YuNet/SFace | — | |
| **Pillow** | ≥10.0 | Image I/O | — | |
| **PyYAML** | ≥6.0 | Config.yaml parser | — | |
| **piexif** | ≥1.1 | EXIF metadata | — | |
| **SQLAlchemy** | ≥2.0 | ORM database | — | |
| **scikit-learn** | ≥1.3 | Clustering, PCA | — | |
| **PySide6** | ≥6.6 | GUI (Qt6) | — | |

### Embedding Runtime

| Csomag | Verzió | Célja | Fallback | Telepítés |
|--------|--------|-------|---------|-----------|
| **ai-edge-litert** | ≥1.0 | TFLite runtime (MobileFaceNet) | tflite-runtime / tensorflow | `pip install ai-edge-litert` |
| **tflite-runtime** | — | Lightweight TFLite fallback | tensorflow | `pip install tflite-runtime` |
| **tensorflow** | — | Heavy (~500 MB) ultimátum fallback | — | `pip install tensorflow` |

**Jelenlegi konfig:** `ai-edge-litert` telepítve (javasolt).

### Arc Detektálás - Opcionális

| Csomag | Verzió | Célja | Fallback | Telepítés |
|--------|--------|-------|---------|-----------|
| **pycoral** | — | Coral Edge TPU support | YuNet/Caffe | [Coral official guide](https://coral.ai/docs/accelerator/get-started/) |
| **insightface** | — | InsightFace SCRFD co-detector | YuNet/Caffe | `pip install insightface onnxruntime` |
| **onnxruntime** | — | ONNX model inference | — | `pip install onnxruntime` |

**Jelenlegi konfig:** `insightface` **nincs** telepítve → fallback YuNet/Caffe-re.

### Google Drive (Opcionális)

| Csomag | Verzió | Célja | Telepítés |
|--------|--------|-------|-----------|
| **google-auth** | ≥2.0 | Google auth | `pip install google-auth` |
| **google-auth-oauthlib** | ≥1.0 | OAuth flow | `pip install google-auth-oauthlib` |
| **google-api-python-client** | ≥2.0 | Drive API | `pip install google-api-python-client` |
| **keyring** | ≥24.0 | OS keychain (Keychain/Credential Manager) | `pip install keyring` |

**Jelenlegi konfig:** Local-only mód → nem szükséges.

### Exportálás (Opcionális)

| Csomag | Verzió | Célja | Telepítés |
|--------|--------|-------|-----------|
| **openpyxl** | ≥3.1 | Excel (.xlsx) export | `pip install openpyxl` |

---

## Fallback Logika (Hiányzó Komponensek)

### Forgatókönyv 1: Coral nincs, YuNet van

```
Detektálás: Coral fallback → YuNet (ONNX modell van) ✓
Embedding: TFLite van ✓
Felismerés: MLP ensemble működik ✓
```

**Eredmény:** ~95% teljes funkció, CPU-alapú, jellegzetes sebesség, no GPU.

### Forgatókönyv 2: Coral nincs, YuNet nincs, Caffe van

```
Detektálás: Coral fallback → YuNet fallback → Caffe ✓
Embedding: TFLite van ✓
Verification: 1 szintű (YuNet landmark nélkül) ⚠
Felismerés: MLP ensemble működik ✓
```

**Eredmény:** Alapfunkció működik, de landmark-alapú igazítás nincs.

### Forgatókönyv 3: Caffe sincs (OpenCV bundled)

```
Detektálás: Coral fallback → YuNet fallback → Caffe fallback → Haar Cascade ✓
Embedding: TFLite van ✓
Felismerés: MLP ensemble működik ✓
```

**Eredmény:** Működik, de Haar lassabb, kevesebb arc.

### Forgatókönyv 4: TFLite nincs, SFace van

```
Detektálás: … ✓ (tetszőleges)
Embedding: TFLite fallback → SFace ✓
Felismerés: MLP ensemble működik ✓ (128D-re átméretezve)
```

**Eredmény:** Működik, 128D embedding helyett 192D.

### Forgatókönyv 5: TFLite sincs, SFace sincs, HOG+PCA Stub

```
Detektálás: … ✓
Embedding: TFLite fallback → SFace fallback → HOG+PCA Stub ✓
Felismerés: MLP ensemble működik ✓ (alacsony minőség)
```

**Eredmény:** Működik, **de felismerési pontosság nagyon csökkent** (~60%).

---

## Telepítési Útmutató

### Alapszintű (Kötelező)

```bash
pip install -r requirements.txt
```

### Coral Edge TPU hozzáadása (Linux)

```bash
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update
sudo apt install libedgetpu1-std  # vagy libedgetpu1-max
pip install pycoral
```

### InsightFace opcionális (jobb profilmeghatározás)

```bash
pip install insightface onnxruntime
# Megjegyzés: pip előfordulhat numpy≥2-t húz → pin back
pip install "numpy<2.0"
```

### SFace modell letöltése (opcionális fallback)

```bash
curl -L https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx \
     -o models/sface.onnx
```

---

## Konfiguráció (config.yaml)

```yaml
detection:
  confidence_threshold: 0.65          # Detektálás küszöb
  min_face_size: 50                   # Min pixel szélesség/magasság
  coral_model_path: models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite
  cpu_model_path: models/res10_300x300_ssd_iter_140000.caffemodel
  yunet_model_path: models/face_detection_yunet_2023mar.onnx
  use_yunet: true                     # YuNet előnykészítés (ha van)
  verification_enabled: true          # Hamis pozitívok szűrése
  adaptive_escalation: true           # Lazított keresés, ha 0 arc

embedding:
  model_path: models/mobilefacenet.tflite
  input_size: [112, 112]
  embedding_dim: 192

deep_recognition:
  hidden_layers: [256, 192, 128, 64, 48, 32]
```

---

## Összefoglalás Táblázat

| Komponens | Elsődleges | Fallback 1 | Fallback 2 | Fallback 3 |
|-----------|-----------|-----------|-----------|-----------|
| **Detektálás** | Coral Edge TPU | YuNet/InsightFace | Caffe SSD | Haar Cascade |
| **Embedding** | TFLite MobileFaceNet | SFace (OpenCV) | HOG+PCA Stub | — |
| **Verification** | YuNet 5-point + Haar voting | YuNet no-landmark | Caffe-only | — |
| **Felismerés** | MLP Ensemble (4-5 network) | — | — | — |
| **Tanítás** | Continual Learning | — | — | — |

---

## Hibakeresés

### "No embedding for face X"
→ TFLite/SFace/HOG lehetett hibát; ellenőrizd a `models/` mappát.

### "Recognition takes forever"
→ CPU-t használod (Coral nincs); normális. GPU-val 10× gyorsabb.

### "Too many false positives"
→ `verification_enabled: true` → `adaptive_escalation` csökkentse az ionót.

### "Unknown person recognition errors"
→ Check `deep_recognition.hidden_layers` ensemble sizeá; lehet egyedül gyenge.
