# 🧠 AZ ÚJ ARCFELISMERÉSI ÉS MODELLTANÍTÁSI RENDSZER

Az új rendszer egy **MLP ensemble-alapú mélytanulás motor**, amely neurális hálózatot tanít a cimkézett arcokra, majd ezzel azonosítja az ismeretleneket. Íme a teljes folyamat:

---

## 📋 FŐBB ÜZEMMÓDOK

| Mód | Mi történik | Lépések |
|-----|------------|--------|
| **rescan** | Új képek beolvasása, feldolgozása, tanítás, felismerés | 10 |
| **rebuild** | Teljes adatbázis újraépítése nulláról (de manual/jóváhagyott arcok megmaradnak) | 11 |
| **train** | Csak beágyazás és modelltanítás (nincs felismerés) | 2 |
| **rebuild_model** | Modell törlése + újratanítás + felismerés | 3 |
| **detect_faces** | Csak arcdetektálás analízishez (nincs arcfelismerés) | 1 |

---

## 🔄 A RESCAN/REBUILD PIPELINE SORRENDJE (10-11 lépés)

### **FASE 1: ADAT ELŐKÉSZÍTÉS**

#### 1. **Scan** (`_run_scan`)
- Új képek keresése a mappákban
- Adatbázisba regisztrálás

#### 2. **Detection** (`_run_detection`)
- Arcok detektálása minden képből → bounding boxok
- Hierarchikus: InsightFace (elsősorban) → YuNet (fallback) → Coral TPU (opcionális)
- Minden arc egy Face rekord az adatbázisban

#### 3. **AI Face Detection** (`_run_ai_face_detection`)
- Elemzési célú arcdetektálás (opcionális)
- Csak az `ai_face_detections` táblában rögzít, nem módosít Face rekordokat

#### 4. **Embedding** (`_run_embedding`)
- Minden Face → 128 vagy 256-dimenziós vektor (SFace vagy TFLite)
- Az Embedding vektor az arc neurális "ujjlenyomata"

#### 5. **Overlap Resolution** (`_run_overlap_resolution`)
- Ha egy Box két arc magas átfedése van (> 70%), az egyik törlődik
- **Szabály:** Ha egyik arc már hozzárendelve van valakihez → az marad, a másik törlődik
- Ha egyik sem → a magasabb konfidenciájú marad

#### 6. **Multi-stage Cleanup** (`_run_multistage_cleanup`)
- **Cél:** Hamis pozitívok eltávolítása (hajfonat, fül, textúra)
- **Csak alacsony konfidenciájú arcok** (felhasználó-jóváhagyott arcok érintetlenek)
- **Technika:** Több detektor ensemble (InsightFace + YuNet + Coral) véleménye
- Ha a többség "nem arc" → törlés

#### 7. **Ignored Filter** (`_run_ignored_filter`)
- Permanensen ignorált arcok szűrése
- Embedding alapú: már korábban "ignore" jelölt arcokhoz hasonlók

### **FASE 2: MÉLYTANULÁS (TANÍTÁS + FELISMERÉS)**

#### 8. **Train & Recognize** (`_run_deep_train_and_recognize`)

##### **8a) TANÍTÁS** (`DeepRecognitionService.train`)

```
1. Tanítási adathalmaz építése
   - Minden CIMKÉZETT arc (embere van) betöltése
   - Források: manual, deep_confirmed, auto_trained
   - Minimum confidence szűrés (alapér: 0.5)

2. Modell újrahasználhatóság ellenőrzése
   - Ha az adat AZONOS az utolsó tanításóta → RÉGI modell felhasználása
   - Hash (fingerprint) összehasonlítása
   - Gyorsítás: csak 1-2 másodperc helyett 5-10 perc

3. Ha újra kell tanítani:
   - MLP ensemble építése (tipikus: 3-5 neurális háló)
   - Adataugmentáció: szintetikus minták (pixelshuffle)
   - Tanítás: cross-entropy loss + weight decay
   - Validáció: holdout set, accuracy kalkulálás (70-95% tipikus)
   - Modell mentése: data/deep_model/deep_face_model.pkl

Kimenet: TrainingRun rekord (személyek száma, arcok száma, pontosság)
```

##### **8b) FELISMERÉS** (`DeepRecognitionService.recognize`)

```
1. Jelölt arcok betöltése
   - "még ismeretlen" arcok (person_id == NULL)
   - Már cimkézett arcok érintetlenek!

2. Minden ismeretlenhez egymás után:
   - Min confidence check (alapér: 0.4)
     └─ Ha detektálási konfidencia < 0.4 → skip
   
   - Embedding ellenőrzése
     └─ Ha nincs beágyazás → skip
   
   - MLP predict
     └─ Forward pass a tanított modellon
     └─ Kimenete: person_id, confidence, vétó oka
   
   - VÉTÓ ellenőrzések (elutasítás okai):
     ├─ Outlier detection: "idegennek tűnik" (open-set gate)
     ├─ Below threshold: valószínűség < 0.6 (alapér)
     ├─ Margin: 2-3 legjobb illeszkedés túl közeli
     ├─ Prototype: olyan személy, akinek kevés tanítási adata van
     └─ User veto: felhasználó korábban "különbözőnek" jelölte
   
   - Stale model check
     └─ A prediktált személy még létezik-e az adatbázisban?
   
   - ASSIGNMENT: Ha mindent átmegy
     └─ Face.person_id = predicted_person_id
     └─ AutoAssignment rekord rögzítés
        (status="auto", score, decision_json, assignment_source="deep_recognition")
```

**Kimenet:** DeepRecognitionStats (felismerések száma + elutasítási okok)

---

### **FASE 3: CSOPORTOSÍTÁS & KONZISZTENCIA**

#### 9. **Clustering** (`_run_clustering`)
- A még **mindig ismeretlen** arcok csoportosítása
- DBSCAN algoritmus, cosine distance
- Automatikus személyek létrehozása: "Unknown-1", "Unknown-2" stb.

#### 10. **Intra-image Consistency** (`_run_intra_image_consistency`)
- Egyazon képen belül egy arc nem lehet két személyhez rendelve
- Ellentmondások javítása (pl.: "Anna" és "Anna (1)")

#### 11. **Suggestions** (`_run_suggestions`)
- Merge javaslatok: hasonló arcok csoportjai
- Névegyezőségi javaslatok: "Péter" több személyhez kapcsolódna

---

## 🧬 A TANÍTÁSI ADATHALMAZ FORRÁSOK

```python
# Mely arcok tanítanak?
TRUSTED_SOURCES = [
    "manual"           # Felhasználó által kézzel rajzolt arc
    "deep_confirmed"   # AI javasolt, felhasználó jóváhagyott
    "auto_trained"     # Earlier successful AI assignments
]

# Az adathalmaz építés:
dataset = build_training_dataset(
    session,
    auto_min_confidence=0.5,
    use_auto_assignments=True,  # Korábbi sikeres AI javaslatok
    exclude_low_quality=False   # Magas minőség szűrés (opcionális)
)
```

---

## 🔐 BIZTONSÁGI SZABÁLYOK

1. **Cimkézett arcok érinthetetlen:** Ha egy arc már "Anna"-hoz van rendelve → nem kerül át máshoz
2. **Nyitott halmaz elutasítás:** Az MLP kimenete 3 választási lehetőség:
   - ✅ Ismert személy (> threshold)
   - ❌ Idegennek tűnik (outlier gate)
   - ❓ Bizonytalan (< threshold vagy margin)
3. **User veto:** "Ez NEM Anna!" → hard vetó, modell nem felülbírálhatja
4. **Stale model:** Ha a modell egy már törölt személyre hivatkozik → assignment nem jön létre

---

## 📊 OUTPUT: MI JÖN LE AZ EGÉSZBŐL?

```
✅ TrainingRun
   - trained_on: 150 arc (45 személy)
   - validation_accuracy: 92.3%
   - augmented_samples: 450 (szintetikus)
   - duration: 4.2 sec

✅ AutoAssignment (10 arc felismerve)
   - Anna: 3 arc (score: 0.87, 0.91, 0.85)
   - Bob: 5 arc (score: 0.92, 0.88, ...)
   - Charlie: 2 arc (score: 0.94, 0.90)

⚠️  Elutasítások:
   - Outlier (idegenek): 15 arc
   - Below threshold: 8 arc
   - User vetoed: 2 arc

🆕 Unknown groups (DBSCAN):
   - Unknown-1: 12 arc
   - Unknown-2: 7 arc
```

---

## ⚡ A RENDSZER ADAPTIVITÁSA

- **Continual learning:** Minden alkalommal amikor jóváhagyod egy AI-javaslatot, az bekerül a tanítási adatba
- **Self-correcting:** User "ezt nem Anna" veto → modell tanul belőle
- **Graceful degradation:** Ha nincs tanítási adat → nem készít assignmenteket
- **Multi-technology:** Overlap resolution és cleanup több detektort használ, nem csak egyet

---

## 📁 FONTOSABB FÁJLOK

| Fájl | Szerepe |
|------|--------|
| `app/workers/deep_pipeline_worker.py` | Orkestrálás: scan → detect → embed → train → recognize → cluster |
| `app/services/deep_recognition_service.py` | Training és Recognition logika |
| `app/deep/classifier.py` | MLP ensemble implementáció |
| `app/deep/dataset.py` | Tanítási adathalmaz építés |
| `app/services/embedding_service.py` | Arcbeágyazás (SFace/TFLite) |
| `app/services/detection_service.py` | Arcdetektálás |
| `app/services/clustering_service.py` | DBSCAN csoportosítás |
| `app/services/overlap_resolution_service.py` | Átfedő dobozok feloldása |
| `app/db/models.py` | Face, TrainingRun, AutoAssignment, FaceCorrection táblák |

---

Összefoglalva: az új rendszer egy **iteratív, megtanuló arcfelismerési motor**, amely tanul a felhasználó döntéseiből és folyamatosan javul.
