# Face-Local teljesítmény-audit

*Dátum: 2026-06-12 (frissítve: 2026-06-13) · Cél: 5000+ személy és több tízezer
kép mellett is folyamatosan használható, stabil, reszponzív alkalmazás.*

---

## 1. Összefoglaló

A vizsgálat szintetikus, méretezett tesztadatbázisokkal (100–5000 személy,
1 000–50 000 kép, 2 000–100 000 arc) mérte végig a fő műveleteket, majd a
feltárt szűk keresztmetszetekre célzott javítások készültek. A legnagyobb
gyorsulások az 5000 személyes / 100 000 arcos skálán (a „Utána” oszlop a
végleges állapot: fedő index + sidebar-átírás + **embedding külön táblába** +
**exportok nyers SQL-lel**):

| Művelet | Előtte | Utána | Gyorsulás |
|---|---:|---:|---:|
| Sidebar frissítés (minden arc-hozzárendelés után fut) | 17 560 ms | **90 ms** | **~195×** |
| Személyek lista (Személyek fül) | 8 959 ms | **171 ms** | **~52×** |
| Keresés (név szűrő) | 4 474 ms | **83 ms** | **~54×** |
| CSV export | 22 092 ms | **952 ms**¹ | **~23×** |
| JSON export | 19 735 ms | **1 354 ms**¹ | **~15×** |
| DB backup | 3 197 ms | ~1 900–3 500 ms¹ | háttérben |
| Settings ablak megnyitása (meleg) | ~1–3 s² | **~110–140 ms** | ~10–20× |
| Személy-szerkesztő megnyitása (meleg) | ~8–10 DB-lekérdezés szinkron | **2 ms** | azonnali |

¹ Ezek a műveletek mostantól **háttérfeladatként** futnak — a maradék idő a
felhasználót nem blokkolja. ² A régi Settings-nyitás idejét a szinkron
ffmpeg-eszközpróba (subprocess), a Drive-cache könyvtárbejárás és a DB-countok
dominálták; ezek gépfüggők.

Három új alrendszer / strukturális változás:

* **Embedding + landmark blobok külön `face_blobs` táblában** — a gyökérok
  végleges megszüntetése (3. szakasz). Ettől már a *régi* sidebar-útvonal is
  feleződött (8,9 s → 4,1 s), mert a `faces` tábla minden szkennelése kicsi
  maradt.
* **Egységes háttérfeladat-rendszer** (`app/tasks/`) — minden hosszú művelet
  (HTML/CSV/JSON/kép export, .facepack export/import, minőség-újraelemzés,
  **újraklaszterezés/felismerés**, crop-konzisztencia-ellenőrzés) ezen fut,
  szüneteltetés/leállítás támogatással.
* **Feladatkezelő ablak** — három helyről nyitható: toolbar „⚙ Feladatok”
  gomb, **bal-alsó státuszsor gomb** (mindig látható), és a jobb oldali
  „N feladat fut” chip (csak amíg fut feladat). Mutatja a futó/várakozó/
  befejezett feladatokat, progress-t, eltelt időt, CPU-időt, folyamat CPU/RAM-ot;
  szünet/folytatás/leállítás/újraindítás.

---

## 2. Mérési módszertan és eszközök (újak, maradandók)

| Eszköz | Mire való |
|---|---|
| `app/perf.py` | Futásidejű mérési pontok: `timed_block()`/`timed()` aggregált időmérés, lassú műveletek WARN-logja, SQL-lekérdezésenkénti időmérés (`attach_sql_timing`, az engine-be bekötve), cache hit-rate számlálók, RSS-memória. |
| `scripts/perf_seed.py` | Szintetikus DB-generátor: `python -m scripts.perf_seed --db x.db --persons 5000 --images 50000`. Valósághű 512-dim float32 embeddingekkel. |
| `scripts/perf_benchmark.py` | A UI-szálon futó munkák pontos másának mérése 4 skálán; Markdown riportot ír. `--quick` a két kis skálához. |
| `docs/perf_report.md` | „Előtte” mérés (a javítások előtti kódúton). |
| `docs/perf_report_after.md` | Köztes mérés (fedő index + sidebar-átírás, még inline blobokkal). |
| `docs/perf_report_v2.md` | Végleges mérés (blob külön táblában + nyers SQL exportok). |

A lassú SQL-ek mostantól futás közben is látszanak a logban
(`SLOW SQL 97 ms: SELECT …`), így a jövőbeli regressziók azonnal kiderülnek.

---

## 3. A gyökérok: az embedding blob a `faces` táblában van

A legfontosabb felfedezés: a `faces` tábla minden sora tartalmaz egy ~2 KB-os
embedding blobot (+ landmark blob). Emiatt **bármilyen teljes szkennelés a
táblán a teljes adatmennyiséget végigolvassa** (100 000 arcnál ~400 MB), akkor
is, ha a lekérdezés csak darabszámot vagy crop-útvonalat kér:

```
darabszám / legjobb-crop lekérdezések indexszel:   7 000–13 000 ms
ugyanezek fedő (covering) indexszel:                  10–60 ms
```

**Javítás 1 — fedő index** (`app/db/database.py`):

```sql
CREATE INDEX ix_faces_person_listing
ON faces(person_id, is_excluded, confidence DESC, crop_path, image_id)
```

Ez önmagában 100–200×-os gyorsulást hoz minden személy-aggregáló lekérdezésen,
mert az index a tábla (és a blobok) érintése nélkül válaszol.

**Javítás 2 — a blobok kiemelése külön táblába (végleges megoldás).** A blobok
átkerültek az 1:1 `face_blobs` táblába (`face_id`, `embedding`, `landmarks`);
a `faces.embedding`/`faces.landmarks` oszlopok megszűntek. Így már **bármely**
`faces`-szkennelés kicsi marad, nem csak a fedő index által lefedett
lekérdezések. Ettől a *régi* (ORM-gráfot betöltő) sidebar-útvonal is feleződött
(8,9 s → 4,1 s 100k arcnál), és a teljes-tábla exportok is gyorsultak.

* **Modell** (`app/db/models.py`): új `FaceBlob` modell + `Face.blob`
  reláció `lazy="selectin"` betöltéssel — egyetlen arc blob-elérése sosem
  fajul N+1-be, a blobot nem igénylő tömeges Face-betöltések pedig
  `lazyload(Face.blob)`-bal opt-outolnak. A `Face.get_embedding()/
  set_embedding()/get_landmarks()/set_landmarks()` API változatlan; SQL-szűrőre
  új `Face.embedding_exists()` (EXISTS a `face_blobs`-on) váltotta a régi
  `Face._embedding.isnot(None)` mintát minden hívóhelyen.
* **Migráció** (`_migrate_face_blobs`): egyszeri, idempotens — átmásolja a
  bájtokat a side-table-be, eldobja a régi oszlopokat (SQLite ≥ 3.35; ahol a
  DROP COLUMN nem elérhető, az adat már másolva van, a használaton kívüli
  oszlop bennmarad), majd `VACUUM`-mal visszanyeri a helyet. A 420 MB-os
  teszt-DB migrációja ~37 s (egyszeri), az ismételt indítás no-op (~17 ms).

---

## 4. Feltárt UI-blokkolások és javításaik

### 4.1 Sidebar frissítés — a legnagyobb probléma (17,6 s → 100 ms)

`MainWindow._refresh_persons` minden frissítéskor (minden egyes
arc-hozzárendelés, átnevezés, összevonás után!) betöltötte az **összes**
Person→Face→Image ORM-gráfot, embedding blobokkal együtt, majd a
`ensure_unique_face_crops` javítást is a UI-szálon futtatta.

**Javítás** (`app/ui/main_window.py`, `app/ui/panels/sidebar_panel.py`):
* A sidebar mostantól könnyűsúlyú `SidebarPerson` view-modelt kap (id, név,
  arcszám, reprezentatív arc) — 4 aggregáló lekérdezés a fedő indexen,
  ORM-gráf betöltés nélkül.
* A reprezentatív arc kiválasztása ablakfüggvénnyel történik (legjobb
  confidence személyenként; a kézi thumbnail-választást egy join őrzi meg).
* `ensure_unique_face_crops` (crop-konzisztencia önjavítás) áthelyezve
  **egyszeri, indításkori háttérfeladatba** — nem fut többé minden frissítésnél.

### 4.2 Személyek fül + keresés (9,0 s / 4,5 s → 229 / 82 ms)

`PersonService.list_persons` a tartalék-thumbnail kereséshez minden arc-sort
átrendezve áthúzott a Pythonba. **Javítás:** ablakfüggvényes lekérdezés a fedő
indexen (1 kis sor személyenként).

### 4.3 Kollázs HTML export (fagyások nagy adatnál)

Problémák: a teljes export a UI-szálon futott; node-onként és arconként külön
lekérdezések (N+1); az összes kollázs-rekord és a teljes JSON egyben épült a
memóriában.

**Javítás** (`app/services/export_service.py`, `app/ui/main_window.py`):
* Háttérfeladatként fut, progress-kijelzéssel és szüneteltetés/leállítás
  támogatással.
* **Streamelt írás**: a HTML fejléc egyszer íródik ki, utána kollázsonként egy
  JSON-rekord — egyszerre csak egyetlen kollázs van a memóriában.
* Kötegelt lekérdezések: képméretek és arcok (személynévvel) kollázsonként
  2 lekérdezésben a korábbi node-onkénti/arconkénti `session.get` helyett.
* Megszakítás/hiba esetén a félkész HTML törlődik.

### 4.4 CSV / JSON / kép export (N+1 + UI-szál)

Személyenként 1-2 lekérdezés + arconként lazy `image` betöltés (100 000 arcnál
~100 000+ lekérdezés). **Javítás:** egyetlen lekérdezés az összes arcra
(eager `image`, blobok kihagyva), csoporttagságok egy lekérdezésben; mindhárom
export háttérfeladatként fut.

### 4.5 .facepack projektcsomag export/import

`QProgressDialog` + `processEvents()` hack — modális, mégis akadozó.
**Javítás:** háttérfeladat; a Feladatkezelő automatikusan megnyílik, a UI
közben teljesen használható.

### 4.6 Settings ablak

Megnyitáskor szinkron futott: ffmpeg-eszközenumeráció (subprocess!), Drive
cache-méret (teljes könyvtárfa-bejárás), DB-countok, és mind a 8 fül felépült.

**Javítás** (`app/ui/dialogs/settings_dialog.py`):
* **Lazy fülek**: a Gyorsbillentyűk / Felvétel / Google Drive fül csak az első
  rákattintáskor épül fel (az `_on_accept` csak a ténylegesen felépült fülek
  értékeit menti).
* **Aszinkron statisztikák**: cache-méret + relatív/abszolút út-countok
  háttérszálon; az ffmpeg audio-eszközpróba háttérszálon, az eszközlista
  utólag olvad be a combókba a kiválasztás megőrzésével.
* **Minőség-újraelemzés** háttérfeladatként, progress-szel, leállíthatóan.
* QThread-életciklus javítás: a próbaszálak modul-szintű erős referenciát
  kapnak — gyorsan bezárt dialógusnál nem omlik össze az app.

### 4.7 Személy-szerkesztő (PersonInfoDialog)

Megnyitáskor ~8–10 szinkron lekérdezés (autocomplete DISTINCT-ek, csoportok,
objektumok, 2 kódvalidáció), és **minden billentyűleütés** szinkron
DB-validációt futtatott a családi kód mezőkben.

**Javítás:** az ablak azonnal megjelenik (2 ms); az autocomplete/csoport/
objektum adatok egyetlen háttérszálon töltődnek és utólag jelennek meg; a
kódvalidáció 350 ms-os debounce-szal fut gépelési szünetben.

---

## 5. Háttérfeladat-rendszer (app/tasks/)

Minden hosszú művelet közös kezelőn fut:

* `TaskManager.submit(név, fn, supports_pause=, on_done=, on_error=, on_cancelled=)`
* A feladatfüggvény `TaskContext`-et kap: `ctx.report(százalék, üzenet)`,
  `ctx.checkpoint()` (leállításnál kivételt dob, szünetnél blokkol).
* FIFO-sor, alapértelmezetten max. 2 párhuzamos feladat; a többi várakozik.
* A meglévő `CancellationToken` (szünet/folytatás/leállítás) primitívre épül.
* Kilépéskor figyelmeztetés, ha még fut feladat.

Jelenleg bekötött feladatok: kollázs HTML export, CSV/JSON/kép export,
.facepack export/import, arcminőség-újraelemzés, **újraklaszterezés/felismerés**
(`_on_recluster` — korábban a UI-szálon `processEvents`-szel futott),
crop-konzisztencia-ellenőrzés (indításkor). A meglévő pipeline/deep/match
workerek saját szálkezelésüket megtartották (már eddig is háttérben futottak).

### Feladatkezelő ablak

Három belépési pont: toolbar „⚙ Feladatok” gomb, a **bal-alsó státuszsor gomb**
(mindig elérhető), és a jobb oldali „N feladat fut” chip (csak amíg fut
feladat). Oszlopok: feladat, állapot, progress %, részlet, indulás, eltelt idő,
CPU-idő; fejlécben a folyamat CPU%/RAM (psutil — a requirements közé felvéve).
Műveletek: szünet/folytatás (ha a feladat támogatja), leállítás, újraindítás.
Minden felirat EN+HU.

---

## 6. Mentési stratégia — audit eredménye

**A felhasználói módosítások már most is azonnal, tranzakciónként mentődnek.**
Minden szerkesztő művelet (személy létrehozás/módosítás/törlés, arc-
hozzárendelés, kapcsolat, címke, csoport, beállítás) `session_scope()`-on
keresztül fut, ami a művelet végén azonnal commitol. A „Mentés” gombok
ugyanezt a commit-utat hívják — UX-elem maradnak, de nem kizárólagos mentési
mechanizmusok. Adatvesztési kockázat nincs:

* SQLite **WAL** mód + `synchronous=NORMAL` + 5 s `busy_timeout` — commit után
  az adat áramszünet-állóan lemezen van.
* Egyetlen személy-szerkesztés commitja 100 000 arcos DB-nél is 2–17 ms —
  nem blokkolja a UI-t.
* A teljes biztonsági mentés (.facepack, SQLite online-backup API-val) most
  már háttérfeladat.
* A QSettings-értékek `sync()`-kel, explicit flush-sal íródnak.

---

## 7. Mérési részletek

### Előtte (docs/perf_report.md)

| művelet (ms) | 100p/1k | 500p/5k | 1000p/10k | 5000p/50k |
|---|---:|---:|---:|---:|
| indulás: init_db + migrációk | 5 | 6 | 7 | 37 |
| sidebar frissítés (régi) | 38 | 291 | 611 | 17 560 |
| személyek lista | 9 | 48 | 87 | 8 959 |
| keresés | 6 | 25 | 49 | 4 474 |
| személy-dialógus (DB-munka) | 4 | 3 | 3 | 6 |
| settings (DB-countok) | 5 | 3 | 5 | 31 |
| JSON export | 289 | 1 404 | 2 855 | 19 735 |
| CSV export | 284 | 1 355 | 2 778 | 22 092 |
| DB backup | 29 | 118 | 301 | 3 197 |
| egy személy-szerkesztés mentése | 2 | 1 | 1 | 8 |

### Utána — végleges (docs/perf_report_v2.md: blob külön tábla + nyers SQL export)

| művelet (ms) | 100p/1k | 500p/5k | 1000p/10k | 5000p/50k |
|---|---:|---:|---:|---:|
| indulás: init_db + migrációk | 8 | 27 | 16 | 55 |
| sidebar frissítés (régi ORM-út, össze­hasonlításra) | 61 | 413 | 742 | 4 071 |
| sidebar frissítés (új, aggregált) | 2 | 10 | 19 | **90** |
| személyek lista | 6 | 23 | 37 | **171** |
| keresés | 4 | 12 | 18 | **83** |
| személy-dialógus (DB-munka) | 4 | 5 | 5 | 6 |
| settings (DB-countok) | 5 | 5 | 4 | 9 |
| JSON export (háttérben) | 199 | 243 | 272 | **1 354** |
| CSV export (háttérben) | 69 | 136 | 246 | **952** |
| DB backup (háttérben) | 98 | 380 | 963 | 3 501 |
| egy személy-szerkesztés mentése | 10 | 4 | 4 | 7 |

A blob-kiemelés után a *régi* ORM sidebar-út is feleződött (8,9 s → 4,1 s),
mert a `faces` tábla minden szkennelése kicsi maradt. A nyers SQL exportok az
ORM-materializációt is megszüntetik: CSV 952 ms (<1 s), JSON 1,35 s.

---

## 8. Megvalósított (második kör) + maradék javaslatok

A korábbi „további javaslatok" lista nagy részét megvalósítottuk:

1. ✅ **Embedding/landmark blobok külön táblában** (`face_blobs`) — 3. szakasz.
2. ✅ **Újraklaszterezés/felismerés háttérfeladatba** (`_on_recluster`).
3. ✅ **Exportok nyers SQL-lel** — CSV 952 ms (<1 s), JSON 1,35 s.
4. ✅ **Sidebar-thumbnailek viewport-alapú lazy betöltése** — a `_PersonThumb`
   placeholderként jön létre, a crop-kép csak a látható ablakba görgetéskor
   töltődik (40 ms debounce, ±160 px puffer). 5000 személy `populate`-je
   ~370 ms, és csak ~18 kép töltődik be azonnal a teljes 5000 helyett.
5. ✅ **psutil a követelmények közé** — a Feladatkezelő CPU/RAM kijelzéséhez.

Maradék (nem kötelező) javaslatok:

* **Astro export ajánlása nagy galériákhoz**: a régi egyfájlos HTML export
  helyett nagy adatnál a már meglévő, lapozó/lazy Astro export a skálázható út.
* **Időszakos WAL-checkpoint** hosszú futás után (a WAL-fájl növekedésének
  kordában tartására) — alacsony prioritás.
* **DB backup** (sqlite backup API) nagy DB-nél másodperces — már háttérben
  fut, de érdemes lehet a `.facepack`-be tömörített, inkrementális mentésre
  váltani.

---

## 8b. Adversariális verifikáció (a 2. kör után)

A magas kockázatú `face_blobs` migrációt és a háttérfeladat-szálkezelést egy
többügynökös, adversariális ellenőrzés vizsgálta (5 dimenzió: migráció-helyesség,
embedding-accessor sweep, EXISTS-szűrő ekvivalencia, export-kimenet egyezőség,
UI/szálkezelés). 5 megerősített találat — mind kijavítva és teszttel igazolva:

1. **Migráció részleges DROP COLUMN** (high): a régi kód az első sikertelen
   oszlop-dobásnál `break`-elt, ami pre-3.35 SQLite-on aszimmetrikus, minden
   indításkor újrapróbálkozó állapotot hagyhatott. Javítva: **oszloponként
   független dobás** (`continue`), a VACUUM csak ha minden eldobás sikerült.
   Szintetikus régi-sémájú DB-n ellenőrizve (mindkét oszlop eldobva, helyes
   `face_blobs` sorok, bájt-pontos visszaolvasás, 2. indítás no-op).
2. **Hiányzó `ctx.checkpoint()`** az újraklaszterezés feladatban (high) —
   hozzáadva, így a sorban állás közben kért leállítást észleli.
3+4. **Task-callback szálbiztonság** (high/medium): a `finished` jelet a worker
   szál bocsátja ki; a callbackek widgeteket módosítanak. Mostantól explicit
   **`Qt.QueuedConnection`** garantálja, hogy az `on_done`/`on_error` mindig a
   UI-szálon fut (teszttel igazolva: `done_on_ui=True`).
5. **`closeEvent` nem várta meg a leállított feladatokat** (low): a `cancel_all`
   után `TaskManager.wait_for_idle(timeout_s=5)` korlátozott ideig megvárja a
   worker szálak tiszta kilépését (nincs „QThread destroyed while running”).

---

## 9. Érintett fájlok

**Új:** `app/perf.py`, `app/tasks/__init__.py`, `app/tasks/manager.py`,
`app/ui/dialogs/task_manager_dialog.py`, `scripts/perf_seed.py`,
`scripts/perf_benchmark.py`, `docs/performance_audit.md`,
`docs/perf_report*.md`.

**Módosított:** `app/db/database.py` (fedő index + SQL-időmérés +
`_migrate_face_blobs`), `app/db/models.py` (`FaceBlob` modell, `Face.blob`
reláció, `embedding_exists()`), `app/ui/main_window.py` (könnyű
sidebar-frissítés, háttérfeladatok, recluster háttérbe, toolbar + bal-alsó
státuszsor gomb + chip, kilépésvédelem), `app/ui/panels/sidebar_panel.py`
(SidebarPerson VM + viewport-lazy thumbnailek), `app/services/person_service.py`
(ablakfüggvényes fallback-crop), `app/services/export_service.py` (streamelt
HTML export + nyers SQL CSV/JSON), `app/services/embedding_service.py` +
~7 további service (`embedding_exists()` szűrő, `FaceBlob` írás),
`app/services/face_quality_service.py` (`lazyload(Face.blob)`),
`app/ui/dialogs/settings_dialog.py` (lazy fülek, aszinkron próbák, szál-élettartam),
`app/ui/dialogs/person_info_dialog.py` (aszinkron betöltés, debounce-os
validáció), `app/ui/dialogs/export_dialog.py` (háttér-exportok),
`app/ui/i18n.py` (EN+HU feliratok), `requirements.txt` (psutil),
`tests/test_family_code_schemes.py` (debounce-flush a tesztekben).

Tesztek: **1208 passed** (a teljes meglévő tesztkészlet zöld). A `face_blobs`
migrációt 420 MB-os, régi sémájú DB-n is leellenőriztük (100 000 blob átmozgatva,
idempotens, EXISTS-szűrő és `get_embedding()` helyes).
