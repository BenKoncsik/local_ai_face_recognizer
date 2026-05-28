---
name: recognition-pipeline-improvements
description: Two-pass recognition with adaptive threshold and same-image context assist — added 2026-05-28
metadata:
  type: project
---

## Két lépéses recognition pipeline javítás

**Implementálva:** 2026-05-28 (commit-ot még kell létrehozni)

**Érintett fájlok:**
- `app/config.py` — 6 új `RecognitionConfig` mező + load_config frissítés
- `app/services/recognition_service.py` — teljes refaktor
- `tests/test_recognition_service.py` — 7 új teszt (14 összesen, 100% zöld)
- `config.example.yaml` — új paraméterek dokumentálva

---

### Miért: A fő probléma

Ugyanazon a képen kézzel bejelölt személyek melletti arc nem lett felismerve — különösen:
- profilnézet, kisebb arc, részleges takarás
- fix threshold (0.72) kizárta a határeset egyezéseket

---

### Pass 1: Adaptive threshold

`_compute_adaptive_threshold(face)` kiszámítja face-specifikus küszöböt:
- `quality_score` (0–1): rosszabb minőség → alacsonyabb küszöb
- `bbox area / (80×80)`: kisebb arc → alacsonyabb küszöb
- `bbox_w / bbox_h` arányszám: profil arc (keskeny) → alacsonyabb küszöb
- Eredmény: `adaptive_min_threshold + (base - min) * combined`
- Alapértelmezett: base=0.72, min=0.55

**Why:** A kis/profilos arcoknak az embedding modell kevesebb infót lát, de az arc mégis felismerhető kellene legyen. A fix 0.72 ezeket kizárta.

**How to apply:** Ha valaki panaszkodik hogy profilos vagy kis arc nem kerül felismerésre, csökkenteni lehet az `adaptive_min_threshold`-ot.

---

### Pass 2: Same-image context assist

`_run_assisted_pass()` azokat a face-eket próbálja újra, amelyek:
1. Az első körben nem lettek hozzárendelve
2. Ugyanazon a képen van legalább 1 kézzel megerősített személy

Működés:
- `_load_image_confirmed_persons(image_ids)` → {image_id: {person_id, ...}}
- A keresés CSAK a képen szereplő, megerősített személyekre van korlátozva
- Alacsonyabb küszöb (0.62) és margin (0.05)
- Negatív korrekciók itt is érvényesülnek

**Why:** Ha Dávid már be van jelölve egy képen, a mellette lévő arc felismerését segíti ha csak Dávid profiljával hasonlítjuk. Ez csökkenti a hamis pozitívok esélyét is.

**How to apply:** `same_image_assist_enabled: false` kikapcsolja ha túl sok téves egyezés jön.

---

### Részletes logging

`_match_face()` minden reject esetén DEBUG szinten logolja:
- `score`, `threshold`, `margin`, `quality_score`, `bbox` méret
- Ki volt a legjobb és második legjobb kandidát

→ Diagnosztikához kapcsold be DEBUG logot.

---

### Migráció

Nem kell adatbázis migráció. Minden mező már létezik a `Face` táblán (`quality_score`, `bbox_*`, `assignment_source`). Az új config mezőknek alapértelmezett értékeik vannak → backward kompatibilis.
