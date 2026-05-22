"""Minimal translation module — English / Hungarian."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

log = logging.getLogger(__name__)

SUPPORTED: Dict[str, str] = {"en": "English", "hu": "Magyar"}
_PREFS_FILE = Path.home() / ".face_local_prefs.json"
_lang: str = "en"

_STRINGS: Dict[str, Dict[str, str]] = {
    # ── Toolbar ───────────────────────────────────────────────────────────────
    "select_folder": {"en": "Select Folder …", "hu": "Mappa kiválasztása …"},
    "no_folder":     {"en": "(no folder selected)", "hu": "(nincs mappa kiválasztva)"},
    "scan_index":    {"en": "Scan & Index", "hu": "Beolvasás és indexelés"},
    "stop":          {"en": "Stop", "hu": "Leállítás"},
    "export_csv":    {"en": "Export CSV", "hu": "CSV exportálás"},
    "export_images": {"en": "Export Images", "hu": "Képek exportálása"},
    "settings":      {"en": "Settings", "hu": "Beállítások"},
    "tpu_status":    {"en": "TPU Status", "hu": "TPU Állapot"},

    # ── Sidebar / action row ──────────────────────────────────────────────────
    "rename_person":      {"en": "Rename Person", "hu": "Személy átnevezése"},
    "merge_into":         {"en": "Merge Into …", "hu": "Összevonás …"},
    "delete_person":      {"en": "Delete Person", "hu": "Személy törlése"},
    "remove_face":        {"en": "Remove Selected Face", "hu": "Kiválasztott arc eltávolítása"},
    "reassign_face":      {"en": "Reassign Face …", "hu": "Arc áthelyezése …"},
    "recluster_all":      {"en": "Re-cluster All", "hu": "Újra csoportosítás"},
    "force_rescan":       {"en": "Force Full Rescan", "hu": "Teljes újrabeolvasás"},
    "force_rescan_title": {"en": "Force Full Rescan", "hu": "Teljes újrabeolvasás"},
    "force_rescan_msg":   {"en": "Delete all detected faces and re-run detection on all {n} images?\n"
                                 "This will use the current detector (Coral TPU if available).",
                           "hu": "Törli az összes felismert arcot és újra futtatja a detektálást mind a(z) {n} képen?\n"
                                 "A jelenlegi detektor lesz használva (Coral TPU ha elérhető)."},
    "people_label":       {"en": "People", "hu": "Személyek"},
    "search_placeholder": {"en": "Search person name …", "hu": "Személy neve …"},
    "n_persons":          {"en": "{n} person(s)", "hu": "{n} személy"},

    # ── Window titles ─────────────────────────────────────────────────────────
    "window_title":       {"en": "Face-Local — Offline Face Grouping",
                           "hu": "Face-Local — Offline arcfelismerő"},
    "activity_log":       {"en": "Activity Log", "hu": "Tevékenységnapló"},

    # ── Dialogs — general ─────────────────────────────────────────────────────
    "ok":         {"en": "OK", "hu": "OK"},
    "cancel":     {"en": "Cancel", "hu": "Mégse"},
    "yes":        {"en": "Yes", "hu": "Igen"},
    "no":         {"en": "No", "hu": "Nem"},
    "close":      {"en": "Close", "hu": "Bezárás"},
    "ready":      {"en": "Ready", "hu": "Kész"},
    "error":      {"en": "Error", "hu": "Hiba"},
    "warning":    {"en": "Warning", "hu": "Figyelmeztetés"},

    # ── Dialogs — delete person ───────────────────────────────────────────────
    "delete_person_title":   {"en": "Delete Person",
                              "hu": "Személy törlése"},
    "delete_person_confirm": {"en": "Delete '{name}' and unassign all their faces?\n"
                                    "This cannot be undone.",
                              "hu": "Törlöd '{name}' személyt és feloldod az összes arcát?\n"
                                    "Ez a művelet nem vonható vissza."},

    # ── Dialogs — scan ────────────────────────────────────────────────────────
    "no_folder_title":   {"en": "No Folder", "hu": "Nincs mappa"},
    "no_folder_msg":     {"en": "Please select a root folder first.",
                          "hu": "Először válasszon ki egy mappát."},
    "busy_title":        {"en": "Busy", "hu": "Foglalt"},
    "busy_msg":          {"en": "A scan is already running.", "hu": "Már fut egy beolvasás."},
    "recluster_title":   {"en": "Re-cluster", "hu": "Újra csoportosítás"},
    "recluster_msg":     {"en": "Re-run clustering on all embedded faces?\n"
                                "Manually assigned names are preserved.",
                          "hu": "Újra csoportosítja az összes beágyazott arcot?\n"
                                "A manuálisan beállított nevek megmaradnak."},
    "reclustering":      {"en": "Re-clustering …", "hu": "Újra csoportosítás …"},
    "recluster_done":    {"en": "Re-clustering complete: {n} person(s)",
                          "hu": "Újra csoportosítás kész: {n} személy"},

    # ── Dialogs — rename ─────────────────────────────────────────────────────
    "empty_name_title":     {"en": "Empty Name", "hu": "Üres név"},
    "empty_name_msg":       {"en": "Name cannot be empty.", "hu": "A név nem lehet üres."},
    "protected_rename_title": {"en": "Cannot Rename", "hu": "Nem átnevezhető"},
    "protected_rename_msg":   {"en": "'Ismeretlen' is a protected category and cannot be renamed.",
                                "hu": "Az 'Ismeretlen' egy védett kategória, nem nevezhető át."},
    "protected_delete_title": {"en": "Cannot Delete", "hu": "Nem törölhető"},
    "protected_delete_msg":   {"en": "'Ismeretlen' is a protected category and cannot be deleted.",
                                "hu": "Az 'Ismeretlen' egy védett kategória, nem törölhető."},

    # ── Dialogs — remove face ─────────────────────────────────────────────────
    "remove_face_title": {"en": "Remove Face", "hu": "Arc eltávolítása"},
    "remove_face_msg":   {"en": "Remove this face from the cluster and exclude it from clustering?",
                          "hu": "Eltávolítja ezt az arcot a csoportból és kizárja az újra csoportosításból?"},

    # ── Dialogs — merge ───────────────────────────────────────────────────────
    "merge_error_title": {"en": "Merge Error", "hu": "Összevonási hiba"},
    "reassign_title":    {"en": "Reassign Face To …", "hu": "Arc áthelyezése …"},

    # ── Export ────────────────────────────────────────────────────────────────
    "export_done":       {"en": "Export Done", "hu": "Exportálás kész"},
    "export_csv_saved":  {"en": "Saved to:\n{path}", "hu": "Mentve:\n{path}"},
    "no_person_title":   {"en": "No Person Selected", "hu": "Nincs személy kiválasztva"},
    "no_person_msg":     {"en": "Select a person in the sidebar first.",
                          "hu": "Először válasszon személyt az oldalsávban."},
    "exported_n":        {"en": "Exported {n} image(s) to:\n{folder}",
                          "hu": "{n} kép exportálva:\n{folder}"},

    # ── Settings dialog ───────────────────────────────────────────────────────
    "settings_title":    {"en": "Settings", "hu": "Beállítások"},
    "lang_label":        {"en": "Language:", "hu": "Nyelv:"},
    "db_group":          {"en": "Database", "hu": "Adatbázis"},
    "current_db":        {"en": "Current:", "hu": "Jelenlegi:"},
    "new_db":            {"en": "New Database …", "hu": "Új adatbázis …"},
    "open_db":           {"en": "Open Database …", "hu": "Adatbázis megnyitása …"},
    "db_switched":       {"en": "Database switched. Please restart the scan.",
                          "hu": "Az adatbázis megváltozott. Indítsa el újra a beolvasást."},
    "db_new_title":      {"en": "Create New Database", "hu": "Új adatbázis létrehozása"},
    "db_open_title":     {"en": "Open Database", "hu": "Adatbázis megnyitása"},

    # ── TPU status dialog ─────────────────────────────────────────────────────
    "tpu_title":         {"en": "Google Coral TPU Status", "hu": "Google Coral TPU Állapot"},
    "tpu_devices":       {"en": "Connected devices:", "hu": "Csatlakoztatott eszközök:"},
    "tpu_none":          {"en": "No Edge TPU device found.", "hu": "Nem található Edge TPU eszköz."},
    "tpu_pycoral_ok":    {"en": "pycoral: installed ({ver})", "hu": "pycoral: telepítve ({ver})"},
    "tpu_pycoral_miss":  {"en": "pycoral: NOT installed", "hu": "pycoral: NINCS telepítve"},
    "tpu_libedge_ok":    {"en": "libedgetpu: found", "hu": "libedgetpu: megtalálva"},
    "tpu_libedge_miss":  {"en": "libedgetpu: NOT found", "hu": "libedgetpu: NEM található"},
    "tpu_error":         {"en": "Error: {msg}", "hu": "Hiba: {msg}"},
    "tpu_ok_label":      {"en": "TPU ready ✓", "hu": "TPU kész ✓"},
    "tpu_warn_label":    {"en": "TPU not available", "hu": "TPU nem elérhető"},
    "tpu_inference_ok":  {"en": "✓ Test inference succeeded — TPU is actively accelerating detection",
                          "hu": "✓ Teszt következtetés sikeres — a TPU aktívan gyorsítja a detektálást"},
    "tpu_inference_fail": {"en": "✗ Library loads but device is NOT responding. Detection will use CPU.",
                           "hu": "✗ A könyvtár betöltődik, de az eszköz NEM válaszol. A detektálás CPU-n fut."},
    "tpu_phantom_tip":    {"en": "Tip: unplug the Coral USB device, wait 5 seconds, plug it back in, "
                                  "then click 'Re-check' below. On macOS also check:\n"
                                  "  System Settings → Privacy & Security → USB / Accessory Security",
                           "hu": "Tipp: húzd ki a Coral USB eszközt, várj 5 másodpercet, dugd vissza, "
                                  "majd kattints az 'Újraellenőrzés' gombra. macOS-en nézd meg:\n"
                                  "  Rendszerbeállítások → Adatvédelem és biztonság → USB / Tartozék biztonság"},
    "tpu_recheck":        {"en": "🔄 Re-check / Újraellenőrzés",
                           "hu": "🔄 Újraellenőrzés / Re-check"},

    # ── Image viewer / manual face marking ────────────────────────────────
    "view_all_images":   {"en": "All Images", "hu": "Összes kép"},
    "view_no_face":      {"en": "Images Without Faces", "hu": "Arc nélküli képek"},
    "view_by_person":    {"en": "By Person", "hu": "Személy szerint"},
    "n_images_no_face":  {"en": "{n} image(s) with no detected face",
                          "hu": "{n} kép arc nélkül"},
    "mark_face":         {"en": "Mark Face Manually", "hu": "Arc kézi jelölése"},
    "mark_face_hint":    {"en": "Drag on the image to mark a face region.",
                          "hu": "Húzd az egeret a képen az arc jelöléséhez."},
    "mark_face_saved":   {"en": "Face saved. Assign it to a person from the sidebar.",
                          "hu": "Arc mentve. Rendeld személyhez az oldalsávból."},

    # ── Name suggestions ──────────────────────────────────────────────────
    "suggestions_btn":   {"en": "Name Suggestions", "hu": "Névajánlatok"},
    "suggestions_title": {"en": "Name Suggestions — Unknown → Known",
                          "hu": "Névajánlatok — Ismeretlen → Ismert"},
    "suggestions_intro": {"en": "Unknown people whose faces resemble an already-named "
                                "person. Approve to merge, or reject to never suggest "
                                "the pair again. Nothing is merged without your approval.",
                          "hu": "Ismeretlen személyek, akiknek arca egy már elnevezett "
                                "személyre hasonlít. Jóváhagyással összevonod, "
                                "elutasítással többé nem ajánlja a párost. Összevonás "
                                "csak a te jóváhagyásoddal történik."},
    "suggestions_threshold": {"en": "Similarity threshold:", "hu": "Hasonlósági küszöb:"},
    "suggestions_count":     {"en": "{n} suggestion(s)", "hu": "{n} ajánlat"},
    "suggestions_empty":     {"en": "No suggestions above the current threshold.",
                              "hu": "Nincs a jelenlegi küszöb feletti ajánlat."},
    "suggestions_approve":   {"en": "Approve", "hu": "Jóváhagyás"},
    "suggestions_reject":    {"en": "Reject", "hu": "Elutasítás"},
    "suggestions_similarity": {"en": "{pct}% match", "hu": "{pct}% egyezés"},
    "suggestions_faces":     {"en": "{n} face(s)", "hu": "{n} arc"},
    "suggestions_approve_confirm": {
        "en": "Merge '{cand}' into '{target}'?\n"
              "All faces of '{cand}' will be reassigned to '{target}'.",
        "hu": "Összevonod '{cand}' személyt ezzel: '{target}'?\n"
              "'{cand}' összes arca átkerül ide: '{target}'.",
    },
    "suggestions_found_title": {"en": "Name Suggestions", "hu": "Névajánlatok"},
    "suggestions_found_msg":   {"en": "Found {n} possible name match(es) for unknown "
                                      "people.\nReview them now?",
                                "hu": "{n} lehetséges névegyezés található ismeretlen "
                                      "személyekhez.\nÁtnézed őket most?"},
}


def t(key: str, **kwargs: object) -> str:
    """Return the translated string for *key* in the current language."""
    entry = _STRINGS.get(key)
    if entry is None:
        log.warning("i18n: unknown key %r", key)
        return key
    text = entry.get(_lang) or entry.get("en") or key
    return text.format(**kwargs) if kwargs else text


def current_language() -> str:
    return _lang


def set_language(lang: str) -> None:
    global _lang
    if lang not in SUPPORTED:
        log.warning("i18n: unsupported language %r — falling back to 'en'", lang)
        lang = "en"
    _lang = lang
    _save_prefs()


def load_prefs() -> None:
    global _lang
    try:
        if _PREFS_FILE.exists():
            data = json.loads(_PREFS_FILE.read_text(encoding="utf-8"))
            lang = data.get("language", "en")
            if lang in SUPPORTED:
                _lang = lang
    except Exception as exc:  # noqa: BLE001
        log.warning("i18n: could not load prefs: %s", exc)


def _save_prefs() -> None:
    try:
        data: dict = {}
        if _PREFS_FILE.exists():
            data = json.loads(_PREFS_FILE.read_text(encoding="utf-8"))
        data["language"] = _lang
        _PREFS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        log.warning("i18n: could not save prefs: %s", exc)
