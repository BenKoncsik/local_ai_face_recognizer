"""Minimal translation module — English / Hungarian."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

log = logging.getLogger(__name__)

SUPPORTED: Dict[str, str] = {"en": "English", "hu": "Magyar"}
_LEGACY_PREFS_FILE = Path.home() / ".face_local_prefs.json"
_lang: str = "en"


def _prefs_file() -> Path:
    """Return the active language preferences file path, creating its directory."""
    try:
        from app.paths import ensure_settings_dir
        return ensure_settings_dir() / "language_prefs.json"
    except Exception:  # noqa: BLE001
        return _LEGACY_PREFS_FILE

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
    "recluster_all":      {"en": "Recognize Unassigned", "hu": "Névtelen arcok felismerése"},
    "force_rescan":       {"en": "Force Full Rescan", "hu": "Teljes újrabeolvasás"},
    "force_rescan_title": {"en": "Force Full Rescan", "hu": "Teljes újrabeolvasás"},
    "force_rescan_msg":   {"en": "Delete all detected faces and re-run detection on all {n} images?\n"
                                 "This will use the current detector (Coral TPU if available).\n"
                                 "Manually marked faces are preserved.",
                           "hu": "Törli az összes automatikusan felismert arcot és újra futtatja a detektálást mind a(z) {n} képen?\n"
                                 "A jelenlegi detektor lesz használva (Coral TPU ha elérhető).\n"
                                 "A kézzel jelölt arcok megmaradnak."},
    "redetect_faces":          {"en": "Re-detect Faces", "hu": "Arcok újrakeresése"},
    "redetect_faces_accurate": {"en": "Re-detect (Accurate)", "hu": "Újrakeresés (pontos)"},
    "redetect_faces_tip":      {"en": "Re-run face detection on all images, fast mode.\n"
                                      "Manually marked faces are preserved.",
                                "hu": "Arcok újrakeresése az összes képen, gyors módban.\n"
                                      "A kézzel jelölt arcok megmaradnak."},
    "redetect_faces_accurate_tip": {"en": "Re-run face detection in high-accuracy mode:\n"
                                         "multiple preprocessing variants, lower confidence threshold.\n"
                                         "Slower but finds more faces.\n"
                                         "Manually marked faces are preserved.",
                                    "hu": "Arcok újrakeresése pontos módban:\n"
                                         "több képfeldolgozási variáció, alacsonyabb konfidencia küszöb.\n"
                                         "Lassabb, de több arcot talál.\n"
                                         "A kézzel jelölt arcok megmaradnak."},
    "redetect_title":          {"en": "Re-detect Faces", "hu": "Arcok újrakeresése"},
    "redetect_accurate_title": {"en": "Re-detect Faces — Accurate Mode",
                                "hu": "Arcok újrakeresése — Pontos mód"},
    "redetect_msg":            {"en": "Re-run fast face detection on all {n} images?\n"
                                      "Existing auto-detected faces will be replaced.\n"
                                      "Manually marked faces are preserved.",
                                "hu": "Újra futtatja a gyors arcdetektálást mind a(z) {n} képen?\n"
                                      "A meglévő automatikusan felismert arcok lecserélődnek.\n"
                                      "A kézzel jelölt arcok megmaradnak."},
    "redetect_accurate_msg":   {"en": "Re-run face detection in HIGH ACCURACY mode on all {n} images?\n"
                                      "Uses multiple preprocessing variants (contrast, brightness)\n"
                                      "and a lower confidence threshold — slower but finds more faces.\n\n"
                                      "Manually marked faces are preserved.",
                                "hu": "Újra futtatja az arcdetektálást PONTOS módban mind a(z) {n} képen?\n"
                                      "Több képfeldolgozási variációt (kontraszt, fényesség) és\n"
                                      "alacsonyabb konfidencia küszöböt használ — lassabb, de több arcot talál.\n\n"
                                      "A kézzel jelölt arcok megmaradnak."},
    "people_label":       {"en": "People", "hu": "Személyek"},
    "search_placeholder": {"en": "Search person name …", "hu": "Személy neve …"},
    "n_persons":          {"en": "{n} person(s)", "hu": "{n} személy"},

    # ── Window titles ─────────────────────────────────────────────────────────
    "window_title":       {"en": "Face-Local — Offline Face Grouping",
                           "hu": "Face-Local — Offline arcfelismerő"},
    "activity_log":       {"en": "Activity Log", "hu": "Tevékenységnapló"},
    "main_toolbar":       {"en": "Main", "hu": "Fő eszköztár"},

    # ── Screen recording ──────────────────────────────────────────────────────
    "rec_start_tip":      {"en": "Start screen recording",
                           "hu": "Képernyőrögzítés indítása"},
    "rec_pause_tip":      {"en": "Pause recording", "hu": "Rögzítés szüneteltetése"},
    "rec_resume_tip":     {"en": "Resume recording", "hu": "Rögzítés folytatása"},
    "rec_stop_tip":       {"en": "Stop & save recording",
                           "hu": "Rögzítés leállítása és mentése"},
    "rec_state_idle":         {"en": "Not recording", "hu": "Nincs felvétel"},
    "rec_state_recording":    {"en": "Recording ●", "hu": "Rögzítés ●"},
    "rec_state_paused":       {"en": "Paused", "hu": "Szüneteltetve"},
    "rec_state_finalizing":   {"en": "Saving…", "hu": "Mentés…"},
    "rec_state_error":        {"en": "Recording error", "hu": "Rögzítési hiba"},
    "rec_person_prefix":      {"en": "Selected person:", "hu": "Kijelölt személy:"},
    "rec_choose_dir":         {"en": "Choose recording folder",
                               "hu": "Rögzítési mappa kiválasztása"},
    "rec_ffmpeg_missing_title": {"en": "ffmpeg not found",
                                 "hu": "Az ffmpeg nem található"},
    "rec_ffmpeg_missing_body": {
        "en": "Screen recording needs the ffmpeg binary. Install it (e.g. "
              "'brew install ffmpeg' on macOS) or set its path in Settings → "
              "Recording.",
        "hu": "A képernyőrögzítéshez szükséges az ffmpeg program. Telepítsd "
              "(pl. macOS-en 'brew install ffmpeg'), vagy add meg az "
              "elérési útját a Beállítások → Rögzítés alatt.",
    },
    "rec_error_title":        {"en": "Recording error", "hu": "Rögzítési hiba"},
    "rec_saved_title":        {"en": "Recording saved", "hu": "Felvétel elmentve"},
    "rec_saved_body":         {"en": "Saved to:\n{path}",
                               "hu": "Elmentve ide:\n{path}"},
    "rec_quality_low":        {"en": "Low", "hu": "Alacsony"},
    "rec_quality_normal":     {"en": "Normal", "hu": "Normál"},
    "rec_quality_better":     {"en": "Better", "hu": "Jobb"},
    "rec_privacy_title":      {"en": "Recording is about to start",
                               "hu": "A rögzítés mindjárt elindul"},
    "rec_privacy_body": {
        "en": "Screen and audio recording will begin. In an online meeting, "
              "make sure the participants are aware that the session is being "
              "recorded.",
        "hu": "A képernyő és a hang rögzítése elindul. Online megbeszélés "
              "esetén győződj meg róla, hogy a résztvevők tudnak a "
              "rögzítésről.",
    },
    "rec_privacy_dont_ask":   {"en": "Don't show this again",
                               "hu": "Ne jelenjen meg többször"},
    "rec_audio_meter_label":  {"en": "Audio", "hu": "Hang"},
    "rec_no_audio_warning": {
        "en": "The recording contains NO audio track. Check the microphone "
              "permission (System Settings → Privacy → Microphone) and, for "
              "system sound, that a loopback device (e.g. BlackHole) is "
              "installed and selected in Settings → Recording.",
        "hu": "A felvétel NEM tartalmaz hangsávot. Ellenőrizd a mikrofon "
              "jogosultságot (Rendszerbeállítások → Adatvédelem → Mikrofon), "
              "rendszerhanghoz pedig azt, hogy van-e telepített loopback eszköz "
              "(pl. BlackHole), és ki van-e választva a Beállítások → Rögzítés "
              "alatt.",
    },
    "rec_no_audio_warning_windows": {
        "en": "The recording contains NO audio track. Check that desktop apps "
              "are allowed to use the microphone (Settings → Privacy & security "
              "→ Microphone → \"Let desktop apps access your microphone\"), and "
              "that a microphone is selected and not muted in Settings → "
              "Recording. For system sound, enable \"Stereo Mix\" in the Sound "
              "control panel (Recording tab → right-click → Show Disabled "
              "Devices) or install a virtual loopback (e.g. VB-Audio Virtual "
              "Cable), then select it in Settings → Recording.",
        "hu": "A felvétel NEM tartalmaz hangsávot. Ellenőrizd, hogy az asztali "
              "alkalmazások használhatják-e a mikrofont (Beállítások → "
              "Adatvédelem és biztonság → Mikrofon → „Asztali alkalmazások "
              "hozzáférhetnek a mikrofonhoz”), és hogy a Beállítások → Rögzítés "
              "alatt ki van-e választva egy mikrofon, és nincs-e némítva. "
              "Rendszerhanghoz engedélyezd a „Stereo Mix” eszközt a Hang "
              "vezérlőpulton (Felvétel fül → jobb klikk → Letiltott eszközök "
              "megjelenítése), vagy telepíts egy virtuális loopback eszközt "
              "(pl. VB-Audio Virtual Cable), majd válaszd ki a Beállítások → "
              "Rögzítés alatt.",
    },
    "rec_audio_input_device": {"en": "Audio input device (microphone)",
                               "hu": "Hangbemeneti eszköz (mikrofon)"},
    "rec_system_audio_device": {"en": "System audio device (loopback)",
                                "hu": "Rendszerhang eszköz (loopback)"},
    "rec_audio_auto":         {"en": "Automatic", "hu": "Automatikus"},
    "rec_mic_volume":         {"en": "Microphone volume", "hu": "Mikrofon hangereje"},
    "rec_system_volume":      {"en": "System audio volume",
                               "hu": "Rendszerhang hangereje"},
    "rec_mute_microphone":    {"en": "Mute microphone", "hu": "Mikrofon némítása"},
    "rec_mute_system_audio":  {"en": "Mute system audio",
                               "hu": "Rendszerhang némítása"},
    "rec_audio_group":        {"en": "Audio", "hu": "Hang"},
    "tab_face_recognition": {"en": "Face Recognition", "hu": "Arcfelismerés"},
    "tab_image_browser":  {"en": "Image Browser", "hu": "Képböngésző"},
    "tab_family_search":  {"en": "Family Search", "hu": "Családi kereső"},
    "tab_locations":      {"en": "Locations", "hu": "Helyek"},
    "tab_collage":        {"en": "Collage", "hu": "Kollázs"},
    "export":             {"en": "Export", "hu": "Export"},
    "collage_import":     {"en": "Collage Import …", "hu": "Kollázs import…"},
    "collage_import_tip": {"en": "Import Picasa collage file (.cxf/.cfx)",
                           "hu": "Picasa kollázs (.cxf/.cfx) beolvasása"},
    "open_collage_file": {"en": "Open Collage File", "hu": "Kollázs fájl megnyitása"},
    "picasa_collage_filter": {"en": "Picasa collage (*.cxf *.cfx);;All files (*)",
                              "hu": "Picasa kollázs (*.cxf *.cfx);;Minden fájl (*)"},
    "extra_search_root": {"en": "Extra search root for resolving Windows paths (optional)",
                          "hu": "Keresési gyökérmappa (opcionális)"},
    "collages_imported": {"en": "{n} collage(s) imported.", "hu": "{n} kollázs importálva."},
    "import_errors":     {"en": "Errors:", "hu": "Hibák:"},
    "collage_import_title": {"en": "Collage Import", "hu": "Kollázs import"},
    "collage_html_export": {"en": "Collage HTML Export", "hu": "Kollázs HTML export"},
    "html_export_folder": {"en": "HTML Export Folder", "hu": "HTML export mappa"},
    "static_site_ready": {"en": "Static website is ready:\n{path}",
                          "hu": "Statikus weboldal elkészült:\n{path}"},
    "check_updates":      {"en": "Check for Updates …", "hu": "Frissítés keresése…"},
    "checking":           {"en": "Checking…", "hu": "Ellenőrzés…"},
    "up_to_date":         {"en": "Up to date", "hu": "Naprakész"},
    "update_available_short": {"en": "Update: v{version}", "hu": "Frissítés: v{version}"},
    "update_status_found": {"en": "New version available: v{version} — click update",
                            "hu": "Új verzió elérhető: v{version} — kattints a frissítésre"},
    "update_notification_title": {"en": "Face-Local update", "hu": "Face-Local frissítés"},
    "update_notification_msg": {"en": "New version available: v{version}. Click to update.",
                                "hu": "Új verzió elérhető: v{version}. Kattints a frissítéshez."},
    "update_connection_failed": {"en": "Could not reach GitHub.\nCheck your internet connection.",
                                 "hu": "Nem sikerült kapcsolódni a GitHub-hoz.\nEllenőrizd az internet kapcsolatot."},
    "app_is_current":     {"en": "The application is up to date.\nCurrent version: v{version}",
                           "hu": "Az alkalmazás naprakész.\nJelenlegi verzió: v{version}"},
    "update_available":   {"en": "New version available", "hu": "Új verzió elérhető"},
    "current":            {"en": "Current", "hu": "Jelenlegi"},
    "latest":             {"en": "Latest", "hu": "Legújabb"},
    "package":            {"en": "Package", "hu": "Csomag"},
    "download_install":   {"en": "Download & Install", "hu": "Letöltés és telepítés"},
    "update_restart":     {"en": "Update & Restart", "hu": "Frissítés és újraindítás"},
    "open_installer":     {"en": "Open installer", "hu": "Telepítő megnyitása"},
    "skip":               {"en": "Skip", "hu": "Kihagyás"},
    "downloading":        {"en": "Downloading…", "hu": "Letöltés…"},
    "downloaded_updating": {"en": "Downloaded — updating & restarting…",
                            "hu": "Letöltve — frissítés és újraindítás…"},
    "downloaded_click_install": {"en": "Downloaded — click to install",
                                 "hu": "Letöltve — kattints a telepítéshez"},
    "force_rescan_tip":   {"en": "Deletes all faces and re-runs detection",
                           "hu": "Törli az összes arcot és újra futtatja a detektálást"},
    "suggestions_tip":    {"en": "Match unknown people against named people",
                           "hu": "Ismeretlen személyek összevetése az elnevezett személyekkel"},
    "person_info":        {"en": "Person Info", "hu": "Személyadatok"},
    "person_info_tip":    {"en": "Edit last name, first name, birth data and notes",
                           "hu": "Vezetéknév, keresztnév, születési adatok és megjegyzés szerkesztése"},
    "gender":             {"en": "Gender", "hu": "Nem"},
    "gender_unknown":     {"en": "Not set", "hu": "Nincs megadva"},
    "gender_male":        {"en": "Male", "hu": "Férfi"},
    "gender_female":      {"en": "Female", "hu": "Nő"},
    "family_code":        {"en": "Family code:", "hu": "Családi azonosító:"},
    "example_family_code": {"en": "e.g. C85", "hu": "pl. C85"},
    "family_code_help": {
        "en": "Examples: C85 = Cikky's 8th child's 5th child, "
              "C80 = spouse of Cikky's 8th child, C0F1 = Cikky's father, "
              "C00T1 = spouse's 1st sibling, C81B = friend of C81.",
        "hu": "Példák: C85 = Cikky 8. gyerekének 5. gyereke, "
              "C80 = Cikky 8. gyerekének házastársa, C0F1 = Cikky apja, "
              "C00T1 = házastárs 1. testvére, C81B = C81 barátja/ismerőse.",
    },
    "family_code_invalid_title": {
        "en": "Invalid family code",
        "hu": "Hibás családi azonosító",
    },
    "external_family_code": {
        "en": "External family code:",
        "hu": "Külső családi azonosító:",
    },
    "example_external_family_code": {"en": "e.g. #Zoli#", "hu": "pl. #Zoli#"},
    "external_family_code_help": {
        "en": "The external person's own family root, e.g. #Zoli# = Zoli, "
              "#Zoli#0 = Zoli's spouse, #Zoli#31 = Zoli's 3rd child's 1st child. "
              "Kept separate from the main family code.",
        "hu": "A külső személy saját családi gyökere, pl. #Zoli# = Zoli, "
              "#Zoli#0 = Zoli házastársa, #Zoli#31 = Zoli 3. gyerekének 1. gyereke. "
              "A fő családi azonosítótól elkülönítve.",
    },
    "external_family_code_invalid_title": {
        "en": "Invalid external family code",
        "hu": "Hibás külső családi azonosító",
    },

    # ── Family search ───────────────────────────────────────────────────────
    "family_person_a":    {"en": "First person", "hu": "Első személy"},
    "family_person_b":    {"en": "Second person", "hu": "Második személy"},
    "family_relationship_filter": {"en": "Relationship", "hu": "Kapcsolat"},
    "family_search_btn":  {"en": "Search", "hu": "Keresés"},
    "family_names_placeholder": {
        "en": "Names separated by commas, e.g. Benedek, Matyi, Domi",
        "hu": "Nevek vesszővel elválasztva, pl. Benedek, Matyi, Domi",
    },
    "family_allow_others_yes": {"en": "Others allowed", "hu": "Lehetnek mások is"},
    "family_allow_others_no": {"en": "Only these people", "hu": "Csak ezek a személyek"},
    "family_details_closed": {"en": "Detailed search", "hu": "Részletes kereső"},
    "family_details_open": {"en": "Detailed search", "hu": "Részletes kereső"},
    "family_detail_person_text": {"en": "Person data", "hu": "Személyadat"},
    "family_detail_person_placeholder": {
        "en": "Name, nickname, birth/death data, notes",
        "hu": "Név, becenév, születési/halálozási adat, megjegyzés",
    },
    "family_detail_image_text": {"en": "Image data", "hu": "Képadat"},
    "family_detail_image_placeholder": {
        "en": "Filename, path, date or place",
        "hu": "Fájlnév, elérési út, dátum vagy hely",
    },
    "family_detail_photo_date": {"en": "Photo date", "hu": "Kép dátuma"},
    "family_detail_place_text": {"en": "Place", "hu": "Hely"},
    "family_detail_place_placeholder": {"en": "Place name", "hu": "Hely neve"},
    "family_gender_any": {"en": "Any gender", "hu": "Bármelyik nem"},
    "family_mode_both":   {"en": "May include others", "hu": "Lehetnek mások is"},
    "family_mode_exact":  {"en": "Only these two", "hu": "Csak ez a két személy"},
    "family_filter_any":  {"en": "Any relationship", "hu": "Bármilyen kapcsolat"},
    "family_filter_spouse": {"en": "Spouse", "hu": "Házastárs"},
    "family_filter_parent": {"en": "First is parent", "hu": "Első a szülő"},
    "family_filter_child": {"en": "First is child", "hu": "Első a gyerek"},
    "family_filter_sibling": {"en": "Sibling", "hu": "Testvér"},
    "family_add_relationship": {"en": "Save relationship:", "hu": "Kapcsolat mentése:"},
    "family_save_relationship": {"en": "Save", "hu": "Mentés"},
    "family_add_spouse": {"en": "Spouses", "hu": "Házastársak"},
    "family_add_a_parent_b": {"en": "First is parent of second", "hu": "Első szülője a másodiknak"},
    "family_add_b_parent_a": {"en": "Second is parent of first", "hu": "Második szülője az elsőnek"},
    "family_select_two": {"en": "Select two different people.", "hu": "Válassz ki két különböző személyt."},
    "family_no_relationship": {"en": "No stored or derived relationship between them.",
                               "hu": "Nincs tárolt vagy számított kapcsolat közöttük."},
    "family_validation_title": {"en": "Invalid relationship", "hu": "Hibás kapcsolat"},
    "family_prev":       {"en": "Previous", "hu": "Előző"},
    "family_next":       {"en": "Next", "hu": "Következő"},
    "family_results_empty": {"en": "No results", "hu": "Nincs találat"},
    "family_results_range": {"en": "{start}-{end} of {total} images",
                             "hu": "{start}-{end} / {total} kép"},
    "family_no_results": {"en": "No matching images.", "hu": "Nincs megfelelő kép."},
    "family_result_item": {"en": "{name}  ({persons} people, {faces} faces)",
                           "hu": "{name}  ({persons} személy, {faces} arc)"},
    "ibp_person_filter_hdr": {"en": "Person filter", "hu": "Személy szűrő"},
    "ibp_person_filter_apply": {"en": "Apply", "hu": "Alkalmaz"},
    "ibp_person_filter_clear": {"en": "Clear", "hu": "Törlés"},

    # ── Dialogs — general ─────────────────────────────────────────────────────
    "ok":         {"en": "OK", "hu": "OK"},
    "cancel":     {"en": "Cancel", "hu": "Mégse"},
    "yes":        {"en": "Yes", "hu": "Igen"},
    "no":         {"en": "No", "hu": "Nem"},
    "close":      {"en": "Close", "hu": "Bezárás"},
    "clear":      {"en": "Clear", "hu": "Törlés"},
    "ready":      {"en": "Ready", "hu": "Kész"},
    "error":      {"en": "Error", "hu": "Hiba"},
    "warning":    {"en": "Warning", "hu": "Figyelmeztetés"},
    "save_error": {"en": "Save Error", "hu": "Mentési hiba"},
    "unknown":    {"en": "Unknown", "hu": "Ismeretlen"},
    "filename":   {"en": "Filename", "hu": "Fájlnév"},
    "path":       {"en": "Path", "hu": "Elérési út"},
    "raw_path":   {"en": "Path (raw)", "hu": "Elérési út (raw)"},
    "persons":    {"en": "Persons", "hu": "Személyek"},
    "no_recognized_face": {"en": "(no recognized face)", "hu": "(nincs felismert arc)"},
    "source_file_missing": {"en": "⚠ Source file not found!", "hu": "⚠ Forrásfájl nem található!"},

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
    "recluster_title":   {"en": "Recognize Unassigned Faces",
                          "hu": "Névtelen arcok felismerése"},
    "recluster_msg":     {"en": "Recognize unassigned and auto-named faces from the people you have already labeled?\n"
                                "Existing manually categorized people are preserved.",
                          "hu": "Felismerje a még névtelen vagy automatikus arcokat a már megjelölt személyek alapján?\n"
                                "A kézzel kategorizált személyek megmaradnak."},
    "reclustering":      {"en": "Recognizing faces …", "hu": "Arcok felismerése …"},
    "recluster_done":    {"en": "Recognition complete: {n} face(s) assigned",
                          "hu": "Felismerés kész: {n} arc hozzárendelve"},

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
    "merge_source_into": {"en": "Merge <b>{name}</b> into:", "hu": "<b>{name}</b> összevonása ide:"},
    "merge_faces_count": {"en": "{name} ({n} faces)", "hu": "{name} ({n} arc)"},
    "merge_source_deleted": {"en": "The source person will be deleted after merging.",
                             "hu": "A forrás személy az összevonás után törlődik."},
    "rename_person_to":  {"en": "Rename <b>{name}</b> to:", "hu": "<b>{name}</b> új neve:"},

    # ── Export ────────────────────────────────────────────────────────────────
    "export_done":       {"en": "Export Done", "hu": "Exportálás kész"},
    "export_csv_saved":  {"en": "Saved to:\n{path}", "hu": "Mentve:\n{path}"},
    "no_person_title":   {"en": "No Person Selected", "hu": "Nincs személy kiválasztva"},
    "no_person_msg":     {"en": "Select a person in the sidebar first.",
                          "hu": "Először válasszon személyt az oldalsávban."},
    "exported_n":        {"en": "Exported {n} image(s) to:\n{folder}",
                          "hu": "{n} kép exportálva:\n{folder}"},
    "export_title":      {"en": "Export", "hu": "Exportálás"},
    # ── Full project package (.facepack) ──────────────────────────────────────
    "pkg_group":         {"en": "Full project package (.facepack)",
                          "hu": "Teljes projektcsomag (.facepack)"},
    "pkg_export_desc":   {"en": "Bundle the database, face crops, local images and "
                                "config into a single portable .facepack file.",
                          "hu": "Az adatbázis, az arckivágások, a lokális képek és a "
                                "konfiguráció egyetlen hordozható .facepack fájlba."},
    "pkg_export_btn":    {"en": "Export full project", "hu": "Teljes projekt exportálása"},
    "pkg_import_desc":   {"en": "Restore a project from a .facepack file into a new "
                                "project folder.",
                          "hu": "Projekt visszaállítása egy .facepack fájlból új "
                                "projektmappába."},
    "pkg_import_btn":    {"en": "Import project", "hu": "Projekt importálása"},
    "pkg_export_dialog": {"en": "Export project package", "hu": "Projektcsomag exportálása"},
    "pkg_import_dialog": {"en": "Select project package", "hu": "Projektcsomag kiválasztása"},
    "pkg_import_dest":   {"en": "Choose destination project folder",
                          "hu": "Válassz cél projektmappát"},
    "pkg_filter":        {"en": "Face-Local package (*.facepack)",
                          "hu": "Face-Local csomag (*.facepack)"},
    "pkg_progress_export": {"en": "Exporting project package…",
                            "hu": "Projektcsomag exportálása…"},
    "pkg_progress_import": {"en": "Importing project package…",
                            "hu": "Projektcsomag importálása…"},
    "pkg_export_done":   {"en": "Project package", "hu": "Projektcsomag"},
    "pkg_export_ok":     {"en": "Saved {images} image(s) and {crops} crop(s) to:\n{path}",
                          "hu": "{images} kép és {crops} kivágás mentve ide:\n{path}"},
    "pkg_export_warn":   {"en": "\n\nThe export finished, but {missing} image(s) "
                                "were not available and could not be bundled "
                                "(see log).",
                          "hu": "\n\nAz export elkészült, de {missing} kép nem volt "
                                "elérhető, és nem került a csomagba (részletek a logban)."},
    "pkg_import_title":  {"en": "Import project", "hu": "Projekt importálása"},
    "pkg_import_confirm": {"en": "This will extract the project into:\n{dest}\n\n"
                                 "Your current project will not be overwritten. "
                                 "Continue?",
                           "hu": "A projekt ide lesz kibontva:\n{dest}\n\n"
                                 "A jelenlegi projekt nem íródik felül. Folytatja?"},
    "pkg_import_ok":     {"en": "Project imported to:\n{dest}\n\nOpen it now?",
                          "hu": "Projekt importálva ide:\n{dest}\n\nMegnyitja most?"},
    "pkg_import_opened": {"en": "Imported project is now active.",
                          "hu": "Az importált projekt mostantól aktív."},
    "pkg_import_later":  {"en": "Imported project is ready at:\n{dest}",
                          "hu": "Az importált projekt készen áll itt:\n{dest}"},
    "pkg_error_title":   {"en": "Project package error", "hu": "Projektcsomag hiba"},
    "export_scope":      {"en": "Scope", "hu": "Hatókör"},
    "export_all_persons": {"en": "All persons", "hu": "Összes személy"},
    "export_selected_person": {"en": "Selected person only: {name}",
                               "hu": "Csak a kiválasztott: {name}"},
    "csv_export":        {"en": "CSV export", "hu": "CSV export"},
    "export_csv_desc":   {"en": "Metadata table: person, face, bounding box, confidence.",
                          "hu": "Metaadat táblázat: személy, arc, bounding box, konfidencia."},
    "export_save_csv":   {"en": "Save CSV …", "hu": "CSV mentése…"},
    "json_export":       {"en": "JSON export", "hu": "JSON export"},
    "export_json_desc":  {"en": "Structured JSON: faces grouped per person.",
                          "hu": "Strukturált JSON: személyenként csoportosított arcok."},
    "export_save_json":  {"en": "Save JSON …", "hu": "JSON mentése…"},
    "export_images_desc": {"en": "Copy face crop thumbnails to a folder.",
                           "hu": "Arc kivágások másolása egy mappába."},
    "export_choose_folder": {"en": "Choose Folder …", "hu": "Mappa kiválasztása…"},
    "export_need_person_tip": {"en": "Select a person in the main window",
                               "hu": "Válassz ki egy személyt a főablakban"},
    "export_astro_group": {"en": "Fast Static Site (Astro)", "hu": "Gyors statikus weboldal (Astro)"},
    "export_astro_desc":  {"en": "Paginated, lazy-loading gallery built with Astro — stays fast with thousands of persons/photos. Generates thumbnails and per-person/per-photo pages. Requires Node.js to build.",
                          "hu": "Lapozott, lazy-loadolt galéria Astro-val — több ezer személy/kép mellett is gyors. Thumbnailöket és személyenkénti/képenkénti oldalakat generál. A buildhez Node.js szükséges."},
    "export_generate_astro": {"en": "Generate Astro Site …", "hu": "Astro weboldal generálása…"},
    "astro_building": {"en": "Building the Astro site (this may take a minute) …",
                       "hu": "Astro weboldal építése (ez eltarthat egy percig)…"},
    "astro_done": {"en": "Astro Site Ready", "hu": "Astro weboldal kész"},
    "astro_open": {"en": "Generated:\n{path}\n\nOpen it in the browser?",
                   "hu": "Generálva:\n{path}\n\nMegnyitod a böngészőben?"},
    "astro_folder": {"en": "Astro Site Folder", "hu": "Astro weboldal mappája"},
    "astro_failed": {"en": "Astro Export Failed", "hu": "Astro export sikertelen"},
    "astro_need_node": {"en": "Node.js/npm was not found. Install Node.js (https://nodejs.org) to build the Astro site.\n\nDetails: {err}",
                        "hu": "A Node.js/npm nem található. Telepítsd a Node.js-t (https://nodejs.org) az Astro weboldal építéséhez.\n\nRészletek: {err}"},
    "csv_exported":      {"en": "CSV Exported", "hu": "CSV exportálva"},
    "json_exported":     {"en": "JSON Exported", "hu": "JSON exportálva"},
    "json_saved":        {"en": "Saved to:\n{path}", "hu": "Mentve:\n{path}"},
    "images_exported":   {"en": "Images Exported", "hu": "Képek exportálva"},
    "files_copied":      {"en": "{n} file(s) copied:\n{folder}", "hu": "{n} fájl másolva:\n{folder}"},

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
    "updates_group":     {"en": "Updates", "hu": "Frissítések"},
    "current_version":   {"en": "Current version: <b>v{version}</b>",
                          "hu": "Jelenlegi verzió: <b>v{version}</b>"},
    "notify_updates":    {"en": "Notify when a new version is available",
                          "hu": "Értesítés ha új verzió érhető el"},
    "check_updates_btn": {"en": "Check for updates", "hu": "Frissítés keresése"},
    "could_not_reach_github": {"en": "Could not reach GitHub", "hu": "Nem sikerült a kapcsolódás"},
    "up_to_date_version": {"en": "Up to date — v{version}", "hu": "Naprakész — v{version}"},
    "new_version_status": {"en": "New version: <b>v{new}</b> (current: v{current})",
                           "hu": "Új verzió: <b>v{new}</b> (jelenlegi: v{current})"},
    "fix":               {"en": "Fix", "hu": "Javítás"},
    "ai_edge_missing":   {"en": "ai-edge-litert missing", "hu": "ai-edge-litert hiányzik"},
    "libedgetpu_missing_short": {"en": "libedgetpu missing", "hu": "libedgetpu hiányzik"},

    # ── TPU status dialog ─────────────────────────────────────────────────────
    "tpu_title":         {"en": "Google Coral TPU Status", "hu": "Google Coral TPU Állapot"},
    "tpu_devices":       {"en": "Connected devices:", "hu": "Csatlakoztatott eszközök:"},
    "tpu_none":          {"en": "No Edge TPU device found.", "hu": "Nem található Edge TPU eszköz."},
    "tpu_pycoral_ok":    {"en": "pycoral: installed ({ver})", "hu": "pycoral: telepítve ({ver})"},
    "tpu_pycoral_miss":  {"en": "pycoral: NOT installed", "hu": "pycoral: NINCS telepítve"},
    "tpu_ai_edge_missing": {"en": "ai-edge-litert: not installed",
                            "hu": "ai-edge-litert: nincs telepítve"},
    "tpu_delegate_loaded": {"en": "EdgeTPU delegate loaded successfully",
                            "hu": "EdgeTPU delegate sikeresen betöltve"},
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
    "tpu_checking":       {"en": "Checking...", "hu": "Ellenőrzés..."},
    "tpu_manual_commands": {"en": "Manual install commands (run in terminal)",
                            "hu": "Kézi telepítési parancsok (futtasd terminálban)"},
    "tpu_copy_commands":  {"en": "Copy commands", "hu": "Parancsok másolása"},
    "tpu_auto_fix":       {"en": "Auto-fix", "hu": "Automatikus javítás"},
    "tpu_run_auto_fix":   {"en": "Run auto-fix", "hu": "Automatikus javítás indítása"},
    "tpu_fix_done":       {"en": "\n✓ Done! Restart the app.",
                           "hu": "\n✓ Kész! Indítsa újra az alkalmazást."},
    "tpu_fix_failed":     {"en": "\n✗ Some commands failed — copy the commands above and run in terminal as admin.",
                           "hu": "\n✗ Néhány parancs meghiúsult — másolja ki a parancsokat és futtassa terminálban rendszergazdaként."},

    # ── Image viewer / manual face marking ────────────────────────────────
    "view_all_images":   {"en": "All Images", "hu": "Összes kép"},
    "view_no_face":      {"en": "Images Without Faces", "hu": "Arc nélküli képek"},
    "view_by_person":    {"en": "By Person", "hu": "Személy szerint"},
    "n_images_no_face":  {"en": "{n} image(s) with no detected face",
                          "hu": "{n} kép arc nélkül"},
    "mark_face":         {"en": "Mark Faces Manually", "hu": "Arcok kézi jelölése"},
    "mark_face_hint":    {"en": "Drag on the image to mark a face region. You can add multiple faces.",
                          "hu": "Húzd az egeret a képen az arc jelöléséhez. Több arcot is hozzáadhatsz."},
    "mark_face_saved":   {"en": "Face saved. Assign it to a person from the sidebar.",
                          "hu": "Arc mentve. Rendeld személyhez az oldalsávból."},
    "mark_face_add_btn": {"en": "Add Face", "hu": "Arc hozzáadása"},
    "mark_face_done_btn": {"en": "Done", "hu": "Kész"},
    "mark_face_count":   {"en": "{n} face(s) marked", "hu": "{n} arc jelölve"},
    "mark_face_none_yet": {"en": "Draw a rectangle to mark a face",
                           "hu": "Rajzolj téglalapot egy arc jelöléséhez"},
    "previous":          {"en": "Previous", "hu": "Előző"},
    "next":              {"en": "Next", "hu": "Következő"},
    "manual_marking":    {"en": "Manual Marking", "hu": "Kézi jelölés"},
    "manual_marking_tip": {"en": "Click, then drag on the image to manually mark a face",
                           "hu": "Kattints, majd húzd az egeret a képen egy arc kézi megjelöléséhez"},
    "fullscreen":        {"en": "Full Screen", "hu": "Teljes képernyő"},
    "fullscreen_tip":    {"en": "Full-screen view", "hu": "Teljes képernyős nézet"},
    "exit_fullscreen":   {"en": "Exit Full Screen", "hu": "Kilépés a teljes képernyőből"},
    "exit_fullscreen_tip": {"en": "Exit full-screen view", "hu": "Kilépés a teljes képernyős nézetből"},
    "draw_face_hint_auto": {"en": "Drag around the face to mark it — saved automatically",
                            "hu": "Húzd az egeret az arc körül a jelöléshez — mentés automatikus"},
    "draw_face_hint":    {"en": "Drag around the face to select it",
                          "hu": "Húzd az egeret az arc körül a kijelöléshez"},
    "redraw_face_hint":  {"en": "Redraw the selection around the face",
                          "hu": "Rajzold újra az arc körüli kijelölést"},
    "redraw_face_rect_hint": {"en": "Redraw the rectangle around the face — it replaces the old one",
                              "hu": "Rajzold újra az arc körüli téglalapot — a régi helyére kerül"},
    "select_folder_scan": {"en": "Select a folder and run a scan",
                           "hu": "Válassz mappát és futtass beolvasást"},
    "folder":            {"en": "Folder:", "hu": "Mappa:"},
    "image_date_period": {"en": "Image date / period:", "hu": "Kép dátuma / időszaka:"},
    "date_period_placeholder": {"en": "e.g. 1954 or 1954.03.12 or the 1930s",
                                "hu": "pl. 1954 vagy 1954.03.12 vagy 1930-as évek"},
    "date_period_tip":   {"en": "The date or period when the image was taken (free text)",
                          "hu": "A kép készítésének dátuma vagy időszaka (szabad szöveg)"},
    "selected_face":     {"en": "Selected face:", "hu": "Kiválasztott arc:"},
    "click_face_on_image": {"en": "Click a face on the image", "hu": "Kattints egy arcra a képen"},
    "identified_person": {"en": "Identified person:", "hu": "Beazonosított személy:"},
    "face_uncategorized": {"en": "This face is uncategorized", "hu": "Ez az arc nincs kategorizálva"},
    "no_face_on_image":  {"en": "No recognized face on this image", "hu": "Nincs felismert arc ezen a képen"},
    "new_name":          {"en": "New name …", "hu": "Új név…"},
    "assign_existing":   {"en": "Assign to existing person:", "hu": "Hozzárendelés meglévő személyhez:"},
    "assign":            {"en": "Assign", "hu": "Hozzárendelés"},
    "create_person":     {"en": "Create new person:", "hu": "Új személy létrehozása:"},
    "person_name":       {"en": "Person name …", "hu": "Személy neve…"},
    "create_and_assign": {"en": "Create and Assign", "hu": "Létrehozás és hozzárendelés"},
    "no_images_in_db":   {"en": "No images in the database", "hu": "Nincs kép az adatbázisban"},
    "cannot_load":       {"en": "Cannot load:\n{path}", "hu": "Nem tölthető be:\n{path}"},
    "rename_person_tip": {"en": "Rename person", "hu": "Személy átnevezése"},
    "edit_person_info_tip": {"en": "Edit person info", "hu": "Személyadatok szerkesztése"},

    # ── Person details ───────────────────────────────────────────────────
    "person_info_title": {"en": "Person Info — {name}", "hu": "Személyadatok — {name}"},
    "last_name":         {"en": "Last name:", "hu": "Vezetéknév:"},
    "first_name":        {"en": "First name:", "hu": "Keresztnév:"},
    "second_name":       {"en": "Second name:", "hu": "Keresztnév 2:"},
    "nickname":          {"en": "Nickname:", "hu": "Becenév:"},
    "married_name":      {"en": "Married name:", "hu": "Férjezett név:"},
    "birth_place":       {"en": "Birth place:", "hu": "Születési hely:"},
    "birth_date":        {"en": "Birth date:", "hu": "Születési idő:"},
    "death_date":        {"en": "Death date:", "hu": "Halálozás ideje:"},
    "death_place":       {"en": "Death place:", "hu": "Halálozás helye:"},
    "notes":             {"en": "Notes:", "hu": "Egyéb megjegyzés:"},
    "example_last_name": {"en": "e.g. Smith", "hu": "pl. Kovács"},
    "example_first_name": {"en": "e.g. John", "hu": "pl. János"},
    "example_second_name": {"en": "e.g. William (optional)", "hu": "pl. István (opcionális)"},
    "example_nickname":  {"en": "e.g. Johnny (optional)", "hu": "pl. Jani (opcionális)"},
    "example_married_name": {"en": "e.g. Jane Smith (optional)",
                             "hu": "pl. Kovács Jánosné (opcionális)"},
    "example_birth_place": {"en": "e.g. Budapest", "hu": "pl. Budapest"},
    "example_birth_date": {"en": "e.g. 1954 or 1954.03.12 or the 1930s",
                           "hu": "pl. 1954 vagy 1954.03.12 vagy 1930-as évek"},
    "example_death_date": {"en": "e.g. 2001 or 2001.11.23",
                           "hu": "pl. 2001 vagy 2001.11.23"},
    "example_death_place": {"en": "e.g. Debrecen", "hu": "pl. Debrecen"},
    "free_notes":        {"en": "Free-text notes …", "hu": "Szabad szöveges megjegyzések…"},
    "person_groups":     {"en": "Groups / communities:", "hu": "Társaságok / közösségek:"},
    "person_groups_protected_tip": {
        "en": "Groups cannot be assigned to the protected 'Unknown' person.",
        "hu": "A védett 'Ismeretlen' személyhez nem rendelhető kategória.",
    },

    # ── Collage ──────────────────────────────────────────────────────────
    "collage_label":     {"en": "Collage:", "hu": "Kollázs:"},
    "fit":               {"en": "Fit", "hu": "Illeszkedés"},
    "faces":             {"en": "Faces", "hu": "Arcok"},
    "select_collage":    {"en": "Select a collage", "hu": "Válassz kollázst"},
    "collage_not_found": {"en": "Collage not found.", "hu": "Kollázs nem található."},
    "untitled":          {"en": "(untitled)", "hu": "(cím nélkül)"},
    "items_count":       {"en": "{n} item(s)", "hu": "{n} elem"},
    "no_collage_selected": {"en": "No collage selected.", "hu": "Nincs kiválasztott kollázs."},
    "select_export_folder": {"en": "Select Export Folder", "hu": "Exportálási mappa kiválasztása"},
    "collage_exported":  {"en": "Annotated collage exported:\n{path}",
                          "hu": "Annotált kollázs exportálva:\n{path}"},
    "export_error":      {"en": "Export Error", "hu": "Export hiba"},
    "collage_item_details": {"en": "Collage Item Details", "hu": "Kollázs elem részletei"},
    "year":              {"en": "Year:", "hu": "Év:"},
    "location":          {"en": "Location:", "hu": "Helyszín:"},
    "event":             {"en": "Event:", "hu": "Esemény:"},
    "example_year":      {"en": "e.g. 1969", "hu": "pl. 1969"},
    "example_location":  {"en": "e.g. Budapest, Balaton", "hu": "pl. Budapest, Balaton"},
    "example_event":     {"en": "e.g. Vacation, Wedding", "hu": "pl. Nyaralás, Esküvő"},
    "free_comment":      {"en": "Free comment …", "hu": "Szabad megjegyzés…"},

    # ── Preview / cluster / sidebar ──────────────────────────────────────
    "select_person_sidebar": {"en": "Select a person from the sidebar",
                              "hu": "Válassz személyt az oldalsávból"},
    "face_count_header": {"en": "{name} — {n} face(s)", "hu": "{name} — {n} arc"},
    "cluster_multiselect_hint": {"en": "Click faces to select them; selected faces can be moved together.",
                                 "hu": "Kattints az arcokra a kijelöléshez; a kijelölt arcok együtt áthelyezhetők."},
    "face_tooltip":      {"en": "<b>{person}</b><br>Face #{id} · confidence {confidence:.2f}<br>Backend: {backend}<br>File: {file}",
                          "hu": "<b>{person}</b><br>Arc #{id} · konfidencia {confidence:.2f}<br>Backend: {backend}<br>Fájl: {file}"},
    "all_faces":         {"en": "All Faces", "hu": "Összes arc"},
    "recluster_tip":     {"en": "Recognize unresolved faces from labeled people",
                          "hu": "Névtelen arcok felismerése a megjelölt személyek alapján"},
    "preview_empty":     {"en": "Click a face thumbnail to preview",
                          "hu": "Kattints egy arc bélyegképre az előnézethez"},
    "preview_tip":       {"en": "Click a face to select it\nRight-click for options\nClick empty area to zoom",
                          "hu": "Kattints egy arcra a kijelöléséhez\nJobb klikk a menühöz\nÜres területen kattints a nagyításhoz"},
    "open_file_manager": {"en": "Open in File Manager", "hu": "Megnyitás fájlkezelőben"},
    "zoom":              {"en": "Zoom", "hu": "Nagyítás"},
    "selection":         {"en": "Selection", "hu": "Kijelölés"},
    "modify_selection":  {"en": "Modify Selection", "hu": "Kijelölés módosítása"},
    "assign_to_person":  {"en": "Assign to Person …", "hu": "Személyhez adás…"},
    "delete_selection":  {"en": "Delete Selection", "hu": "Kijelölés törlése"},
    "unknown_face":      {"en": "Unknown face", "hu": "Ismeretlen arc"},
    "overlay_bboxes":    {"en": "Boxes", "hu": "Keretek"},
    "overlay_labels":    {"en": "Labels", "hu": "Nevek"},
    "overlay_bbox_lbl":  {"en": "Box:", "hu": "Keret:"},
    "overlay_label_lbl": {"en": "Label:", "hu": "Név:"},
    "overlay_bbox_tip":  {"en": "Bounding box opacity", "hu": "Keretek áttetszősége"},
    "overlay_label_tip": {"en": "Label opacity", "hu": "Nevek áttetszősége"},
    "prev_image":        {"en": "← Prev", "hu": "← Előző"},
    "next_image":        {"en": "Next →", "hu": "Következő →"},

    # ── Overlapping unknown face cleanup ─────────────────────────────────
    "overlap_dialog_title": {"en": "Overlapping / Duplicate Face Boxes",
                             "hu": "Átfedő / duplikált arckeretek"},
    "overlap_summary":   {"en": "{images} image(s) examined, {matches} suspicious overlap(s) found.",
                          "hu": "{images} kép vizsgálva, {matches} gyanús átfedés találat."},
    "overlap_col_image": {"en": "Image", "hu": "Kép"},
    "overlap_col_preview": {"en": "Preview", "hu": "Előnézet"},
    "overlap_col_person": {"en": "Reference person", "hu": "Referencia személy"},
    "overlap_col_unknown_id": {"en": "Candidate face ID", "hu": "Jelölt arc ID"},
    "overlap_col_known_id": {"en": "Reference face ID", "hu": "Referencia arc ID"},
    "overlap_col_overlap": {"en": "Overlap", "hu": "Átfedés"},
    "overlap_col_delete": {"en": "Delete?", "hu": "Törölhető"},
    "overlap_delete_checkbox_tip": {"en": "Delete this candidate face record",
                                    "hu": "Ennek a jelölt face rekordnak a törlése"},
    "overlap_select_all": {"en": "Select All", "hu": "Mind kijelölése"},
    "overlap_select_none": {"en": "Clear Selection", "hu": "Mind törlése a kijelölésből"},
    "overlap_open_image": {"en": "Open Image", "hu": "Kép megnyitása"},
    "overlap_delete_selected": {"en": "Delete Selected", "hu": "Kijelöltek törlése"},
    "overlap_no_selection_title": {"en": "No Selection", "hu": "Nincs kijelölés"},
    "overlap_no_selection_msg": {"en": "No candidate face box is selected for deletion.",
                                 "hu": "Nincs törlésre kijelölt jelölt arckeret."},
    "overlap_search_error": {"en": "Could not search overlapping face boxes:\n{error}",
                             "hu": "Nem sikerült az átfedő arckeretek keresése:\n{error}"},
    "overlap_no_matches_title": {"en": "No Overlaps Found", "hu": "Nincs átfedő találat"},
    "overlap_no_matches_msg": {"en": "{images} image(s) examined. No overlapping or potentially duplicated faces were found.",
                               "hu": "{images} kép vizsgálva. Nem található átfedő vagy potenciálisan duplikált arc."},
    "overlap_status_found": {"en": "Overlap search: {images} image(s), {matches} candidate(s)",
                             "hu": "Átfedés keresés: {images} kép, {matches} találat"},
    "overlap_confirm_title": {"en": "Delete Overlapping Face Boxes",
                              "hu": "Átfedő arckeretek törlése"},
    "overlap_confirm_msg": {"en": "Delete {n} selected overlapping face record(s)?",
                            "hu": "Biztosan törlöd a kijelölt {n} átfedő arckeret rekordot?"},
    "overlap_delete_error": {"en": "Could not delete selected face boxes:\n{error}",
                             "hu": "Nem sikerült a kijelölt arckeretek törlése:\n{error}"},
    "overlap_deleted_status": {"en": "Deleted {n} overlapping face record(s)",
                               "hu": "{n} átfedő face rekord törölve"},
    "overlap_delete_skipped_msg": {"en": "{n} selected face record(s) were skipped because they no longer exist or are no longer eligible.",
                                   "hu": "{n} kijelölt face rekord kihagyva, mert már nem létezik vagy már nem törölhető."},
    "overlap_preview_unavailable": {"en": "No preview", "hu": "Nincs előnézet"},
    "overlap_preview_known_label": {"en": "reference", "hu": "referencia"},
    "overlap_preview_unknown_label": {"en": "candidate", "hu": "jelölt"},

    # ── Toolbar — short labels (no emoji) ────────────────────────────────
    "tb_export":              {"en": "Export",       "hu": "Export"},

    # ── Export dialog — collage sections ──────────────────────────────────
    "export_collage_import_group": {"en": "Collage Import",
                                    "hu": "Kollázs készítése"},
    "export_collage_import_desc":  {"en": "Import a Picasa collage file (.cxf / .cfx) and "
                                          "add it to the collage view.",
                                    "hu": "Picasa kollázs fájl (.cxf / .cfx) beolvasása és "
                                          "hozzáadása a kollázs nézethez."},
    "export_collage_import_btn":   {"en": "Import Collage …",
                                    "hu": "Kollázs importálása …"},
    "export_collage_html_group":   {"en": "Collage HTML Gallery",
                                    "hu": "Kollázs HTML galéria készítése"},
    "export_collage_html_desc":    {"en": "Export all collages as a static HTML gallery "
                                          "that can be opened in any web browser.",
                                    "hu": "Az összes kollázs exportálása statikus HTML galériaként, "
                                          "amely bármely böngészőben megnyitható."},
    "export_collage_html_btn":     {"en": "Export HTML Gallery …",
                                    "hu": "HTML galéria exportálása …"},

    # ── Image metadata export ─────────────────────────────────────────────
    "export_metadata_group":       {"en": "Image Metadata Export",
                                    "hu": "Kép metaadat export"},
    "export_metadata_desc":        {"en": "Export per-image metadata (persons, date, location, GPS) "
                                          "to CSV or XLSX for archiving, research, or external processing.",
                                    "hu": "Képenkénti metaadatok (személyek, dátum, helyszín, GPS) "
                                          "exportálása CSV vagy XLSX formátumba archiválási, kutatási "
                                          "vagy külső feldolgozási célokra."},
    "export_metadata_format":      {"en": "Format:", "hu": "Formátum:"},
    "export_metadata_csv":         {"en": "CSV", "hu": "CSV"},
    "export_metadata_xlsx":        {"en": "XLSX (Excel)", "hu": "XLSX (Excel)"},
    "export_metadata_person_mode": {"en": "Persons column:", "hu": "Személyek oszlop:"},
    "export_metadata_persons_list": {"en": "One field, comma-separated",
                                     "hu": "Egy mező, vesszővel elválasztva"},
    "export_metadata_persons_cols": {"en": "Separate columns (Person1, Person2 …)",
                                     "hu": "Külön oszlopok (Person1, Person2 …)"},
    "export_metadata_fields":      {"en": "Fields to include:", "hu": "Exportálandó mezők:"},
    "export_metadata_filename":    {"en": "Filename", "hu": "Fájlnév"},
    "export_metadata_relpath":     {"en": "Relative path", "hu": "Relatív útvonal"},
    "export_metadata_persons":     {"en": "Persons", "hu": "Személyek"},
    "export_metadata_date":        {"en": "Date", "hu": "Dátum"},
    "export_metadata_location":    {"en": "Location", "hu": "Helyszín"},
    "export_metadata_gps":         {"en": "GPS coordinates", "hu": "GPS koordináták"},
    "export_metadata_btn":         {"en": "Export …", "hu": "Exportálás …"},
    "export_metadata_done":        {"en": "Export Complete", "hu": "Export kész"},
    "export_metadata_saved":       {"en": "Saved to:\n{path}", "hu": "Mentve:\n{path}"},
    "export_metadata_no_fields":   {"en": "Please select at least one field to export.",
                                    "hu": "Válasszon legalább egy exportálandó mezőt."},
    "export_metadata_no_fields_title": {"en": "No Fields Selected", "hu": "Nincs mező kiválasztva"},

    # ── Face metadata embedding (into image files / sidecar JSON) ─────────
    "fmeta_group":        {"en": "Embed Persons Into Image Files",
                           "hu": "Személyek beágyazása a képfájlokba"},
    "fmeta_desc":         {"en": "Write the recognised persons (id, name, face box) into each "
                                 "image's own metadata (XMP/EXIF), or a JSON file next to it.",
                           "hu": "A felismert személyeket (azonosító, név, arc-pozíció) a képek "
                                 "saját metaadatába (XMP/EXIF) vagy egy mellettük lévő JSON fájlba írja."},
    "fmeta_warning":      {"en": "Person names, ids and face positions may be written into the image "
                                 "file's metadata or a JSON file next to it. Other programs or people "
                                 "can read these if you share the image.",
                           "hu": "A személynevek, azonosítók és arcpozíciók bekerülhetnek a képfájl "
                                 "metaadataiba vagy egy mellette lévő JSON fájlba. Ezeket más programok "
                                 "vagy más személyek is elolvashatják, ha a képet továbbküldöd."},
    "fmeta_confirm_title": {"en": "Embed person metadata?", "hu": "Személy-metaadat beágyazása?"},
    "fmeta_opt_name":     {"en": "Include person name", "hu": "Személynév mentése"},
    "fmeta_opt_notes":    {"en": "Include notes", "hu": "Megjegyzések mentése"},
    "fmeta_opt_sidecar":  {"en": "Only write sidecar JSON (never modify images)",
                           "hu": "Csak sidecar JSON készüljön (a képek soha ne módosuljanak)"},
    "fmeta_btn_current":  {"en": "📝  Save current image", "hu": "📝  Aktuális kép mentése"},
    "fmeta_btn_all":      {"en": "📝  Save all images", "hu": "📝  Összes kép mentése"},
    "fmeta_no_current":   {"en": "No image is currently selected.", "hu": "Nincs kiválasztott kép."},
    "fmeta_done_title":   {"en": "Metadata Export Complete", "hu": "Metaadat-export kész"},
    "fmeta_summary":      {"en": "Processed: {total}\nEmbedded: {embedded}\nSidecar JSON: {sidecar}\n"
                                 "Skipped: {skipped}\nFailed: {failed}",
                           "hu": "Feldolgozva: {total}\nKépbe írva: {embedded}\nSidecar JSON: {sidecar}\n"
                                 "Kihagyva: {skipped}\nHiba: {failed}"},
    "fmeta_summary_cancelled": {"en": "Export cancelled by you.\n\nProcessed: {total}\n"
                                      "Not processed: {remaining}\nEmbedded: {embedded}\n"
                                      "Sidecar JSON: {sidecar}\nSkipped: {skipped}\nFailed: {failed}",
                                "hu": "Az exportot megszakítottad.\n\nFeldolgozva: {total}\n"
                                      "Kimaradt: {remaining}\nKépbe írva: {embedded}\n"
                                      "Sidecar JSON: {sidecar}\nKihagyva: {skipped}\nHiba: {failed}"},
    "fmeta_export_starting": {"en": "Starting export…", "hu": "Export indítása…"},
    "fmeta_cancel":       {"en": "Cancel export", "hu": "Export megszakítása"},
    "fmeta_cancelling":   {"en": "Cancelling — finishing the current image…",
                           "hu": "Megszakítás — az aktuális kép befejezése…"},
    "fmeta_processing":   {"en": "Processing {done}/{total}: {name}",
                           "hu": "Feldolgozás {done}/{total}: {name}"},

    # ── Status bar update notification ────────────────────────────────────
    "status_update_available":     {"en": "New version available: v{version}  ↑",
                                    "hu": "Új verzió elérhető: v{version}  ↑"},

    # ── Image browser panel ───────────────────────────────────────────────
    "ibp_prev":               {"en": "< Prev",        "hu": "< Előző"},
    "ibp_next":               {"en": "Next >",        "hu": "Következő >"},
    "ibp_manual_mark":        {"en": "Manual Mark",   "hu": "Kézi jelölés"},
    "ibp_fullscreen":         {"en": "Fullscreen",    "hu": "Teljes képernyő"},
    "ibp_exit_fullscreen":    {"en": "Exit FS",       "hu": "Kilépés"},
    "ibp_draw_hint_add":      {"en": "Drag on image to mark face — saves automatically",
                               "hu": "Húzd az egeret az arc körül a jelöléshez — mentés automatikus"},
    "ibp_draw_hint_edit":     {"en": "Redraw the face rectangle — replaces the old one",
                               "hu": "Rajzold újra az arc körüli téglalapot — a régi helyére kerül"},
    "ibp_folder_hdr":         {"en": "Folder:",         "hu": "Mappa:"},
    "ibp_filename_hdr":       {"en": "File:",           "hu": "Fájl:"},
    "ibp_date_hdr":           {"en": "Photo date / period:",
                               "hu": "Kép dátuma / időszaka:"},
    "ibp_date_placeholder":   {"en": "e.g. 1954  or  1954.03.12  or  1930s",
                               "hu": "pl. 1954  vagy  1954.03.12  vagy  1930-as évek"},
    "ibp_exif_suggestion":    {"en": "EXIF suggestion:",
                               "hu": "EXIF javaslat:"},
    "ibp_accept_exif":        {"en": "Accept",          "hu": "Elfogad"},
    "ibp_accept_exif_tooltip":{"en": "Copy the EXIF date suggestion into the date field",
                               "hu": "EXIF dátum javaslat átvétele a dátum mezőbe"},
    "ibp_face_hdr":           {"en": "Selected face:",    "hu": "Kiválasztott arc:"},
    "ibp_click_face":         {"en": "Click a face in the image",
                               "hu": "Kattints egy arcra a képen"},
    "ibp_identified":         {"en": "Identified person:", "hu": "Beazonosított személy:"},
    "ibp_not_identified":     {"en": "This face is not categorised",
                               "hu": "Ez az arc nincs kategorizálva"},
    "ibp_no_face_detected":   {"en": "No face detected in this image",
                               "hu": "Nincs felismert arc ezen a képen"},
    "ibp_rename_btn":         {"en": "Rename",         "hu": "Átnevezés"},
    "ibp_person_info_btn":    {"en": "Person Details", "hu": "Személyadatok"},
    "ibp_rename_tooltip":     {"en": "Rename person",   "hu": "Személy átnevezése"},
    "ibp_person_info_tooltip":{"en": "Edit person details", "hu": "Személyadatok szerkesztése"},
    "ibp_rename_placeholder": {"en": "New name…",       "hu": "Új név…"},
    "ibp_assign_hdr":         {"en": "Assign to existing person:",
                               "hu": "Hozzárendelés meglévő személyhez:"},
    "ibp_assign_btn":         {"en": "Assign",          "hu": "Hozzárendelés"},
    "pss_search_placeholder": {"en": "Search person…",  "hu": "Személy keresése…"},
    "pss_no_results":         {"en": "No results",      "hu": "Nincs találat"},
    "pss_match_sort":         {"en": "Order by face match", "hu": "Arc-egyezőség szerinti sorrend"},
    "pss_match_percent":      {"en": "{name} – {pct}%",  "hu": "{name} – {pct}%"},
    "ibp_new_hdr":            {"en": "Create new person:", "hu": "Új személy létrehozása:"},
    "ibp_new_placeholder":    {"en": "Person name…",    "hu": "Személy neve…"},
    "ibp_create_btn":         {"en": "Create & Assign", "hu": "Létrehozás és hozzárendelés"},
    "ibp_no_images":          {"en": "No images in database",
                               "hu": "Nincs kép az adatbázisban"},
    "ibp_load_error":         {"en": "Cannot load:\n{path}",
                               "hu": "Nem tölthető be:\n{path}"},
    "ibp_empty_name_title":   {"en": "Empty Name",      "hu": "Üres név"},
    "ibp_empty_name_msg":     {"en": "Person name cannot be empty.",
                               "hu": "A személynév nem lehet üres."},
    "ibp_ctx_assign_person":  {"en": "Assign to person…", "hu": "Személyhez adás…"},
    "ibp_ctx_edit_bbox":      {"en": "Edit bbox",        "hu": "Bbox módosítása"},
    "ibp_ctx_edit_frame":     {"en": "Edit bounding box", "hu": "Keret szerkesztése"},
    "ibp_ctx_delete":         {"en": "Delete",           "hu": "Törlés"},
    "ibp_ctx_unknown_face":   {"en": "Unknown face",     "hu": "Ismeretlen arc"},
    "ibp_ctx_manual_mark":    {"en": "Manual face selection", "hu": "Kézi arc kijelölés"},
    # Bbox interactive editor
    "ibp_bbox_edit_hint":     {
        "en": "Drag handles to resize · drag centre to move · Enter = apply · Esc = cancel",
        "hu": "Húzd a fogópontokat az átméretezéshez · középső = mozgatás · Enter = alkalmaz · Esc = mégse",
    },
    "ibp_bbox_apply":         {"en": "Apply",            "hu": "Alkalmaz"},
    "ibp_bbox_cancel_edit":   {"en": "Cancel",           "hu": "Mégse"},
    "ibp_select_hint":        {"en": "Select a folder and run a scan",
                               "hu": "Válassz mappát és futtass beolvasást"},
    "ibp_manual_mark_tooltip": {"en": "Click, then drag on the image to manually mark a face",
                                "hu": "Kattints, majd húzd az egeret a képen egy arc kézi megjelöléséhez"},
    "ibp_fullscreen_tooltip": {"en": "Full-screen view",
                               "hu": "Teljes képernyős nézet"},
    "ibp_exit_fullscreen_tooltip": {"en": "Exit full-screen view",
                                    "hu": "Kilépés a teljes képernyős nézetből"},
    "ibp_date_tooltip":      {"en": "The date or period when the image was taken (free text)",
                              "hu": "A kép készítésének dátuma vagy időszaka (szabad szöveg)"},
    "ibp_place_filter_placeholder": {"en": "Filter by place…", "hu": "Szűrés hely szerint…"},
    "ibp_place_filter_btn": {"en": "Filter", "hu": "Szűr"},
    "ibp_place_filter_clear": {"en": "All", "hu": "Mind"},
    "ibp_place_filter_missing_title": {"en": "Place not found", "hu": "Nincs ilyen hely"},
    "ibp_place_filter_missing_msg": {"en": "No place named “{name}”.",
                                     "hu": "Nincs „{name}” nevű hely."},
    "ibp_place_hdr": {"en": "Place:", "hu": "Hely:"},
    "ibp_place_none": {"en": "No place assigned", "hu": "Nincs hely hozzárendelve"},
    "ibp_place_assign_btn": {"en": "Assign", "hu": "Hozzárendel"},
    "ibp_place_create_btn": {"en": "Create / assign typed", "hu": "Beírt létrehozása"},
    "ibp_place_need_name": {"en": "Type a place name first.",
                            "hu": "Először írj be egy helynevet."},
    "ibp_place_rename_placeholder": {"en": "Name anonymous GPS place…",
                                     "hu": "Névtelen GPS-hely neve…"},
    "ibp_place_rename_btn": {"en": "Rename place", "hu": "Hely átnevezése"},
    "ibp_place_coords_label": {"en": "Place GPS: {lat:.6f}, {lon:.6f}",
                               "hu": "Helyszín GPS: {lat:.6f}, {lon:.6f}"},
    "ibp_place_coords_none":  {"en": "Place GPS: —", "hu": "Helyszín GPS: —"},
    "place_search_placeholder": {"en": "Search place…", "hu": "Hely keresése…"},

    # ── Image GPS coordinates ───────────────────────────────────────────
    "ibp_gps_hdr":               {"en": "Image GPS coordinates:",
                                  "hu": "Kép GPS koordinátája:"},
    "ibp_gps_placeholder":       {"en": "lat, lon  e.g. 46.818068, 17.785863",
                                  "hu": "szél., hossz.  pl. 46.818068, 17.785863"},
    "ibp_gps_save_btn":          {"en": "Save GPS", "hu": "GPS mentése"},
    "ibp_gps_clear_btn":         {"en": "Clear", "hu": "Törlés"},
    "ibp_gps_source_exif":       {"en": "Source: EXIF", "hu": "Forrás: EXIF"},
    "ibp_gps_source_manual":     {"en": "Source: manual", "hu": "Forrás: kézi"},
    "ibp_gps_source_place":      {"en": "Source: inherited from place",
                                  "hu": "Forrás: helyszíntől örökölt"},
    "ibp_gps_source_none":       {"en": "Source: none", "hu": "Forrás: nincs"},
    "ibp_gps_invalid":           {"en": "Invalid coordinates — not saved",
                                  "hu": "Érvénytelen koordináta — nem mentve"},
    "ibp_gps_write_exif_place":  {"en": "Write place GPS to EXIF",
                                  "hu": "Helyszín GPS írása EXIF-be"},
    "ibp_gps_write_exif_place_tip": {
        "en": "Write the linked place's coordinates into this image's EXIF GPS fields",
        "hu": "A hozzárendelt helyszín koordinátájának beírása a kép EXIF GPS mezőibe",
    },

    # ── Image note ──────────────────────────────────────────────────────
    "ibp_note_hdr":         {"en": "Note:",                "hu": "Megjegyzés:"},
    "ibp_note_placeholder": {"en": "Add a note…",          "hu": "Megjegyzés hozzáadása…"},
    "ibp_note_tooltip":     {"en": "Free-text note attached to this image",
                             "hu": "Szabad szöveges megjegyzés a képhez"},

    # ── EXIF date update button ─────────────────────────────────────────
    "ibp_update_exif_date_btn":  {"en": "Update EXIF date from photo date",
                                  "hu": "EXIF készítési dátum frissítése a kép dátuma alapján"},
    "ibp_update_exif_date_tip":  {
        "en": "Write the photo date into the image EXIF DateTimeOriginal field",
        "hu": "A kép dátumának beírása az EXIF DateTimeOriginal mezőbe",
    },

    # ── Places / Locations panel ────────────────────────────────────────
    "places_filter_name": {"en": "Name…", "hu": "Név…"},
    "places_filter_person_id": {"en": "Person id…", "hu": "Személy id…"},
    "places_filter_date_from": {"en": "Date from…", "hu": "Dátumtól…"},
    "places_filter_date_to": {"en": "Date to…", "hu": "Dátumig…"},
    "places_filter_min_images": {"en": "Min images ", "hu": "Min képek "},
    "places_filter_coords_any": {"en": "Any coordinates", "hu": "Bármilyen koordináta"},
    "places_filter_coords_yes": {"en": "Has coordinates", "hu": "Van koordináta"},
    "places_filter_coords_no": {"en": "No coordinates", "hu": "Nincs koordináta"},
    "places_filter_anon": {"en": "Anonymous EXIF", "hu": "Névtelen EXIF"},
    "places_filter_apply": {"en": "Apply", "hu": "Alkalmaz"},
    "places_name": {"en": "Name", "hu": "Név"},
    "places_coords": {"en": "Coordinates", "hu": "Koordináta"},
    "places_coords_edit_tip": {
        "en": "Double-click to edit coordinates (e.g. 47.4979, 19.0402)",
        "hu": "Dupla kattintás a koordináták szerkesztéséhez (pl. 47.4979, 19.0402)",
    },
    "places_coords_invalid_title": {"en": "Invalid coordinates", "hu": "Érvénytelen koordináta"},
    "places_coords_invalid_msg": {
        "en": "Could not save coordinates: {error}",
        "hu": "A koordináták mentése sikertelen: {error}",
    },
    "places_image_count": {"en": "Images", "hu": "Képek"},
    "places_person_count": {"en": "Persons", "hu": "Személyek"},
    "places_source": {"en": "Source", "hu": "Forrás"},
    "places_type": {"en": "Type", "hu": "Típus"},
    "places_type_exact": {"en": "Exact place", "hu": "Pontos hely"},
    "places_type_area": {"en": "Area", "hu": "Tágabb hely"},
    "places_type_region": {"en": "Region", "hu": "Régió"},
    "places_filter_type_any": {"en": "Any type", "hu": "Bármilyen típus"},
    "places_accuracy_radius": {"en": "Accuracy radius (m)", "hu": "Pontossági sugár (m)"},
    "places_parent": {"en": "Parent place", "hu": "Szülő hely"},
    "places_parent_none": {"en": "(no parent — top level)", "hu": "(nincs szülő — legfelső szint)"},
    "places_new_btn": {"en": "New place", "hu": "Új hely"},
    "places_edit_btn": {"en": "Edit", "hu": "Szerkesztés"},
    "place_edit_title_new": {"en": "New place", "hu": "Új hely"},
    "place_edit_title_edit": {"en": "Edit place", "hu": "Hely szerkesztése"},
    "place_edit_name": {"en": "Name", "hu": "Név"},
    "place_edit_type": {"en": "Type", "hu": "Típus"},
    "place_edit_lat": {"en": "Latitude", "hu": "Szélesség"},
    "place_edit_lon": {"en": "Longitude", "hu": "Hosszúság"},
    "place_edit_radius": {"en": "Accuracy radius (m)", "hu": "Pontossági sugár (m)"},
    "place_edit_radius_auto": {"en": "Auto (type default)", "hu": "Automatikus (típus szerint)"},
    "place_edit_parent": {"en": "Parent place", "hu": "Szülő hely"},
    "place_edit_empty_name": {"en": "Name cannot be empty.", "hu": "A név nem lehet üres."},
    "place_edit_settlement": {"en": "Settlement", "hu": "Település"},
    "place_edit_settlement_ph": {"en": "e.g. Balatonszemes", "hu": "pl. Balatonszemes"},
    "place_edit_street": {"en": "Street", "hu": "Utca"},
    "place_edit_street_ph": {"en": "e.g. Bajcsy-Zsilinszky utca", "hu": "pl. Bajcsy-Zsilinszky utca"},
    "place_edit_house": {"en": "House number", "hu": "Házszám"},
    "place_edit_coord_source": {"en": "Coordinate source", "hu": "Koordináta forrása"},
    "place_edit_need_settlement": {
        "en": "Provide a settlement or a name.",
        "hu": "Adj meg települést vagy nevet.",
    },
    "place_edit_hint_area": {
        "en": "This is a broad place (settlement), not a precise address.",
        "hu": "Ez tágabb hely (település), nem pontos cím.",
    },
    "place_edit_hint_manual": {
        "en": "Coordinate set manually from the map.",
        "hu": "A koordináta kézzel, a térképen lett megadva.",
    },
    "place_edit_geocoding_off": {
        "en": "Online geocoding is off — enable it in Settings or pick a point on the map.",
        "hu": "Az online geokódolás ki van kapcsolva — engedélyezd a Beállításokban, vagy jelölj ki pontot a térképen.",
    },
    "place_edit_save_error": {"en": "Failed to save place: {error}",
                              "hu": "Nem sikerült menteni a helyet: {error}"},
    "places_images": {"en": "Images taken there", "hu": "Itt készült képek"},
    "places_persons": {"en": "People linked through faces", "hu": "Arcok alapján kapcsolódó személyek"},
    "places_no_coords": {"en": "No coordinates", "hu": "Nincs koordináta"},
    "places_no_thumbnail": {"en": "No thumbnail", "hu": "Nincs bélyegkép"},
    "places_merge_btn": {"en": "Merge selected", "hu": "Kijelöltek összevonása"},
    "places_refresh_btn": {"en": "Refresh", "hu": "Frissítés"},
    "places_merge_title": {"en": "Merge places", "hu": "Helyek összevonása"},
    "places_merge_need_two": {"en": "Select at least two places to merge.",
                              "hu": "Legalább két helyet jelölj ki az összevonáshoz."},
    "places_merge_keep_name": {"en": "Keep name:", "hu": "Maradó név:"},
    "places_merge_keep_coords": {"en": "Keep coordinates:", "hu": "Maradó koordináta:"},
    "places_merge_keep_thumbnail": {"en": "Keep thumbnail:", "hu": "Maradó bélyegkép:"},
    "places_merge_error": {"en": "Failed to save changes: {error}",
                           "hu": "Nem sikerült menteni: {error}"},
    "places_map_no_gps": {
        "en": "No valid GPS coordinates for this location.",
        "hu": "Ehhez a helyhez nincs érvényes GPS-koordináta.",
    },
    "places_map_offline": {
        "en": "Offline — map tiles unavailable",
        "hu": "Offline — térképcsempék nem elérhetők",
    },
    "places_map_no_webengine": {
        "en": "Map not available (WebEngine module missing).",
        "hu": "Térkép nem elérhető (WebEngine modul hiányzik).",
    },
    "places_gallery_no_images": {"en": "No images", "hu": "Nincs kép"},
    "places_gallery_dbl_click":  {"en": "Double-click to view full size", "hu": "Dupla kattintás a teljes mérethez"},
    "places_gallery_click_to_close": {"en": "Click to close", "hu": "Kattints a bezáráshoz"},
    "places_gallery_set_thumb":  {"en": "Set as place thumbnail", "hu": "Beállítás hely bélyegképének"},
    "places_gallery_clear_thumb": {"en": "Reset to automatic thumbnail", "hu": "Automatikus bélyegkép visszaállítása"},
    "set_as_person_thumbnail":   {"en": "Set as person thumbnail", "hu": "Beállítás személy bélyegképének"},
    "clear_person_thumbnail":    {"en": "Reset to automatic thumbnail", "hu": "Automatikus bélyegkép visszaállítása"},
    "thumbnail_set_ok":          {"en": "Thumbnail updated.", "hu": "Bélyegkép frissítve."},
    "thumbnail_clear_ok":        {"en": "Automatic thumbnail restored.", "hu": "Automatikus bélyegkép visszaállítva."},
    "thumbnail_set_error":       {"en": "Could not set thumbnail: {error}", "hu": "Nem sikerült beállítani a bélyegképet: {error}"},
    "places_rename_btn":     {"en": "Rename", "hu": "Átnevezés"},
    "places_rename_tooltip": {"en": "Edit the name of this place", "hu": "Hely nevének szerkesztése"},
    "places_name_edit_tip":  {"en": "Double-click to rename", "hu": "Dupla kattintás az átnevezéshez"},

    # ── Persons maintenance page (persons_*) ──────────────────────────────
    "tab_persons":            {"en": "Persons", "hu": "Személyek"},
    "persons_filter_name":    {"en": "Name / nickname…", "hu": "Név / becenév…"},
    "persons_filter_code":    {"en": "Family code…", "hu": "Családi kód…"},
    "persons_filter_apply":   {"en": "Apply", "hu": "Alkalmaz"},
    "persons_col_id":         {"en": "ID", "hu": "Azonosító"},
    "persons_col_thumbnail":  {"en": "Thumbnail", "hu": "Bélyegkép"},
    "persons_col_name":       {"en": "Name", "hu": "Név"},
    "persons_col_family_code":{"en": "Family code", "hu": "Családi kód"},
    "persons_col_last_name":  {"en": "Last name", "hu": "Vezetéknév"},
    "persons_col_first_name": {"en": "First name", "hu": "Keresztnév"},
    "persons_col_second_name":{"en": "Second name", "hu": "Második név"},
    "persons_col_nickname":   {"en": "Nickname", "hu": "Becenév"},
    "persons_col_married_name":{"en": "Married name", "hu": "Házassági név"},
    "persons_col_gender":     {"en": "Gender", "hu": "Nem"},
    "persons_col_birth_place":{"en": "Birth place", "hu": "Születési hely"},
    "persons_col_birth_date": {"en": "Birth date", "hu": "Születési dátum"},
    "persons_col_death_place":{"en": "Death place", "hu": "Halálozási hely"},
    "persons_col_death_date": {"en": "Death date", "hu": "Halálozási dátum"},
    "persons_col_notes":      {"en": "Notes", "hu": "Megjegyzés"},
    "persons_col_auto_named": {"en": "Auto-named", "hu": "Automatikusan elnevezett"},
    "persons_col_protected":  {"en": "Protected", "hu": "Védett"},
    "persons_yes":            {"en": "Yes", "hu": "Igen"},
    "persons_no":             {"en": "No", "hu": "Nem"},
    "persons_count":          {"en": "{n} persons", "hu": "{n} személy"},
    "persons_detail_faces":   {"en": "Faces", "hu": "Arcok"},
    "persons_detail_images":  {"en": "Images", "hu": "Képek"},
    "persons_face_count":     {"en": "Faces:", "hu": "Arcok:"},
    "persons_image_count":    {"en": "Images:", "hu": "Képek:"},
    "persons_edit_btn":       {"en": "Edit data", "hu": "Adatok szerkesztése"},
    "persons_rename_btn":     {"en": "Rename", "hu": "Átnevezés"},
    "persons_thumbnail_btn":  {"en": "Change thumbnail", "hu": "Bélyegkép módosítása"},
    "persons_refresh_btn":    {"en": "Refresh", "hu": "Frissítés"},
    "persons_no_thumbnail":   {"en": "No thumbnail", "hu": "Nincs bélyegkép"},
    "persons_no_selection":   {"en": "Select a person", "hu": "Válassz egy személyt"},
    "persons_rename_title":   {"en": "Rename person", "hu": "Személy átnevezése"},
    "persons_rename_prompt":  {"en": "New name:", "hu": "Új név:"},
    "persons_save_error":     {"en": "Could not save: {error}", "hu": "Nem sikerült menteni: {error}"},
    "persons_protected_msg":  {"en": "This person is protected and cannot be renamed or edited this way.",
                               "hu": "Ez a személy védett, így nem nevezhető át vagy módosítható ezzel a művelettel."},
    "persons_thumb_picker_title": {"en": "Choose thumbnail — {name}", "hu": "Bélyegkép választása — {name}"},
    "persons_thumb_picker_hint":  {"en": "Click a face to use it as the thumbnail.",
                                   "hu": "Kattints egy arcra, hogy az legyen a bélyegkép."},
    "persons_thumb_no_faces":     {"en": "This person has no usable face crops.",
                                   "hu": "Ennek a személynek nincs használható kivágott arca."},
    "persons_thumb_set_ok":   {"en": "Thumbnail updated.", "hu": "Bélyegkép frissítve."},
    "persons_name_edit_tip":  {"en": "Double-click to rename", "hu": "Dupla kattintás az átnevezéshez"},

    # ── Batch face move (move selected faces to another person) ────────────
    "persons_faces_hint":     {"en": "Click faces to select them for moving.",
                               "hu": "Kattints az arcokra a kijelöléshez (áthelyezéshez)."},
    "faces_selected_none":    {"en": "No faces selected", "hu": "Nincs kijelölt arc"},
    "faces_selected_n":       {"en": "{n} face(s) selected", "hu": "{n} arc kijelölve"},
    "persons_move_faces_btn": {"en": "Move to another person", "hu": "Áthelyezés másik személyhez"},
    "move_faces_title":       {"en": "Move faces", "hu": "Arcok áthelyezése"},
    "move_faces_header":      {"en": "Move {n} selected face(s) to:",
                               "hu": "{n} kijelölt arc áthelyezése ide:"},
    "move_faces_pick_existing": {"en": "Choose an existing person:",
                                 "hu": "Válassz meglévő személyt:"},
    "move_faces_create_new":  {"en": "…or create a new person:",
                               "hu": "…vagy hozz létre új személyt:"},
    "move_faces_new_placeholder": {"en": "New person's name", "hu": "Új személy neve"},
    "move_faces_confirm":     {"en": "Move", "hu": "Áthelyezés"},
    "move_faces_same_person": {"en": "The faces already belong to this person.",
                               "hu": "A kijelölt arcok már ehhez a személyhez tartoznak."},
    "move_faces_confirm_title": {"en": "Confirm move", "hu": "Áthelyezés megerősítése"},
    "move_faces_confirm_msg": {"en": "Move the {n} selected face(s) from \"{src}\" to \"{dst}\"?",
                               "hu": "Biztosan áthelyezed a kijelölt {n} arcot \"{src}\" személytől \"{dst}\" személyhez?"},
    "move_faces_error_title": {"en": "Could not move faces", "hu": "Az áthelyezés nem sikerült"},
    "move_faces_done":        {"en": "{n} face(s) moved to {dst}.",
                               "hu": "{n} arc áthelyezve {dst} személyhez."},
    "move_faces_undo":        {"en": "Undo", "hu": "Visszavonás"},
    "move_faces_undone":      {"en": "Move undone.", "hu": "Áthelyezés visszavonva."},
    "move_faces_undo_error":  {"en": "Could not undo the move: {error}",
                               "hu": "Az áthelyezés visszavonása nem sikerült: {error}"},

    # ── 3-column image browser (ib3_*) ────────────────────────────────────
    "ib3_folders_hdr":         {"en": "Folders",           "hu": "Mappák"},
    "ib3_folders_hdr_n":       {"en": "Folders ({n})",     "hu": "Mappák ({n})"},
    "ib3_no_folders":          {"en": "No images in database",
                                "hu": "Nincs kép az adatbázisban"},
    "ib3_select_folder_hint":  {"en": "← Select a folder",
                                "hu": "← Válassz mappát"},
    "ib3_select_image_hint":   {"en": "← Select an image",
                                "hu": "← Válassz képet"},
    "ib3_no_images_in_folder": {"en": "No images in this folder",
                                "hu": "Nincs kép ebben a mappában"},
    "ib3_toggle_left_tip":     {"en": "Show / hide the folder panel",
                                "hu": "Mappa panel mutatása / elrejtése"},
    "ib3_toggle_right_tip":    {"en": "Show / hide the info panel",
                                "hu": "Infó panel mutatása / elrejtése"},
    "ib3_save_meta_btn":       {"en": "💾  Save EXIF data",
                                "hu": "💾  Exif adatok mentése"},
    "ib3_save_meta_tip":       {"en": "Write everything about this image into its metadata: "
                                      "recognised persons + face boxes (XMP/EXIF or sidecar JSON), "
                                      "GPS coordinates and the photo date (EXIF). If the image has a "
                                      "colourised/B&W pair, both copies are updated.",
                                "hu": "Mindent a kép metaadatába ír: a felismert személyeket + "
                                      "arc-pozíciókat (XMP/EXIF vagy sidecar JSON), a GPS-koordinátákat "
                                      "és a fénykép dátumát (EXIF). Ha a képnek van színezett/fekete-fehér "
                                      "párja, mindkét példány frissül."},
    "ib3_save_meta_summary":   {"en": "Processed: {total}\nPersons embedded: {embedded}\n"
                                      "Sidecar JSON: {sidecar}\nGPS written: {gps}\nDate written: {date}\n"
                                      "Failed: {failed}",
                                "hu": "Feldolgozva: {total}\nSzemélyek beágyazva: {embedded}\n"
                                      "Sidecar JSON: {sidecar}\nGPS mentve: {gps}\nDátum mentve: {date}\n"
                                      "Hiba: {failed}"},

    # ── Settings tabs ────────────────────────────────────────────────────
    "settings_tab_general":     {"en": "General",               "hu": "Általános"},
    "settings_tab_pairing":     {"en": "Image Pairing",         "hu": "Képpárosítás"},
    "settings_tab_quality":     {"en": "Face Quality",          "hu": "Arc minőség"},
    "settings_tab_shortcuts":   {"en": "Shortcuts",             "hu": "Billentyűparancsok"},

    # ── Face quality filter settings ──────────────────────────────────────
    "fq_group":             {"en": "Face Quality Filter",
                             "hu": "Arc minőségszűrő"},
    "fq_exclude_toggle":    {"en": "Exclude low-quality faces from model building",
                             "hu": "Gyenge minőségű arcok kizárása modellépítésből"},
    "fq_exclude_tip":       {"en": "When enabled, back-facing, blurry, too-small or low-confidence "
                                   "face detections are skipped during embedding, recognition, "
                                   "clustering and name suggestions. "
                                   "The bounding box is still shown and manual name assignment "
                                   "always works regardless of this setting.",
                             "hu": "Ha be van kapcsolva, a háttal álló, homályos, túl kicsi vagy "
                                   "alacsony konfidenciájú arcdetektálások ki lesznek hagyva az "
                                   "embedding, felismerés, klaszterezés és névajánlatok során. "
                                   "A bounding box továbbra is látható marad, és a kézi "
                                   "névhozzárendelés mindig működik függetlenül ettől a beállítástól."},
    "fq_thresholds_group":  {"en": "Quality Thresholds",
                             "hu": "Minőségi küszöbértékek"},
    "fq_min_confidence":    {"en": "Min. detection confidence:",
                             "hu": "Min. detektálási konfidencia:"},
    "fq_min_area":          {"en": "Min. face area (px²):",
                             "hu": "Min. arcterület (px²):"},
    "fq_min_sharpness":     {"en": "Min. sharpness (Laplacian variance):",
                             "hu": "Min. élességi érték (Laplacian variancia):"},
    "fq_reanalyze_btn":     {"en": "Re-evaluate All Faces …",
                             "hu": "Összes arc újraértékelése …"},
    "fq_reanalyze_confirm": {"en": "Re-evaluate quality for all {n} face(s) in the database?\n"
                                   "This may take a few minutes.",
                             "hu": "Újraértékeli a minőséget az adatbázis mind a(z) {n} arcán?\n"
                                   "Ez néhány percig tarthat."},
    "fq_reanalyze_done":    {"en": "Quality re-evaluation complete: {n} face(s) processed.",
                             "hu": "Minőségi újraértékelés kész: {n} arc feldolgozva."},
    "fq_low_quality_tip":   {"en": "Low-quality face — excluded from model building\n"
                                   "(back-facing / blurry / too small / low confidence)\n"
                                   "Right-click to override manually.",
                             "hu": "Gyenge minőségű arc — kizárva a modellépítésből\n"
                                   "(háttal álló / homályos / túl kicsi / alacsony konfidencia)\n"
                                   "Jobb klikk a kézi felülbíráláshoz."},
    "fq_ctx_force_include":  {"en": "Force include in model building",
                              "hu": "Kényszerített belefoglalás a modellépítésbe"},
    "fq_ctx_force_exclude":  {"en": "Exclude from model building",
                              "hu": "Kizárás a modellépítésből"},

    # ── Deoldified pairing ────────────────────────────────────────────────
    "geocoding_group":          {"en": "Address geocoding", "hu": "Cím-geokódolás"},
    "geocoding_enable":         {"en": "Enable online geocoding (Nominatim / OpenStreetMap)",
                                 "hu": "Online geokódolás engedélyezése (Nominatim / OpenStreetMap)"},
    "geocoding_enable_tip":     {"en": "Off by default. When enabled, settlement/street autocomplete "
                                       "and address lookups query OpenStreetMap. Results are cached "
                                       "locally; your own entered addresses always work offline.",
                                 "hu": "Alapból kikapcsolva. Bekapcsolva a település/utca kiegészítés és "
                                       "a címkeresés az OpenStreetMap-et kérdezi. Az eredmények helyben "
                                       "cache-elődnek; a saját beírt címek offline is működnek."},
    "deoldified_group":         {"en": "Deoldified / Colorized Pairing",
                                 "hu": "Deoldified / Színezett képpárosítás"},
    "deoldified_toggle":        {"en": "Automatically pair deoldified (colorized) images",
                                 "hu": "Deoldified képek automatikus párosítása"},
    "deoldified_toggle_tip":    {"en": "When enabled, images containing '-deoldified' in their "
                                       "filename are automatically paired with their original "
                                       "black-and-white counterpart. Face data from the original "
                                       "is shown on both versions.",
                                 "hu": "Ha be van kapcsolva, a '-deoldified' szót tartalmazó "
                                       "képek automatikusan párosítódnak az eredeti "
                                       "fekete-fehér képpel. Az arcadatok az eredetiről "
                                       "mindkét változaton megjelennek."},
    "deoldified_sync_toggle":   {"en": "Automatically copy data between paired images",
                                 "hu": "Adatok automatikus átvétele a párképek közt"},
    "deoldified_sync_toggle_tip": {"en": "When a deoldified pair is opened, copy faces and image "
                                       "data from the side that has them into the empty side. "
                                       "Only runs when exactly one side is empty; never "
                                       "overwrites existing data.",
                                 "hu": "Deoldified pár megnyitásakor az arcok és képadatok "
                                       "átmásolása a kitöltött oldalról az üresbe. Csak akkor "
                                       "fut, ha pontosan az egyik oldal üres; meglévő adatot "
                                       "nem ír felül."},
    "ibp_deol_pair_lbl":        {"en": "View:",               "hu": "Nézet:"},
    "ibp_view_original_bw":     {"en": "Original B&W",        "hu": "Eredeti fekete-fehér"},
    "ibp_view_colorized":       {"en": "Colorized",            "hu": "Színezett változat"},
    "ibp_view_compare":         {"en": "Compare",              "hu": "Összehasonlítás"},
    "ibp_view_compare_tip":     {"en": "Drag the slider to reveal the colorized image on the "
                                       "right and the black-and-white on the left.",
                                 "hu": "Húzd a csúszkát: jobbra a színezett, balra a "
                                       "fekete-fehér változat látszik."},
    "ibp_deol_sync":            {"en": "Copy data from pair",  "hu": "Adatok átvétele a párról"},
    "ibp_deol_sync_tip":        {"en": "Copy faces and image data from the paired image into "
                                       "the empty one. Only runs when exactly one of the two "
                                       "images has data; never overwrites existing data.",
                                 "hu": "Arcok és képadatok másolása a párképről az üres képbe. "
                                       "Csak akkor fut le, ha a kettő közül pontosan az egyiken "
                                       "van adat; meglévő adatot soha nem ír felül."},
    "ibp_deol_sync_done":       {"en": "Copied {faces} face(s) and {fields} metadata field(s).",
                                 "hu": "{faces} arc és {fields} metaadat-mező átmásolva."},
    "ibp_deol_sync_skipped":    {"en": "Nothing copied: both images already have data, or both "
                                       "are empty.",
                                 "hu": "Nem történt másolás: mindkét képen van adat, vagy "
                                       "mindkettő üres."},
    "ibp_deol_sync_no_pair":    {"en": "No paired image found for the current image.",
                                 "hu": "Nincs párkép a jelenlegi képhez."},

    # ── Image library ─────────────────────────────────────────────────────
    "img_lib_group":            {"en": "Image Library",
                                 "hu": "Képkönyvtár"},
    "img_lib_root_label":       {"en": "Library root:",
                                 "hu": "Könyvtár gyökere:"},
    "img_lib_not_configured":   {"en": "(not configured — using absolute paths)",
                                 "hu": "(nincs beállítva — abszolút útvonalakat használ)"},
    "img_lib_status_ok":        {"en": "Connected: {path}",
                                 "hu": "Csatlakoztatva: {path}"},
    "img_lib_status_missing":   {"en": "Root not found: {path}",
                                 "hu": "Gyökér nem található: {path}"},
    "img_lib_change_btn":       {"en": "Change …",
                                 "hu": "Módosítás …"},
    "img_lib_migrate_btn":      {"en": "Migrate to Relative Paths",
                                 "hu": "Relatív útvonalakra konvertálás"},
    "img_lib_migrate_tip":      {"en": "Convert all absolute image paths in the database "
                                       "to relative paths based on the library root. "
                                       "Run this once after setting the root for the first time.",
                                 "hu": "Az adatbázisban tárolt abszolút útvonalak konvertálása "
                                       "relatív útvonalakra a könyvtár gyökere alapján. "
                                       "Ezt egyszer kell futtatni a gyökér első beállítása után."},
    "img_lib_select_title":     {"en": "Select Image Library Root Folder",
                                 "hu": "Képkönyvtár gyökérmappájának kiválasztása"},
    "img_lib_migrate_title":    {"en": "Migrate Image Paths",
                                 "hu": "Képútvonalak migrálása"},
    "img_lib_migrate_confirm":  {"en": "Convert {n} image record(s) to relative paths?\n\n"
                                       "Library root: {root}\n\n"
                                       "The original file_path values are preserved. "
                                       "The migration can be repeated safely.",
                                 "hu": "Konvertálja a(z) {n} képrekordot relatív útvonalakra?\n\n"
                                       "Könyvtár gyökere: {root}\n\n"
                                       "Az eredeti fájlútvonalak megmaradnak. "
                                       "A migráció biztonságosan ismételhető."},
    "img_lib_migrating":        {"en": "Migrating image paths …",
                                 "hu": "Képútvonalak migrálása …"},
    "img_lib_migrate_done":     {"en": "Migration complete: {migrated} converted, {skipped} skipped.",
                                 "hu": "Migráció kész: {migrated} konvertálva, {skipped} kihagyva."},
    "img_lib_migrate_errors":   {"en": "Files not found during migration ({n}):",
                                 "hu": "Migráció során nem talált fájlok ({n}):"},
    "img_lib_missing_title":    {"en": "Image Library Not Found",
                                 "hu": "Képkönyvtár nem található"},
    "img_lib_missing_msg":      {"en": "The configured image library root was not found:\n\n"
                                       "{path}\n\n"
                                       "Images cannot be loaded until a valid root is selected. "
                                       "Choose the new location of your image folder, or skip to "
                                       "continue with reduced functionality.",
                                 "hu": "A beállított képkönyvtár gyökere nem található:\n\n"
                                       "{path}\n\n"
                                       "A képek nem tölthetők be érvényes gyökér nélkül. "
                                       "Válassza ki a képkönyvtár új helyét, vagy ugorja át "
                                       "a korlátozott működés folytatásához."},
    "img_lib_find_btn":         {"en": "Find New Location …",
                                 "hu": "Új helyszín keresése …"},
    "img_lib_skip_btn":         {"en": "Skip for Now",
                                 "hu": "Kihagyás egyelőre"},
    "img_lib_root_changed":     {"en": "Image library root updated.",
                                 "hu": "Képkönyvtár gyökere frissítve."},
    "img_lib_n_relative":       {"en": "{n} image(s) with relative paths",
                                 "hu": "{n} kép relatív útvonallal"},
    "img_lib_n_absolute":       {"en": "{n} image(s) with absolute paths only",
                                 "hu": "{n} kép csak abszolút útvonallal"},
    "img_lib_no_root_for_migrate": {"en": "Please set the library root before migrating.",
                                    "hu": "A migráció előtt állítsa be a könyvtár gyökerét."},

    # ── Scan modes dialog ─────────────────────────────────────────────────
    "scanModes.title":
        {"en": "Scan & Maintenance",
         "hu": "Beolvasás és karbantartás"},
    "scanModes.openButton":
        {"en": "Scan & Maintenance …",
         "hu": "Beolvasás és karbantartás …"},
    "scanModes.close":
        {"en": "Close",
         "hu": "Bezárás"},

    # incremental (scan & index)
    "scanModes.incremental.title":
        {"en": "Scan & Index",
         "hu": "Beolvasás és indexelés"},
    "scanModes.incremental.description":
        {"en": "Scans the selected folder for new images that are not yet in the database. "
               "Runs face detection, embedding and recognition only on these new files. "
               "Existing faces and person assignments are left completely untouched. "
               "Fast and safe — ideal for daily incremental updates after adding new photos.",
         "hu": "Csak az adatbázisban még nem szereplő új képeket dolgozza fel. "
               "Arcdetektálást, embeddinget és felismerést csak az új fájlokon futtat. "
               "A meglévő arcok és személyesítések érintetlenek maradnak. "
               "Gyors és biztonságos — mindennapi használatra, ha új fotókat adtál hozzá."},
    "scanModes.incremental.startButton":
        {"en": "Start Scan",
         "hu": "Beolvasás indítása"},

    # full rescan
    "scanModes.fullRescan.title":
        {"en": "Force Full Rescan",
         "hu": "Teljes újrabeolvasás"},
    "scanModes.fullRescan.description":
        {"en": "Deletes ALL automatically detected faces — including those already assigned to "
               "a person — and re-runs fast face detection on every image from scratch. "
               "Only manually drawn faces are preserved. "
               "Use this when the detector has changed significantly or the entire face "
               "database needs to be rebuilt. "
               "⚠ Warning: all person assignments based on auto-detected faces will be lost.",
         "hu": "Törli az ÖSSZES automatikusan felismert arcot — beleértve a személyekhez már "
               "hozzárendelteket is —, majd újra futtatja a gyors arcdetektálást minden képen. "
               "Csak a kézzel rajzolt arcok maradnak meg. "
               "Akkor érdemes használni, ha a detektor alapvetően megváltozott, vagy "
               "az egész arcadatbázist újra kell építeni. "
               "⚠ Figyelem: az automatikusan felismert arcokhoz tartozó személyesítések elvesznek."},
    "scanModes.fullRescan.startButton":
        {"en": "Start Full Rescan",
         "hu": "Teljes újrabeolvasás indítása"},
    "scanModes.fullRescan.warning":
        {"en": "Destructive — named face assignments will be lost",
         "hu": "Romboló — az elnevezett arc-hozzárendelések elvesznek"},

    # face rescan (fast)
    "scanModes.faceRescan.title":
        {"en": "Re-detect Faces (Fast)",
         "hu": "Arcok újrakeresése (gyors)"},
    "scanModes.faceRescan.description":
        {"en": "Removes unnamed, unassigned auto-detected faces and resets all images for "
               "re-detection. Faces that are already assigned to a named person are kept as "
               "training examples. Re-runs fast face detection on all images. "
               "Use this to find faces the detector previously missed while keeping your "
               "manual person assignments intact.",
         "hu": "Törli a névtelen, hozzá nem rendelt automatikus arcokat, és minden képet "
               "újra feldolgozásra jelöl. A már személyhez rendelt arcok megmaradnak "
               "tanítási példaként. Gyors módban újra futtatja az arcdetektálást minden képen. "
               "Akkor hasznos, ha a detektor korábban hiányos volt, de a kézi "
               "személyesítéseket meg akarod tartani."},
    "scanModes.faceRescan.startButton":
        {"en": "Start Re-detection (Fast)",
         "hu": "Újrakeresés indítása (gyors)"},

    # face rescan (accurate)
    "scanModes.preciseRescan.title":
        {"en": "Re-detect Faces (Accurate)",
         "hu": "Arcok újrakeresése (pontos)"},
    "scanModes.preciseRescan.description":
        {"en": "Same as fast re-detection but uses high-accuracy mode: runs multiple "
               "preprocessing variants per image (CLAHE contrast, gamma brightening, "
               "histogram equalisation, bilateral filtering) with a lower confidence "
               "threshold, then merges duplicate bounding boxes. "
               "Finds significantly more faces, especially in dark or low-contrast photos. "
               "⚠ Considerably slower and more resource-intensive than fast mode.",
         "hu": "Ugyanaz mint a gyors újrakeresés, de pontos módban: képenként több "
               "képfeldolgozási variációt futtat (CLAHE kontraszt, gamma-fényesítés, "
               "hisztogram-kiegyenlítés, bilaterális szűrés), alacsonyabb konfidencia-küszöbbel, "
               "majd összevonja az átfedő bounding box-okat. "
               "Lényegesen több arcot talál, különösen sötét vagy gyenge kontrasztú fotókon. "
               "⚠ Jelentősen lassabb és erőforrásigényesebb a gyors módnál."},
    "scanModes.preciseRescan.startButton":
        {"en": "Start Re-detection (Accurate)",
         "hu": "Újrakeresés indítása (pontos)"},
    "scanModes.preciseRescan.warning":
        {"en": "Slow — may take several minutes on large libraries",
         "hu": "Lassú — nagy képkönyvtárnál több percig tarthat"},

    # reset automatically created Unknown identities
    "scanModes.resetUnknowns.title":
        {"en": "Rebuild Unknown Identities",
         "hu": "Unknown személyek újraépítése"},
    "scanModes.resetUnknowns.description":
        {"en": "Deletes every automatically created 'Unknown N' person and makes their faces "
               "unassigned again. Face boxes and embeddings are preserved. The recognition "
               "pipeline then runs again, using your named people as training examples, and "
               "rebuilds the remaining Unknown clusters from scratch.",
         "hu": "Törli az összes automatikusan létrehozott „Unknown N” személyt, és az arcaikat "
               "újra hozzárendeletlen állapotba teszi. Az arckeretek és embeddingek megmaradnak. "
               "Ezután újra lefut a felismerési folyamat az elnevezett személyek tanítási "
               "mintáival, majd a fennmaradó Unknown klaszterek tisztán újraépülnek."},
    "scanModes.resetUnknowns.startButton":
        {"en": "Rebuild Unknown Identities",
         "hu": "Unknown személyek újraépítése"},
    "scanModes.resetUnknowns.warning":
        {"en": "Auto-created Unknown groups will be deleted and rebuilt",
         "hu": "Az automatikus Unknown csoportok törlődnek és újraépülnek"},
    "reset_unknowns_title":
        {"en": "Rebuild Unknown Identities",
         "hu": "Unknown személyek újraépítése"},
    "reset_unknowns_msg":
        {"en": "Delete every automatically created 'Unknown N' person and run recognition again?\n\n"
               "Named people, face boxes and embeddings will be preserved.",
         "hu": "Törli az összes automatikusan létrehozott „Unknown N” személyt, majd újra "
               "futtatja a felismerést?\n\n"
               "Az elnevezett személyek, arckeretek és embeddingek megmaradnak."},
    "reset_unknowns_status":
        {"en": "Rebuilding Unknown identities: {persons} person(s) deleted, {faces} face(s) reset",
         "hu": "Unknown személyek újraépítése: {persons} személy törölve, {faces} arc alaphelyzetbe állítva"},

    # overlapping question-mark cleanup
    "scanModes.overlapCleanup.title":
        {"en": "Find Overlapping ? Boxes",
         "hu": "Átfedő ? keretek keresése"},
    "scanModes.overlapCleanup.description":
        {"en": "Searches the current database for unassigned question-mark face boxes "
               "that significantly overlap already named faces. It only lists suspicious "
               "candidates first; nothing is deleted until you review the list, keep the "
               "checkboxes you want, and confirm deletion.",
         "hu": "Megkeresi az adatbázisban azokat a kérdőjeles, személyhez nem rendelt "
               "arckereteket, amelyek jelentősen átfednek egy már elnevezett arccal. "
               "Először csak listázza a gyanús találatokat; semmit nem töröl addig, "
               "amíg át nem nézed a listát, ki nem választod a törlendőket, és meg "
               "nem erősíted a törlést."},
    "scanModes.overlapCleanup.startButton":
        {"en": "Find Overlapping ? Boxes",
         "hu": "Átfedő ? keretek keresése"},
    "scanModes.overlapCleanup.warning":
        {"en": "Review step included — known named faces are never deleted",
         "hu": "Átnézési lépéssel — az ismert, elnevezett arcokat soha nem törli"},

    # ── Identity repair scan ──────────────────────────────────────────────
    "scanModes.identityRepair.title":
        {"en": "Identity Repair Scan",
         "hu": "Identitás-helyreállító vizsgálat"},
    "scanModes.identityRepair.description":
        {"en": "Scans the whole database for 'Unknown N' identities that are "
               "really the same person split apart over many runs (e.g. "
               "Unknown 98 ≈ Unknown 155). It only proposes merges first; "
               "nothing is consolidated until you review the list, keep the "
               "pairs you want, and confirm.",
         "hu": "Átvizsgálja az egész adatbázist olyan „Unknown N” identitások "
               "után, amelyek valójában ugyanaz a személy, csak több futás "
               "során szétestek (pl. Unknown 98 ≈ Unknown 155). Először csak "
               "összevonásokat javasol; semmit nem von össze, amíg át nem nézed "
               "a listát, ki nem választod a párokat, és meg nem erősíted."},
    "scanModes.identityRepair.startButton":
        {"en": "Scan for Fragmented Identities",
         "hu": "Széttöredezett identitások keresése"},
    "scanModes.identityRepair.warning":
        {"en": "Review step included — only auto-named Unknown persons are merged",
         "hu": "Átnézési lépéssel — csak automatikus „Unknown” személyeket von össze"},

    # ── Clean up empty Unknown persons ────────────────────────────────────
    "scanModes.cleanupEmptyUnknowns.title":
        {"en": "Clean Up Empty Unknown Persons",
         "hu": "Üres Unknown személyek tisztítása"},
    "scanModes.cleanupEmptyUnknowns.description":
        {"en": "Removes automatically created 'Unknown N' persons that no longer have "
               "any faces attached. These empty placeholders are left behind when their "
               "last face is reassigned, removed, or excluded, and show up as blank '?' "
               "entries in the person list. Named people and the protected 'Ismeretlen' "
               "person are always kept, and no person that still owns a face is touched. "
               "Face boxes and embeddings are not affected.",
         "hu": "Eltávolítja azokat az automatikusan létrehozott „Unknown N” személyeket, "
               "amelyekhez már egyetlen arc sem tartozik. Ezek az üres helykitöltők akkor "
               "maradnak vissza, amikor az utolsó arcukat átrendelik, eltávolítják vagy "
               "kizárják, és üres „?” bejegyzésként jelennek meg a személylistában. Az "
               "elnevezett személyek és a védett „Ismeretlen” személy mindig megmaradnak, "
               "és egyetlen arccal rendelkező személyt sem érint. Az arckereteket és "
               "embeddingeket nem befolyásolja."},
    "scanModes.cleanupEmptyUnknowns.startButton":
        {"en": "Clean Up Empty Unknown Persons",
         "hu": "Üres Unknown személyek tisztítása"},
    "scanModes.cleanupEmptyUnknowns.warning":
        {"en": "Safe — only face-less auto-named Unknown persons are removed",
         "hu": "Biztonságos — csak az arc nélküli automatikus „Unknown” személyeket törli"},
    "cleanup_empty_unknowns_title":
        {"en": "Clean Up Empty Unknown Persons",
         "hu": "Üres Unknown személyek tisztítása"},
    "cleanup_empty_unknowns_msg":
        {"en": "Delete every automatically created 'Unknown' person that has no faces?\n\n"
               "Named people and the protected 'Ismeretlen' person are preserved.",
         "hu": "Törli az összes olyan automatikusan létrehozott „Unknown” személyt, "
               "amelyhez nem tartozik arc?\n\n"
               "Az elnevezett személyek és a védett „Ismeretlen” személy megmaradnak."},
    "cleanup_empty_unknowns_status":
        {"en": "Cleaned up {persons} empty Unknown person(s)",
         "hu": "{persons} üres Unknown személy törölve"},
    "cleanup_empty_unknowns_none":
        {"en": "No empty Unknown persons were found.",
         "hu": "Nem található üres Unknown személy."},

    # Identity repair results dialog
    "repair_title":      {"en": "Identity Repair — Merge Fragmented Unknowns",
                          "hu": "Identitás-helyreállítás — Töredékek összevonása"},
    "repair_intro":      {"en": "Pairs of 'Unknown' identities that appear to be the "
                                "same person. Keep the pairs you want to merge, then "
                                "confirm. Transitively linked pairs are consolidated "
                                "into one identity.",
                          "hu": "Olyan „Unknown” identitáspárok, amelyek azonos "
                                "személynek tűnnek. Hagyd bejelölve az összevonandó "
                                "párokat, majd erősítsd meg. A láncban összekapcsolt "
                                "párok egyetlen identitássá olvadnak."},
    "repair_count":      {"en": "{n} merge candidate(s)", "hu": "{n} összevonási javaslat"},
    "repair_no_matches_title": {"en": "No fragments found", "hu": "Nincs töredék"},
    "repair_no_matches_msg":   {"en": "No fragmented Unknown identities were found.",
                                "hu": "Nem található széttöredezett „Unknown” identitás."},
    "repair_pair_label": {"en": "{a} ≈ {b}  ({pct}% match, {fa}+{fb} faces)",
                          "hu": "{a} ≈ {b}  ({pct}% egyezés, {fa}+{fb} arc)"},
    "repair_merge_btn":  {"en": "Merge Selected", "hu": "Kijelöltek összevonása"},
    "repair_confirm_title": {"en": "Confirm merge", "hu": "Összevonás megerősítése"},
    "repair_confirm_msg":   {"en": "Merge {n} candidate pair(s)? This consolidates the "
                                   "linked Unknown identities and cannot be auto-undone.",
                             "hu": "Összevonod a(z) {n} javasolt párt? Ez egyesíti a "
                                   "kapcsolódó „Unknown” identitásokat, és nem vonható "
                                   "vissza automatikusan."},
    "repair_done_status":   {"en": "Identity repair: {groups} group(s) consolidated, "
                                   "{merged} fragment(s) merged away",
                             "hu": "Identitás-helyreállítás: {groups} csoport egyesítve, "
                                   "{merged} töredék összevonva"},
    "repair_error":      {"en": "Identity repair failed: {error}",
                          "hu": "Az identitás-helyreállítás sikertelen: {error}"},

    # ── Face diagnostics ("why this identity?") ───────────────────────────
    "diag_menu":         {"en": "Why this identity?", "hu": "Miért ez az identitás?"},
    "diag_title":        {"en": "Face Diagnostics — face #{id}",
                          "hu": "Arc diagnosztika — arc #{id}"},
    "diag_current":      {"en": "Current identity", "hu": "Jelenlegi identitás"},
    "diag_source":       {"en": "Assigned via", "hu": "Hozzárendelés módja"},
    "diag_threshold":    {"en": "Adaptive threshold", "hu": "Adaptív küszöb"},
    "diag_quality":      {"en": "Quality", "hu": "Minőség"},
    "diag_named_header": {"en": "Top named-person matches", "hu": "Legjobb nevesített egyezések"},
    "diag_unknown_header": {"en": "Top Unknown matches", "hu": "Legjobb „Unknown” egyezések"},
    "diag_verdict":      {"en": "Verdict", "hu": "Magyarázat"},
    "diag_none":         {"en": "(none)", "hu": "(nincs)"},
    "diag_error":        {"en": "Diagnostics failed: {error}",
                          "hu": "A diagnosztika sikertelen: {error}"},

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
    "suggestions_full_image":       {"en": "Full Image", "hu": "Teljes kép"},
    "suggestions_all_images":       {"en": "All Images", "hu": "Összes kép"},
    "suggestions_compare":          {"en": "Compare", "hu": "Összehasonlítás"},
    "suggestions_gallery_title":    {"en": "{name} — All Faces", "hu": "{name} — Összes arc"},
    "suggestions_compare_title":    {"en": "Compare: {cand} vs {target}",
                                     "hu": "Összehasonlítás: {cand} vs {target}"},
    "suggestions_full_img_title":   {"en": "Full Image", "hu": "Teljes kép"},
    "suggestions_candidate":        {"en": "Unknown", "hu": "Ismeretlen"},
    "suggestions_target":           {"en": "Named person", "hu": "Ismert személy"},
    "suggestions_no_faces":         {"en": "No faces found.", "hu": "Nincs arc."},
    "gallery_exclude_from_merge":   {"en": "Exclude from merge",
                                     "hu": "Kihagyás az összevonásból"},
    "gallery_include_in_merge":     {"en": "Undo exclusion",
                                     "hu": "Kizárás visszavonása"},
    "gallery_excluded_tooltip":     {"en": "Excluded from merge — will stay separate",
                                     "hu": "Kizárva az összevonásból — külön marad"},
    "gallery_merge_exclusion_hint": {"en": "Right-click a face to exclude it from the merge",
                                     "hu": "Jobb klikk egy arcra a kizáráshoz"},
    "gallery_excluded_count":       {"en": "{n} excluded from merge",
                                     "hu": "{n} kizárva az összevonásból"},
    "merge_all_excluded":           {"en": "All faces are excluded from the merge — "
                                           "nothing to merge.",
                                     "hu": "Minden arc ki van zárva az összevonásból — "
                                           "nincs mit összevonni."},
    "suggestions_click_full_image": {"en": "Click to open full image",
                                     "hu": "Kattints a teljes kép megnyitásához"},
    "suggestions_keyboard_hint":    {"en": "↑↓ navigate · Enter approve · Del reject · L later · Space compare",
                                     "hu": "↑↓ navigálás · Enter jóváhagyás · Del elutasítás · L később · Szóköz összehasonlítás"},

    # ── Background merge-matching ──────────────────────────────────────────
    "suggestions_defer":   {"en": "Later", "hu": "Később"},
    "suggestions_confidence": {"en": "{pct}% confidence", "hu": "{pct}% bizonyosság"},
    "suggestions_show_deferred": {"en": "Show postponed",
                                  "hu": "Halasztottak mutatása"},
    "suggestions_auto_merge": {"en": "Auto-Merge", "hu": "Automatikus összevonás"},
    "suggestions_auto_merge_max_faces": {"en": "Max unknown faces:",
                                         "hu": "Max. ismeretlen arcok:"},
    "suggestions_auto_merge_min_confidence": {"en": "Min confidence:",
                                              "hu": "Min. bizonyosság:"},
    "suggestions_auto_merge_result": {"en": "Automatically merged {n} suggestion(s).",
                                      "hu": "{n} ajánlat automatikusan összevonva."},
    "suggestions_auto_merge_none": {"en": "No suggestions matched the auto-merge criteria.",
                                    "hu": "Nincs olyan ajánlat, amely az automatikus összevonás feltételeit teljesítené."},
    "match_chip_idle":    {"en": "Matching: idle", "hu": "Egyeztetés: tétlen"},
    "match_chip_running": {"en": "Matching {label}: {processed}/{total} ({pct}%) · {found} found",
                          "hu": "Egyeztetés {label}: {processed}/{total} ({pct}%) · {found} találat"},
    "match_chip_paused":  {"en": "Matching paused", "hu": "Egyeztetés szüneteltetve"},
    "match_menu_pause":   {"en": "Pause matching", "hu": "Egyeztetés szüneteltetése"},
    "match_menu_resume":  {"en": "Resume matching", "hu": "Egyeztetés folytatása"},
    "match_menu_cancel":  {"en": "Cancel matching", "hu": "Egyeztetés megszakítása"},
    "match_menu_review":  {"en": "Review suggestions …", "hu": "Ajánlatok áttekintése …"},
    "match_chip_done":    {"en": "Matching: {n} suggestion(s)",
                          "hu": "Egyeztetés: {n} ajánlat"},
    "match_chip_failed":  {"en": "Matching failed", "hu": "Egyeztetés hibás"},
    "match_chip_cancelled": {"en": "Matching cancelled", "hu": "Egyeztetés megszakítva"},
    "match_chip_tip":     {"en": "Background name-matching status — click to review suggestions",
                          "hu": "Háttér névegyeztetés állapota — kattints az ajánlatokhoz"},

    # ── Google Drive cache (settings read-only display) ────────────────────
    "gdrive_cache_group":      {"en": "Google Drive Cache",
                                "hu": "Google Drive cache"},
    "gdrive_cache_dir_label":  {"en": "Cache directory (read-only):",
                                "hu": "Cache mappa (csak megtekintés):"},
    "gdrive_cache_size_label": {"en": "Current size:",
                                "hu": "Jelenlegi méret:"},
    "gdrive_cache_size_value": {"en": "{mb:.1f} MB",
                                "hu": "{mb:.1f} MB"},
    "gdrive_cache_note":       {"en": "Temporary files downloaded from Google Drive. "
                                      "Cleaned up on every start and exit. "
                                      "Max 500 MB; older files are evicted automatically.",
                                "hu": "A Google Drive-ról letöltött ideiglenes fájlok. "
                                      "Minden indításkor és kilépéskor törlődnek. "
                                      "Maximum 500 MB; a régebbi fájlok automatikusan törlődnek."},
    "gdrive_offline_error":    {"en": "Google Drive mode requires an internet connection.",
                                "hu": "Google Drive módhoz internetkapcsolat szükséges."},

    # ── Google Drive mode (Settings tab + workflow) ────────────────────────
    "settings_tab_gdrive":    {"en": "Google Drive", "hu": "Google Drive"},
    "settings_tab_recording": {"en": "Recording", "hu": "Rögzítés"},
    "rec_set_output_dir":     {"en": "Output folder:", "hu": "Mentési mappa:"},
    "rec_set_browse":         {"en": "Browse…", "hu": "Tallózás…"},
    "rec_set_quality":        {"en": "Quality:", "hu": "Minőség:"},
    "rec_set_fps":            {"en": "Frame rate (FPS):", "hu": "Képkocka/mp (FPS):"},
    "rec_set_segment":        {"en": "Segment length (s):",
                               "hu": "Szegmens hossza (mp):"},
    "rec_set_cursor":         {"en": "Capture mouse cursor",
                               "hu": "Egérkurzor rögzítése"},
    "rec_set_microphone":     {"en": "Capture microphone",
                               "hu": "Mikrofon rögzítése"},
    "rec_set_system_audio":   {"en": "Capture system audio (if available)",
                               "hu": "Rendszerhang rögzítése (ha elérhető)"},
    "rec_set_concat":         {"en": "Merge segments into one file when stopping",
                               "hu": "Szegmensek összefűzése egy fájlba leállításkor"},
    "rec_set_ffmpeg_path":    {"en": "ffmpeg path (optional):",
                               "hu": "ffmpeg útvonal (opcionális):"},
    "rec_set_unset":          {"en": "(ask each time)", "hu": "(kérdezzen rá mindig)"},
    "rec_set_display_group":  {"en": "Screens to record", "hu": "Képernyők rögzítése"},
    "rec_set_mode_active":    {"en": "Active application window",
                               "hu": "Aktív alkalmazásablak"},
    "rec_set_mode_all":       {"en": "All screens", "hu": "Összes képernyő"},
    "rec_set_mode_selected":  {"en": "Selected screens", "hu": "Kiválasztott képernyők"},
    "rec_set_primary_marker": {"en": "(primary)", "hu": "(elsődleges)"},
    "rec_set_monitor_word":   {"en": "monitor", "hu": "monitor"},
    "rec_set_no_displays":    {"en": "(no monitors detected)",
                               "hu": "(nem észlelhető monitor)"},
    "rec_set_auto_fps":       {"en": "Reduce frame rate for multi-monitor capture",
                               "hu": "Képkockaszám csökkentése többmonitoros rögzítésnél"},

    "gdrive_account_group":   {"en": "Google account", "hu": "Google fiók"},
    "gdrive_account_none":    {"en": "(no account signed in)",
                               "hu": "(nincs bejelentkezett fiók)"},
    "gdrive_account_signed_in": {"en": "Signed in as: {email}",
                                 "hu": "Bejelentkezve: {email}"},
    "gdrive_add_account_btn": {"en": "Add Google account …",
                               "hu": "Google fiók hozzáadása …"},
    "gdrive_signout_btn":     {"en": "Sign out", "hu": "Kijelentkezés"},
    "gdrive_signin_in_progress": {"en": "Opening browser for Google sign-in …",
                                  "hu": "Böngésző megnyitása a Google bejelentkezéshez …"},
    "gdrive_signin_ok":       {"en": "Signed in as {email}.",
                               "hu": "Sikeres bejelentkezés: {email}."},
    "gdrive_signin_failed":   {"en": "Google sign-in failed: {error}",
                               "hu": "Google bejelentkezés sikertelen: {error}"},
    "gdrive_signin_cancelled": {"en": "Sign-in cancelled by user.",
                                "hu": "A bejelentkezést a felhasználó megszakította."},
    "gdrive_signout_confirm_title": {"en": "Sign out", "hu": "Kijelentkezés"},
    "gdrive_signout_confirm_msg":   {
        "en": "Remove the stored credentials for {email}?\n"
              "You will need to sign in again to use Google Drive mode.",
        "hu": "Eltávolítod {email} mentett bejelentkezését?\n"
              "Az újbóli használathoz ismét be kell jelentkezned.",
    },

    "gdrive_oauth_not_configured_title": {
        "en": "Google OAuth not configured",
        "hu": "Google OAuth nincs beállítva",
    },
    "gdrive_oauth_not_configured_msg": {
        "en": "The Google OAuth client_id has not been set in this build. "
              "Replace the placeholder in app/gdrive/oauth_config.py before "
              "using Google Drive mode.",
        "hu": "A Google OAuth client_id nincs beállítva ebben a build-ben. "
              "A Google Drive mód használata előtt cseréld le a "
              "placeholder értékeket az app/gdrive/oauth_config.py-ban.",
    },

    "gdrive_folder_group":    {"en": "Project folder on Drive",
                               "hu": "Projektmappa a Drive-on"},
    "gdrive_folder_none":     {"en": "(no project folder selected)",
                               "hu": "(nincs projektmappa kiválasztva)"},
    "gdrive_pick_folder_btn": {"en": "Choose Drive folder …",
                               "hu": "Drive mappa kiválasztása …"},
    "gdrive_clear_folder_btn": {"en": "Clear", "hu": "Törlés"},
    "gdrive_folder_id_hint":  {"en": "Drive folder ID:",
                               "hu": "Drive mappa azonosító:"},
    "gdrive_pick_folder_title": {"en": "Choose Google Drive Folder",
                                 "hu": "Google Drive mappa kiválasztása"},
    "gdrive_pick_folder_intro": {
        "en": "Paste a Google Drive folder URL or ID. You can copy a URL "
              "from the Drive browser address bar.",
        "hu": "Illessz be egy Google Drive mappa URL-t vagy azonosítót. "
              "Az URL-t a Drive böngésző címsorából másolhatod ki.",
    },
    "gdrive_pick_folder_placeholder": {
        "en": "https://drive.google.com/drive/folders/…  or  folder ID",
        "hu": "https://drive.google.com/drive/folders/…  vagy  mappa ID",
    },
    "gdrive_folder_invalid":  {"en": "That does not look like a Drive folder.",
                               "hu": "Ez nem egy Drive mappának tűnik."},
    "gdrive_folder_unreachable": {
        "en": "Could not reach folder: {error}",
        "hu": "A mappa nem érhető el: {error}",
    },
    "gdrive_folder_selected": {"en": "Folder selected: {name}",
                               "hu": "Mappa kiválasztva: {name}"},

    "gdrive_mode_group":      {"en": "Drive mode — ON / OFF", "hu": "Drive mód — BE / KI"},
    "gdrive_mode_toggle":     {
        "en": "ON — Use Google Drive (only Drive images visible and processed)",
        "hu": "BE — Google Drive használata (csak Drive képek láthatók és feldolgozottak)",
    },
    "gdrive_mode_tip":        {
        "en": "ON: The app connects to Google Drive on startup, downloads the "
              "project database and processes only Drive images. Local folder "
              "is ignored.\n"
              "OFF: Only local images from the selected folder are used. "
              "Drive is not touched.\n"
              "Requires account + project folder to be configured above.",
        "hu": "BE: Az alkalmazás induláskor csatlakozik a Google Drive-hoz, "
              "letölti a projekt adatbázisát, és csak Drive-os képekkel dolgozik. "
              "A helyi mappa figyelmen kívül marad.\n"
              "KI: Csak a kiválasztott helyi mappában lévő képek kerülnek "
              "feldolgozásra. A Drive nincs érintve.\n"
              "Szükséges: fiók + projektmappa megadása fent.",
    },
    "gdrive_db_sync_toggle":  {"en": "Sync database with Drive automatically",
                               "hu": "Adatbázis automatikus szinkronizálása "
                                     "a Drive-val"},

    "gdrive_status_group":    {"en": "Status", "hu": "Állapot"},
    "gdrive_status_offline":  {"en": "Status: offline — sign-in disabled",
                               "hu": "Állapot: offline — bejelentkezés nem elérhető"},
    "gdrive_status_online":   {"en": "Status: online",
                               "hu": "Állapot: online"},
    "gdrive_status_incomplete": {"en": "Pick an account and a project folder to enable Drive mode.",
                                 "hu": "Válassz fiókot és projektmappát a Drive mód aktiválásához."},
    "gdrive_last_sync":       {"en": "Last sync: {when}",
                               "hu": "Utolsó szinkron: {when}"},
    "gdrive_last_sync_never": {"en": "Last sync: never",
                               "hu": "Utolsó szinkron: még nem volt"},
    "gdrive_sync_now_btn":    {"en": "Sync now", "hu": "Szinkronizálás most"},
    "gdrive_cache_clear_btn": {"en": "Clear cache now",
                               "hu": "Cache törlése most"},
    "gdrive_cache_cleared":   {"en": "Cache cleared ({n} file(s) removed).",
                               "hu": "Cache törölve ({n} fájl)."},

    "gdrive_error_title":     {"en": "Google Drive Error",
                               "hu": "Google Drive hiba"},
    "gdrive_info_title":      {"en": "Google Drive",
                               "hu": "Google Drive"},

    # ── Main-window Drive toolbar / status chip ───────────────────────────────
    "gdrive_open_project_btn":   {"en": "Drive Project …",
                                  "hu": "Drive projekt …"},
    "gdrive_open_project_tip":   {
        "en": "Open a Google Drive folder as the active project.\n"
              "The database is downloaded from Drive; changes sync back on save.",
        "hu": "Nyiss meg egy Google Drive mappát aktív projektként.\n"
              "Az adatbázis letöltődik, a változások szinkronizálódnak.",
    },
    "gdrive_close_project_btn":  {"en": "Close Drive",
                                  "hu": "Drive bezárása"},
    "gdrive_close_project_tip":  {
        "en": "Upload pending changes and release the project lock.",
        "hu": "Feltölti a változásokat és feloldja a projektzárat.",
    },
    "gdrive_chip_idle":          {"en": "Drive: —",     "hu": "Drive: —"},
    "gdrive_chip_opening":       {"en": "Drive: connecting …",
                                  "hu": "Drive: csatlakozás …"},
    "gdrive_chip_open":          {"en": "Drive: {name}", "hu": "Drive: {name}"},
    "gdrive_chip_syncing":       {"en": "Drive: syncing …",
                                  "hu": "Drive: szinkronizálás …"},
    "gdrive_chip_error":         {"en": "Drive: error",  "hu": "Drive: hiba"},
    "gdrive_chip_closing":       {"en": "Drive: closing …",
                                  "hu": "Drive: bezárás …"},
    "gdrive_open_failed":        {"en": "Could not open Drive project:\n{error}",
                                  "hu": "Nem sikerült megnyitni a Drive projektet:\n{error}"},
    "gdrive_project_opened":     {"en": "Google Drive project opened: {name}",
                                  "hu": "Google Drive projekt megnyitva: {name}"},
    "gdrive_confirm_close_title": {"en": "Close Drive Project",
                                   "hu": "Drive projekt bezárása"},
    "gdrive_confirm_close_msg":  {
        "en": "Upload pending changes and close the Drive project?\n"
              "The local database cache will be deleted.",
        "hu": "Feltölti a változásokat és bezárja a Drive projektet?\n"
              "A helyi adatbázis gyorsítótár törlődik.",
    },
    "gdrive_closing_wait":       {"en": "Closing Google Drive session …",
                                  "hu": "Google Drive munkamenet bezárása …"},
    "gdrive_not_configured_title": {"en": "Drive Not Configured",
                                    "hu": "Drive nincs konfigurálva"},
    "gdrive_not_configured_msg": {
        "en": "Open Settings → Google Drive and sign in, then pick a project folder.",
        "hu": "Nyisd meg a Beállítások → Google Drive fület, jelentkezz be, "
              "majd válassz project mappát.",
    },
    "ibp_drive_downloading": {
        "en": "⬇ Downloading from Google Drive …",
        "hu": "⬇ Letöltés a Google Drive-ról …",
    },
    "ibp_drive_download_failed": {
        "en": "⚠ Drive download failed: {error}",
        "hu": "⚠ Drive letöltés sikertelen: {error}",
    },
    "gdrive_mode_enabled_opening": {
        "en": "Google Drive mode is ON — connecting to project …",
        "hu": "Google Drive mód BE van kapcsolva — csatlakozás a projekthez …",
    },
    "gdrive_scan_no_session": {
        "en": "Google Drive mode is ON but the project is not open yet.\n"
              "Wait for the Drive connection to finish, then scan again.",
        "hu": "A Google Drive mód be van kapcsolva, de a projekt még nincs megnyitva.\n"
              "Várj, amíg a Drive kapcsolat felépül, majd próbáld újra a beolvasást.",
    },

    # ── Universal search bar ─────────────────────────────────────────────────
    "search_universal_placeholder": {
        "en": "Search persons, dates, places, files…",
        "hu": "Keresés: személyek, dátumok, helyek, fájlok…",
    },
    "search_token_person":      {"en": "Person",    "hu": "Személy"},
    "search_token_nickname":    {"en": "Nickname",  "hu": "Becenév"},
    "search_token_family_code": {"en": "Family ID", "hu": "Csal. azon."},
    "search_token_place":       {"en": "Place",     "hu": "Hely"},
    "search_token_date":        {"en": "Date",      "hu": "Dátum"},
    "search_token_image":       {"en": "Image",     "hu": "Képadat"},
    "search_token_object":      {"en": "Object",    "hu": "Objektum"},
    "search_only_person_cb": {
        "en": "Only this person",
        "hu": "Csak az adott személy",
    },
    "search_names_detail_label": {"en": "Names", "hu": "Nevek"},
    "search_names_detail_placeholder": {
        "en": "Names separated by commas, e.g. Benedek, Matyi",
        "hu": "Nevek vesszővel elválasztva, pl. Benedek, Matyi",
    },
    "ibp_universal_placeholder": {
        "en": "Filter by filename, folder, place, date, person…",
        "hu": "Szűrés fájlnév, mappa, hely, dátum, személy szerint…",
    },
    "ibp_search_results_hdr": {
        "en": "Search results ({n})",
        "hu": "Keresési találatok ({n})",
    },
    "ibp_search_no_results": {
        "en": "No matching images.",
        "hu": "Nincs megfelelő kép.",
    },

    # ── Keyboard shortcuts settings ──────────────────────────────────────
    "sc_enable_all":        {"en": "Enable keyboard shortcuts",
                             "hu": "Billentyűparancsok engedélyezése"},
    "sc_modify_btn":        {"en": "Modify",     "hu": "Módosít"},
    "sc_save_btn":          {"en": "Save",        "hu": "Mentés"},
    "sc_cancel_btn":        {"en": "Cancel",      "hu": "Mégse"},
    "sc_delete_btn":        {"en": "Delete",      "hu": "Törlés"},
    "sc_not_set":           {"en": "Not set",     "hu": "Nincs beállítva"},
    "sc_press_key":         {"en": "⏺ Press a key…",
                             "hu": "⏺ Nyomj le egy billentyűt…"},
    "sc_conflict_msg":      {"en": "Already in use on this page: {func}",
                             "hu": "Ezen az oldalon már használatban: {func}"},
    # categories / pages
    "sc_cat_general":       {"en": "General",       "hu": "Általános"},
    "sc_cat_image":         {"en": "Image Browser", "hu": "Képböngésző"},
    "sc_cat_faces":         {"en": "Faces",          "hu": "Arcok"},
    "sc_cat_collage":       {"en": "Collage",        "hu": "Kollázs"},

    # function names
    "sc_fn_settings":       {"en": "Open Settings",          "hu": "Beállítások megnyitása"},
    "sc_fn_search_focus":   {"en": "Focus Search",           "hu": "Keresés fókusz"},
    "sc_fn_fullscreen":     {"en": "Fullscreen",             "hu": "Teljes képernyő"},
    "sc_fn_log_panel":      {"en": "Log Panel",              "hu": "Log panel"},
    "sc_fn_image_prev":     {"en": "Previous Image",         "hu": "Előző kép"},
    "sc_fn_image_next":     {"en": "Next Image",             "hu": "Következő kép"},
    "sc_fn_manual_sel":     {"en": "Manual Selection Mode",  "hu": "Kézi kijelölés mód"},
    "sc_fn_face_assign":      {"en": "Assign to Person",       "hu": "Személyhez adás"},
    "sc_fn_face_cycle_next":  {"en": "Next Face on Image",     "hu": "Következő arc a képen"},
    "sc_fn_face_confirm":     {"en": "Confirm Assignment",     "hu": "Hozzárendelés véglegesítése"},
    "sc_fn_deselect":       {"en": "Deselect / Cancel (Esc)","hu": "Kijelölés megszüntetése (Esc)"},
    "sc_fn_bbox_delete":    {"en": "Delete BBox",            "hu": "BBox törlés"},
    "sc_fn_bbox_edit":      {"en": "Edit BBox",              "hu": "BBox módosítás"},
    "sc_fn_bbox_next":      {"en": "Next BBox",              "hu": "Következő bbox"},
    "sc_fn_bbox_prev":      {"en": "Previous BBox",          "hu": "Előző bbox"},
    "sc_fn_zoom_in":        {"en": "Zoom In",                "hu": "Zoom be"},
    "sc_fn_zoom_out":       {"en": "Zoom Out",               "hu": "Zoom ki"},
    "sc_fn_fit":            {"en": "Fit to Screen",          "hu": "Fit to screen"},
    "sc_fn_info":           {"en": "Info Panel",             "hu": "Információ panel"},
    "sc_fn_person_new":     {"en": "New Person",             "hu": "Új személy létrehozása"},
    "sc_fn_person_rename":  {"en": "Rename Person",          "hu": "Átnevezés"},
    "sc_fn_person_merge":   {"en": "Merge Persons",          "hu": "Merge"},
    "sc_fn_person_reassign":{"en": "Reassign Face",          "hu": "Újrahozzárendelés"},
    "sc_fn_person_exclude": {"en": "Exclude Face",           "hu": "Kizárás"},
    "sc_fn_collage_import": {"en": "Import Collage",         "hu": "Kollázs import"},
    "sc_fn_face_overlay":   {"en": "Face Overlay Toggle",    "hu": "Face overlay kapcsoló"},
    "sc_fn_node_delete":    {"en": "Delete Node",            "hu": "Node törlés"},
    "sc_fn_html_export":    {"en": "HTML Export",            "hu": "HTML export"},
    "sc_fn_bbox_undo":      {"en": "Undo BBox Edit",         "hu": "Arckeret szerkesztés visszavonása"},
    "sc_fn_bbox_redo":      {"en": "Redo BBox Edit",         "hu": "Arckeret szerkesztés visszaállítása"},

    # ── Object tagging ─────────────────────────────────────────────────────────
    "tab_objects":          {"en": "Objects",                "hu": "Objektumok"},
    "object_mode":          {"en": "📌 Object",              "hu": "📌 Objektum"},
    "object_mode_tip":      {"en": "Click a point on the image to tag an object",
                             "hu": "Kattints egy pontra a képen objektum megjelöléséhez"},
    "object_point_hint":    {"en": "Click a point on the image to place an object marker.",
                             "hu": "Kattints egy pontra a képen az objektum-jelölő elhelyezéséhez."},
    "object_rect_hint":     {"en": "Drag a rectangle around the object on the image.",
                             "hu": "Húzz egy téglalapot az objektum köré a képen."},
    "overlay_objects":      {"en": "Objects",                "hu": "Objektumok"},
    "overlay_objects_tip":  {"en": "Show/hide object boxes and adjust their opacity",
                             "hu": "Objektum-keretek megjelenítése/elrejtése és átlátszóságuk állítása"},

    # Picker dialog (click → choose/create object)
    "object_picker_title":  {"en": "Tag an Object",          "hu": "Objektum megjelölése"},
    "object_picker_search": {"en": "Find existing object …", "hu": "Meglévő objektum keresése …"},
    "object_picker_existing":{"en": "Existing objects",      "hu": "Meglévő objektumok"},
    "object_picker_new":    {"en": "Create new object",      "hu": "Új objektum létrehozása"},
    "object_picker_occ_note":{"en": "Note for this image (optional)",
                             "hu": "Megjegyzés ehhez a képhez (opcionális)"},
    "object_use_selected":  {"en": "Tag selected",           "hu": "Kiválasztott megjelölése"},
    "object_create_and_tag":{"en": "Create & tag",           "hu": "Létrehozás és megjelölés"},

    # Object fields
    "object_name":          {"en": "Name",                   "hu": "Név"},
    "object_description":   {"en": "Description",            "hu": "Leírás"},
    "object_notes":         {"en": "Notes",                  "hu": "Megjegyzések"},
    "object_example_name":  {"en": "e.g. BMW E91",           "hu": "pl. BMW E91"},
    "object_example_desc":  {"en": "e.g. 2004 BMW E91 330D Touring",
                             "hu": "pl. 2004-es BMW E91 330D Touring"},

    # Objects panel
    "objects_filter_name":  {"en": "Filter by name …",       "hu": "Szűrés név szerint …"},
    "objects_col_name":     {"en": "Name",                   "hu": "Név"},
    "objects_col_images":   {"en": "Images",                 "hu": "Képek"},
    "objects_col_notes":    {"en": "Notes",                  "hu": "Megjegyzések"},
    "objects_col_persons":  {"en": "Persons",                "hu": "Személyek"},
    "objects_new":          {"en": "New Object",             "hu": "Új objektum"},
    "objects_delete":       {"en": "Delete Object",          "hu": "Objektum törlése"},
    "objects_merge":        {"en": "Merge …",                "hu": "Összevonás …"},
    "objects_edit":         {"en": "Edit",                   "hu": "Szerkesztés"},
    "objects_empty":        {"en": "No object selected.",    "hu": "Nincs kiválasztott objektum."},
    "objects_none":         {"en": "No objects yet.",        "hu": "Még nincsenek objektumok."},
    "objects_delete_confirm":{"en": "Delete object '{name}' and all its occurrences?",
                             "hu": "Törlöd a(z) '{name}' objektumot és minden előfordulását?"},

    # Object detail / data sheet
    "object_detail_created":{"en": "Created",                "hu": "Létrehozva"},
    "object_detail_updated":{"en": "Last modified",          "hu": "Utolsó módosítás"},
    "object_detail_images": {"en": "Related images",         "hu": "Kapcsolódó képek"},
    "object_detail_notes":  {"en": "Related notes",          "hu": "Kapcsolódó megjegyzések"},
    "object_detail_persons":{"en": "Related persons",        "hu": "Kapcsolódó személyek"},
    "object_detail_gallery":{"en": "Gallery",                "hu": "Galéria"},
    "object_set_thumbnail": {"en": "Set as object thumbnail",
                             "hu": "Beállítás objektum bélyegképként"},
    "object_clear_thumbnail":{"en": "Clear object thumbnail",
                             "hu": "Objektum bélyegkép törlése"},
    "object_detail_comments":{"en": "All comments",          "hu": "Összes megjegyzés"},
    "object_no_comments":   {"en": "No per-image comments.", "hu": "Nincsenek képenkénti megjegyzések."},

    # Person ↔ object roles
    "object_role":          {"en": "Role",                   "hu": "Szerep"},
    "object_role_owner":        {"en": "Owner",              "hu": "Tulajdonos"},
    "object_role_former_owner": {"en": "Former owner",       "hu": "Korábbi tulajdonos"},
    "object_role_driver":       {"en": "Driver",             "hu": "Sofőr"},
    "object_role_creator":      {"en": "Creator",            "hu": "Készítő"},
    "object_role_user":         {"en": "User",               "hu": "Használó"},
    "object_role_family":       {"en": "Family member",      "hu": "Családtag"},
    "object_role_other":        {"en": "Other",              "hu": "Egyéb"},
    "object_add_person":    {"en": "Add person …",           "hu": "Személy hozzáadása …"},
    "object_remove_person": {"en": "Remove",                 "hu": "Eltávolítás"},

    # Person detail integration
    "person_related_objects":{"en": "Related objects",       "hu": "Kapcsolódó objektumok"},
    "person_no_objects":    {"en": "No related objects.",    "hu": "Nincsenek kapcsolódó objektumok."},

    # Merge dialog
    "object_merge_title":   {"en": "Merge Objects",          "hu": "Objektumok összevonása"},
    "object_merge_target":  {"en": "Keep this object (target)",
                             "hu": "Ezt az objektumot tartsd meg (cél)"},
    "object_merge_hint":    {"en": "All occurrences, notes and person links are preserved.",
                             "hu": "Minden előfordulás, megjegyzés és személykapcsolat megmarad."},
    "object_tagged_ok":     {"en": "Object tagged.",         "hu": "Objektum megjelölve."},
    "object_search":        {"en": "Search objects",         "hu": "Objektumok keresése"},

    # Context menu (right-click on the image / on an object marker)
    "object_ctx_mark_here": {"en": "📌 Tag object here",      "hu": "📌 Objektum megjelölése itt"},
    "object_ctx_open":      {"en": "Open object",            "hu": "Objektum megnyitása"},
    "object_ctx_edit":      {"en": "Edit object …",          "hu": "Objektum szerkesztése …"},
    "object_ctx_edit_frame":{"en": "Edit frame",            "hu": "Keret szerkesztése"},
    "object_ctx_edit_note": {"en": "Edit note …",           "hu": "Megjegyzés szerkesztése …"},
    "object_ctx_delete_occurrence": {"en": "Delete object marker here",
                             "hu": "Objektum-jelölő törlése"},
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
        target = _prefs_file()
        # One-time migration: move legacy ~/.face_local_prefs.json to new location.
        if not target.exists() and _LEGACY_PREFS_FILE.exists():
            try:
                import shutil
                shutil.copy2(_LEGACY_PREFS_FILE, target)
                log.info("i18n: migrated language prefs from %s to %s", _LEGACY_PREFS_FILE, target)
            except Exception as mig_exc:  # noqa: BLE001
                log.warning("i18n: could not migrate prefs: %s", mig_exc)
        if target.exists():
            data = json.loads(target.read_text(encoding="utf-8"))
            lang = data.get("language", "en")
            if lang in SUPPORTED:
                _lang = lang
    except Exception as exc:  # noqa: BLE001
        log.warning("i18n: could not load prefs: %s", exc)


def _save_prefs() -> None:
    try:
        target = _prefs_file()
        data: dict = {}
        if target.exists():
            data = json.loads(target.read_text(encoding="utf-8"))
        data["language"] = _lang
        target.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        log.warning("i18n: could not save prefs: %s", exc)
