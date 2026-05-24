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
    "recluster_all":      {"en": "Recognize Unassigned", "hu": "Névtelen arcok felismerése"},
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
    "main_toolbar":       {"en": "Main", "hu": "Fő eszköztár"},
    "tab_face_recognition": {"en": "Face Recognition", "hu": "Arcfelismerés"},
    "tab_image_browser":  {"en": "Image Browser", "hu": "Képböngésző"},
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
    "export_html_group": {"en": "Static HTML Gallery", "hu": "Statikus weboldal"},
    "export_html_desc":  {"en": "Browser gallery: persons, crops, and original photos with interactive face overlays. Searchable.",
                          "hu": "Böngészőben megnyitható galéria: személyek, arc-kivágások és eredeti képek interaktív arckeretekkel. Keresés is elérhető."},
    "export_generate_html": {"en": "Generate HTML Gallery …", "hu": "HTML galéria generálása…"},
    "csv_exported":      {"en": "CSV Exported", "hu": "CSV exportálva"},
    "json_exported":     {"en": "JSON Exported", "hu": "JSON exportálva"},
    "json_saved":        {"en": "Saved to:\n{path}", "hu": "Mentve:\n{path}"},
    "html_gallery_folder": {"en": "HTML Gallery Folder", "hu": "HTML galéria mappája"},
    "html_gallery_done": {"en": "HTML Gallery Ready", "hu": "HTML galéria kész"},
    "html_gallery_open": {"en": "Generated:\n{path}\n\nOpen it in the browser?",
                          "hu": "Generálva:\n{path}\n\nMegnyitod a böngészőben?"},
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
    "mark_face":         {"en": "Mark Face Manually", "hu": "Arc kézi jelölése"},
    "mark_face_hint":    {"en": "Drag on the image to mark a face region.",
                          "hu": "Húzd az egeret a képen az arc jelöléséhez."},
    "mark_face_saved":   {"en": "Face saved. Assign it to a person from the sidebar.",
                          "hu": "Arc mentve. Rendeld személyhez az oldalsávból."},
    "previous":          {"en": "Previous", "hu": "Előző"},
    "next":              {"en": "Next", "hu": "Következő"},
    "manual_marking":    {"en": "Manual Marking", "hu": "Kézi jelölés"},
    "manual_marking_tip": {"en": "Click, then drag on the image to manually mark a face",
                           "hu": "Kattints, majd húzd az egeret a képen egy arc kézi megjelöléséhez"},
    "fullscreen":        {"en": "Full Screen", "hu": "Teljes képernyő"},
    "fullscreen_tip":    {"en": "Full-screen view (F11)", "hu": "Teljes képernyős nézet (F11)"},
    "exit_fullscreen":   {"en": "Exit Full Screen", "hu": "Kilépés a teljes képernyőből"},
    "exit_fullscreen_tip": {"en": "Exit full-screen view (F11)", "hu": "Kilépés a teljes képernyős nézetből (F11)"},
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

    # ── Toolbar — short labels (no emoji) ────────────────────────────────
    "tb_collage_import":      {"en": "Collage+",    "hu": "Kollázs+"},
    "tb_collage_html_export": {"en": "Collage HTML","hu": "Kollázs HTML"},
    "tb_update_check":        {"en": "Update…",     "hu": "Frissítés…"},
    "tb_export":              {"en": "Export",       "hu": "Export"},

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
    "ibp_date_hdr":           {"en": "Photo date / period:",
                               "hu": "Kép dátuma / időszaka:"},
    "ibp_date_placeholder":   {"en": "e.g. 1954  or  1954.03.12  or  1930s",
                               "hu": "pl. 1954  vagy  1954.03.12  vagy  1930-as évek"},
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
    "ibp_ctx_edit_bbox":      {"en": "Edit bbox",        "hu": "Bbox módosítása"},
    "ibp_ctx_delete":         {"en": "Delete",           "hu": "Törlés"},
    "ibp_ctx_unknown_face":   {"en": "Unknown face",     "hu": "Ismeretlen arc"},
    "ibp_select_hint":        {"en": "Select a folder and run a scan",
                               "hu": "Válassz mappát és futtass beolvasást"},
    "ibp_manual_mark_tooltip": {"en": "Click, then drag on the image to manually mark a face",
                                "hu": "Kattints, majd húzd az egeret a képen egy arc kézi megjelöléséhez"},
    "ibp_fullscreen_tooltip": {"en": "Full-screen view (F11)",
                               "hu": "Teljes képernyős nézet (F11)"},
    "ibp_exit_fullscreen_tooltip": {"en": "Exit full-screen view (F11)",
                                    "hu": "Kilépés a teljes képernyős nézetből (F11)"},
    "ibp_date_tooltip":      {"en": "The date or period when the image was taken (free text)",
                              "hu": "A kép készítésének dátuma vagy időszaka (szabad szöveg)"},

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
