"""Capture the Scan & maintenance (Beolvasás és karbantartás) dialog."""
from __future__ import annotations
import logging, sys
from pathlib import Path
logging.basicConfig(level=logging.WARNING)

from app.ui.i18n import load_prefs
load_prefs()

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer, QEventLoop
from app.ui.theme import apply_theme
from app.ui.dialogs.scan_modes_dialog import ScanModesDialog

OUT = Path("docs/wiki/local_ai_face_recognizer.wiki/images")

app = QApplication(sys.argv)
app.setApplicationName("Face-Local")
app.setOrganizationName("face-local")
apply_theme(app)

noop = lambda: None
dlg = ScanModesDialog(
    on_incremental=noop, on_full_rescan=noop,
    on_face_rescan_fast=noop, on_face_rescan_accurate=noop,
    on_reset_unknown_persons=noop, on_find_overlapping_unknown_faces=noop,
    on_identity_repair_scan=noop,
)
dlg.resize(620, 860)
dlg.show()


def pump(ms):
    loop = QEventLoop(); QTimer.singleShot(ms, loop.quit); loop.exec()


def run():
    pump(1000)
    pix = dlg.grab()
    p = OUT / "scan-maintenance-dialog.png"
    pix.save(str(p))
    print(f"saved {p} ({pix.width()}x{pix.height()})")
    app.quit()


QTimer.singleShot(200, run)
app.exec()
