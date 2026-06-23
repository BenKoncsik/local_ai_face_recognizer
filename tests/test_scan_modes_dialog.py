"""Smoke test: the (AI-only) scan & maintenance dialog builds without error.

After the classic vector-recognition pipeline was removed, ScanModesDialog has a
single AI-centric layout (no Classic tab). This guards against missing i18n keys
or a broken card wiring after that restructure.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QPushButton

from app.ui.dialogs.scan_modes_dialog import ScanModesDialog


@pytest.fixture
def dialog(qtbot):
    calls: list[str] = []

    def cb(name: str):
        return lambda: calls.append(name)

    dlg = ScanModesDialog(
        on_folder_rescan=cb("folder_rescan"),
        on_reset_unknown_persons=cb("reset_unknowns"),
        on_find_overlapping_unknown_faces=cb("overlap"),
        on_find_embedding_duplicate_faces=cb("embedding_dup"),
        on_identity_repair_scan=cb("identity_repair"),
        on_cleanup_empty_unknown_persons=cb("cleanup_empty"),
        on_revalidate_face_geometry=cb("geometry"),
        on_manage_ignored_faces=cb("ignored"),
        on_deep_rescan=cb("deep_rescan"),
        on_deep_rebuild=cb("deep_rebuild"),
        on_deep_train=cb("deep_train"),
        on_deep_face_detect=cb("deep_face_detect"),
        on_deep_rebuild_model=cb("deep_rebuild_model"),
    )
    qtbot.addWidget(dlg)
    dlg._calls = calls  # type: ignore[attr-defined]
    return dlg


def test_dialog_builds_with_all_cards(dialog):
    # Every operation renders a button; the dialog plus the close button means a
    # healthy build has well more than a handful of buttons.
    buttons = dialog.findChildren(QPushButton)
    assert len(buttons) >= 14  # 13 action cards + close


def test_folder_rescan_button_invokes_callback(dialog, qtbot):
    # The folder re-read card is the AI-backed replacement for the old classic
    # "Scan & Index"; clicking it must fire its callback.
    dialog._launch_folder_rescan()
    assert "folder_rescan" in dialog._calls


def test_geometry_cleanup_button_invokes_callback(dialog):
    dialog._launch_revalidate_face_geometry()
    assert "geometry" in dialog._calls


def test_force_rebuild_model_button_invokes_callback(dialog):
    # The forced "rebuild AI model from scratch" option.
    dialog._launch_deep_rebuild_model()
    assert "deep_rebuild_model" in dialog._calls
