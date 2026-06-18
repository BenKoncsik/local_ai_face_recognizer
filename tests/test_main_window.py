"""Smoke tests for the main application window."""

from __future__ import annotations

import pytest

from app.config import AppConfig
from app.db.database import init_db
from app.ui.main_window import MainWindow


@pytest.fixture()
def config(tmp_path):
    db_path = tmp_path / "main_window.db"
    init_db(db_path)
    cfg = AppConfig()
    cfg.storage.db_path = str(db_path)
    cfg.storage.crops_dir = str(tmp_path / "crops")
    cfg.crops_dir_resolved.mkdir(parents=True, exist_ok=True)
    return cfg


def _stub_heavy_startup(monkeypatch) -> None:
    """Prevent network / background workers from starting during smoke tests."""
    for name in (
        "_start_update_check",
        "_start_match_worker",
        "_start_crop_repair_task",
        "_check_image_library_on_startup",
        "_restore_last_folder",
        "_setup_gdrive_chip",
    ):
        monkeypatch.setattr(MainWindow, name, lambda self, _n=name: None)


def test_main_window_construct(config, qtbot, monkeypatch):
    _stub_heavy_startup(monkeypatch)
    window = MainWindow(config=config)
    qtbot.addWidget(window)
    assert window._config is config
    assert window._db_path == str(config.db_path_resolved)
    window.close()
