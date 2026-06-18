"""Shared pytest configuration for the Face-Local test suite."""

from __future__ import annotations

import os


def pytest_configure(config) -> None:  # noqa: ANN001
    """Prepare Qt before pytest-qt creates the QApplication."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication

        QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    except Exception:  # noqa: BLE001 — headless / no display
        pass
