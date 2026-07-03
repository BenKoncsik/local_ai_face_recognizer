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

    _disable_blocking_message_boxes()


def _disable_blocking_message_boxes() -> None:
    """Make QMessageBox convenience popups non-blocking during tests.

    The static ``QMessageBox.warning``/``information``/``critical``/``question``/
    ``about`` helpers spin a native modal event loop that waits for a user click.
    Under the offscreen platform there is no one to click, so any code path that
    reaches one of them freezes the whole test run indefinitely (this is what
    made the suite hang for hours). Replace them with immediate no-op returns so a
    stray popup fails the assertion around it instead of blocking forever. Tests
    that need specific behaviour still override these locally via monkeypatch.
    """
    try:
        from PySide6.QtWidgets import QMessageBox
    except Exception:  # noqa: BLE001 — headless / no PySide6
        return

    ok = QMessageBox.StandardButton.Ok
    for _name in ("warning", "information", "critical", "question", "about"):
        setattr(QMessageBox, _name, staticmethod(lambda *a, _r=ok, **k: _r))
