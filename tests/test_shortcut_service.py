"""Tests for keyboard shortcut definitions and QShortcut wiring."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QWidget

from app.services.shortcut_service import ShortcutDef, ShortcutService, get_shortcut_service


class _EmptyQS:
    def value(self, key, default=None, type=None):
        return default

    def setValue(self, key, value):
        pass

    def sync(self):
        pass


@pytest.fixture()
def shortcut_host(qapp, monkeypatch):
    monkeypatch.setattr("app.app_settings.app_qsettings", lambda: _EmptyQS())
    host = QWidget()
    host.show()
    return host


def test_normalize_key_ignores_modifier_only(qapp):
    from PySide6.QtGui import QKeyEvent

    from app.services.shortcut_service import normalize_key

    event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key_Control, Qt.ControlModifier)
    assert normalize_key(event) == ""


def test_shortcut_service_all_shortcuts_and_get():
    svc = ShortcutService()
    all_defs = svc.all_shortcuts()
    assert len(all_defs) > 0
    first = all_defs[0]
    assert svc.get(first.id) is first
    assert svc.get("nonexistent.id") is None


def test_shortcut_def_defaults_and_context():
    sc = ShortcutDef("test.action", "sc_fn_test", "sc_cat_image", "Ctrl+T")
    assert sc.current_key == "Ctrl+T"
    assert sc.context == "image"
    assert sc.is_global is False


def test_shortcut_def_global_context():
    sc = ShortcutDef("general.test", "sc_fn_test", "sc_cat_general", "Ctrl+G")
    assert sc.context == "general"
    assert sc.is_global is True


def test_shortcut_service_register_and_bind_fires_handler(shortcut_host):
    svc = ShortcutService()
    fired: list[str] = []

    svc.get("bbox.delete").current_key = "Del"
    svc.set_host(shortcut_host)
    svc.register("bbox.delete", lambda: fired.append("deleted"))
    svc.set_active_context("image")

    shortcut = svc._qshortcuts.get("bbox.delete")
    assert shortcut is not None
    assert shortcut.key().toString(QKeySequence.PortableText) == svc.get("bbox.delete").current_key

    assert svc.dispatch("Del") is True
    assert fired == ["deleted"]


def test_shortcut_service_unregister_removes_qshortcut(shortcut_host):
    svc = ShortcutService()
    svc.get("bbox.edit").current_key = "E"
    svc.set_host(shortcut_host)
    svc.register("bbox.edit", lambda: None)
    assert "bbox.edit" in svc._qshortcuts

    svc.unregister("bbox.edit")
    assert "bbox.edit" not in svc._handlers
    assert "bbox.edit" not in svc._qshortcuts


def test_shortcut_service_set_key_updates_binding(shortcut_host):
    svc = ShortcutService()
    svc.set_host(shortcut_host)
    svc.register("image.info", lambda: None)

    svc.set_key("image.info", "Ctrl+I")
    shortcut = svc._qshortcuts["image.info"]
    assert shortcut.key().toString(QKeySequence.PortableText) == "Ctrl+I"


def test_shortcut_service_enabled_and_capturing_gate(shortcut_host):
    svc = ShortcutService()
    fired: list[str] = []
    svc.get("bbox.delete").current_key = "Del"
    svc.set_host(shortcut_host)
    svc.register("bbox.delete", lambda: fired.append("x"))
    svc.set_active_context("image")

    svc.set_enabled(False)
    assert svc.dispatch("Del") is False
    assert fired == []

    svc.set_enabled(True)
    svc.set_capturing(True)
    assert svc.dispatch("Del") is False

    svc.set_capturing(False)
    assert svc.dispatch("Del") is True
    assert fired == ["x"]


def test_shortcut_service_find_conflict_and_save(qapp, monkeypatch):
    svc = ShortcutService()
    svc.get("bbox.delete").current_key = "Del"
    svc.get("bbox.edit").current_key = "Del"

    conflict = svc.find_conflict("Del", exclude_id="bbox.delete")
    assert conflict is not None
    assert conflict.id == "bbox.edit"

    saved: dict[str, object] = {}

    class FakeQS:
        def value(self, key, default=None, type=None):
            return saved.get(key, default)

        def setValue(self, key, value):
            saved[key] = value

        def sync(self):
            saved["synced"] = True

    monkeypatch.setattr("app.app_settings.app_qsettings", lambda: FakeQS())
    svc.set_enabled(False)
    svc.save()

    assert saved["shortcuts/enabled"] is False
    assert saved["synced"] is True


def test_get_shortcut_service_returns_singleton():
    first = get_shortcut_service()
    second = get_shortcut_service()
    assert first is second


@pytest.mark.parametrize(
    ("widget_type", "expected"),
    [
        ("lineedit", True),
        ("plain", False),
    ],
)
def test_is_input_widget(widget_type, expected, qapp):
    from PySide6.QtWidgets import QLineEdit, QWidget

    svc = ShortcutService()
    widget = QLineEdit() if widget_type == "lineedit" else QWidget()
    assert svc.__class__.__module__  # keep linter happy about svc usage
    from app.services.shortcut_service import is_input_widget

    assert is_input_widget(widget) is expected
