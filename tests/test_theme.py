"""Tests for application theme helpers."""

from __future__ import annotations

import pytest
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication

from app.ui.theme import ACCENT, BASE, TEXT, apply_palette, apply_theme


class TestTheme:
    def test_apply_palette_sets_colors(self, qapp):
        apply_palette(qapp)
        pal = qapp.palette()
        assert pal.color(QPalette.Window).name().upper() == BASE.upper()
        assert pal.color(QPalette.WindowText).name().upper() == TEXT.upper()
        assert pal.color(QPalette.Highlight).name().upper() == ACCENT.upper()

    def test_apply_theme_sets_stylesheet(self, qapp):
        apply_theme(qapp)
        sheet = qapp.styleSheet()
        assert "QWidget" in sheet
        assert BASE in sheet
        assert "font-family" in sheet
