"""Application-wide QSS stylesheet — Catppuccin Mocha dark theme."""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

# ── Palette tokens ────────────────────────────────────────────────────────────
CRUST       = "#11111B"
MANTLE      = "#181825"
BASE        = "#1E1E2E"
SURFACE0    = "#313244"
SURFACE1    = "#45475A"
SURFACE2    = "#585B70"
OVERLAY0    = "#6C7086"
OVERLAY1    = "#7F849C"
TEXT        = "#CDD6F4"
SUBTEXT0    = "#A6ADC8"
SUBTEXT1    = "#BAC2DE"
LAVENDER    = "#B4BEFE"
BLUE        = "#89B4FA"
SAPPHIRE    = "#74C7EC"
SKY         = "#89DCEB"
TEAL        = "#94E2D5"
GREEN       = "#A6E3A1"
YELLOW      = "#F9E2AF"
PEACH       = "#FAB387"
RED         = "#F38BA8"
MAROON      = "#EBA0AC"
PINK        = "#F5C2E7"
MAUVE       = "#CBA6F7"
ROSEWATER   = "#F5E0DC"

ACCENT      = BLUE        # primary interactive colour
ACCENT_ALT  = LAVENDER    # secondary / focus ring


def apply_palette(app: QApplication) -> None:
    """Set QPalette so native widgets inherit the right base colours."""
    p = QPalette()
    c = QColor

    p.setColor(QPalette.Window,          c(BASE))
    p.setColor(QPalette.WindowText,      c(TEXT))
    p.setColor(QPalette.Base,            c(MANTLE))
    p.setColor(QPalette.AlternateBase,   c(CRUST))
    p.setColor(QPalette.Text,            c(TEXT))
    p.setColor(QPalette.BrightText,      c(RED))
    p.setColor(QPalette.Button,          c(SURFACE0))
    p.setColor(QPalette.ButtonText,      c(TEXT))
    p.setColor(QPalette.Highlight,       c(ACCENT))
    p.setColor(QPalette.HighlightedText, c(CRUST))
    p.setColor(QPalette.Link,            c(BLUE))
    p.setColor(QPalette.LinkVisited,     c(MAUVE))
    p.setColor(QPalette.ToolTipBase,     c(SURFACE1))
    p.setColor(QPalette.ToolTipText,     c(TEXT))
    p.setColor(QPalette.PlaceholderText, c(OVERLAY0))

    p.setColor(QPalette.Disabled, QPalette.Text,       c(OVERLAY0))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, c(OVERLAY0))
    p.setColor(QPalette.Disabled, QPalette.WindowText, c(OVERLAY0))

    app.setPalette(p)


STYLESHEET = f"""
/* ── Global ──────────────────────────────────────────────────────────────── */
QWidget {{
    background-color: {BASE};
    color: {TEXT};
    font-family: "Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
    selection-background-color: {ACCENT};
    selection-color: {CRUST};
}}

QWidget:disabled {{
    color: {OVERLAY0};
}}

/* ── Main window / splitter ───────────────────────────────────────────────── */
QMainWindow {{
    background-color: {BASE};
}}

QSplitter::handle {{
    background-color: {SURFACE0};
    width: 2px;
    height: 2px;
}}

QSplitter::handle:hover {{
    background-color: {ACCENT};
}}

/* ── Toolbar ──────────────────────────────────────────────────────────────── */
QToolBar {{
    background-color: {MANTLE};
    border-bottom: 1px solid {SURFACE0};
    padding: 4px 6px;
    spacing: 4px;
}}

QToolBar::separator {{
    background-color: {SURFACE1};
    width: 1px;
    margin: 4px 6px;
}}

/* ── Buttons ──────────────────────────────────────────────────────────────── */
QPushButton {{
    background-color: {SURFACE0};
    color: {TEXT};
    border: 1px solid {SURFACE1};
    border-radius: 6px;
    padding: 5px 14px;
    min-height: 22px;
}}

QPushButton:hover {{
    background-color: {SURFACE1};
    border-color: {ACCENT};
    color: {TEXT};
}}

QPushButton:pressed {{
    background-color: {SURFACE2};
    border-color: {ACCENT_ALT};
}}

QPushButton:disabled {{
    background-color: {MANTLE};
    color: {OVERLAY0};
    border-color: {SURFACE0};
}}

QPushButton:focus {{
    border-color: {ACCENT_ALT};
    outline: none;
}}

/* Destructive / red button via inline stylesheet (delete person) */
QPushButton[destructive="true"] {{
    color: {RED};
    border-color: {MAROON};
}}

QPushButton[destructive="true"]:hover {{
    background-color: #3d2030;
    border-color: {RED};
}}

/* ── Tab widget ───────────────────────────────────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {SURFACE0};
    border-top: none;
    background-color: {BASE};
}}

QTabBar::tab {{
    background-color: {MANTLE};
    color: {SUBTEXT0};
    border: 1px solid {SURFACE0};
    border-bottom: none;
    padding: 7px 18px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
}}

QTabBar::tab:selected {{
    background-color: {BASE};
    color: {TEXT};
    border-bottom: 2px solid {ACCENT};
}}

QTabBar::tab:hover:!selected {{
    background-color: {SURFACE0};
    color: {TEXT};
}}

/* ── GroupBox ─────────────────────────────────────────────────────────────── */
QGroupBox {{
    background-color: {MANTLE};
    border: 1px solid {SURFACE0};
    border-radius: 8px;
    margin-top: 22px;
    padding: 6px 4px 4px 4px;
    font-weight: 600;
    color: {SUBTEXT1};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    top: 3px;
    padding: 0 6px;
    color: {ACCENT};
    font-size: 12px;
    background-color: {MANTLE};
}}

/* ── List widget ──────────────────────────────────────────────────────────── */
QListWidget {{
    background-color: {MANTLE};
    border: 1px solid {SURFACE0};
    border-radius: 6px;
    outline: none;
}}

QListWidget::item {{
    padding: 5px 8px;
    border-radius: 4px;
}}

QListWidget::item:alternate {{
    background-color: {CRUST};
}}

QListWidget::item:selected {{
    background-color: {ACCENT};
    color: {CRUST};
}}

QListWidget::item:hover:!selected {{
    background-color: {SURFACE0};
}}

/* ── Line edit / search ───────────────────────────────────────────────────── */
QLineEdit {{
    background-color: {CRUST};
    border: 1px solid {SURFACE1};
    border-radius: 6px;
    padding: 5px 10px;
    color: {TEXT};
    selection-background-color: {ACCENT};
}}

QLineEdit:focus {{
    border-color: {ACCENT};
    background-color: {MANTLE};
}}

QLineEdit::placeholder {{
    color: {OVERLAY0};
}}

/* ── Scroll bars ──────────────────────────────────────────────────────────── */
QScrollBar:vertical {{
    background-color: {MANTLE};
    width: 10px;
    border-radius: 5px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background-color: {SURFACE2};
    border-radius: 5px;
    min-height: 30px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {ACCENT};
}}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background-color: {MANTLE};
    height: 10px;
    border-radius: 5px;
}}

QScrollBar::handle:horizontal {{
    background-color: {SURFACE2};
    border-radius: 5px;
    min-width: 30px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: {ACCENT};
}}

QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {{
    width: 0;
}}

QScrollArea {{
    background-color: transparent;
    border: none;
}}

/* ── Dock widget ──────────────────────────────────────────────────────────── */
QDockWidget {{
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
    color: {SUBTEXT0};
}}

QDockWidget::title {{
    background-color: {MANTLE};
    padding: 5px 8px;
    border-bottom: 1px solid {SURFACE0};
    font-weight: 600;
    color: {SUBTEXT1};
    text-align: left;
}}

QDockWidget::close-button,
QDockWidget::float-button {{
    border: none;
    background: transparent;
    padding: 2px;
}}

QDockWidget::close-button:hover,
QDockWidget::float-button:hover {{
    background-color: {SURFACE0};
    border-radius: 3px;
}}

/* ── Status bar ───────────────────────────────────────────────────────────── */
QStatusBar {{
    background-color: {MANTLE};
    border-top: 1px solid {SURFACE0};
    color: {SUBTEXT0};
    font-size: 12px;
    padding: 2px 8px;
}}

QStatusBar::item {{
    border: none;
}}

/* ── Progress bar ─────────────────────────────────────────────────────────── */
QProgressBar {{
    background-color: {CRUST};
    border: 1px solid {SURFACE0};
    border-radius: 5px;
    height: 8px;
    text-align: center;
    color: transparent;
}}

QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 4px;
}}

/* ── Plain text edit (log) ────────────────────────────────────────────────── */
QPlainTextEdit {{
    background-color: {CRUST};
    color: {TEXT};
    border: 1px solid {SURFACE0};
    border-radius: 6px;
    font-family: "Cascadia Code", "Fira Code", "Consolas", monospace;
    font-size: 11px;
    selection-background-color: {ACCENT};
}}

/* ── Tooltip ──────────────────────────────────────────────────────────────── */
QToolTip {{
    background-color: {SURFACE1};
    color: {TEXT};
    border: 1px solid {SURFACE2};
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 12px;
}}

/* ── Message box ──────────────────────────────────────────────────────────── */
QMessageBox {{
    background-color: {BASE};
}}

QMessageBox QLabel {{
    color: {TEXT};
}}

/* ── Dialog ───────────────────────────────────────────────────────────────── */
QDialog {{
    background-color: {BASE};
}}

/* ── Combo box ────────────────────────────────────────────────────────────── */
QComboBox {{
    background-color: {SURFACE0};
    border: 1px solid {SURFACE1};
    border-radius: 6px;
    padding: 5px 10px;
    color: {TEXT};
    min-height: 22px;
}}

QComboBox:hover {{
    border-color: {ACCENT};
}}

QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 22px;
    border-left: 1px solid {SURFACE1};
    border-top-right-radius: 6px;
    border-bottom-right-radius: 6px;
}}

QComboBox QAbstractItemView {{
    background-color: {SURFACE0};
    border: 1px solid {SURFACE1};
    selection-background-color: {ACCENT};
    selection-color: {CRUST};
    outline: none;
}}

/* ── Spin box ─────────────────────────────────────────────────────────────── */
QSpinBox, QDoubleSpinBox {{
    background-color: {CRUST};
    border: 1px solid {SURFACE1};
    border-radius: 6px;
    padding: 4px 8px;
    color: {TEXT};
}}

QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {ACCENT};
}}

/* ── Check box ────────────────────────────────────────────────────────────── */
QCheckBox {{
    spacing: 8px;
    color: {TEXT};
}}

QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 2px solid {SURFACE2};
    border-radius: 4px;
    background-color: {MANTLE};
}}

QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}

QCheckBox::indicator:hover {{
    border-color: {ACCENT_ALT};
}}

/* ── Label ────────────────────────────────────────────────────────────────── */
QLabel {{
    background-color: transparent;
    color: {TEXT};
}}

/* ── Header view (tables) ─────────────────────────────────────────────────── */
QHeaderView::section {{
    background-color: {SURFACE0};
    color: {SUBTEXT1};
    border: none;
    border-right: 1px solid {SURFACE1};
    padding: 5px 8px;
    font-weight: 600;
}}

/* ── Table / tree widget ──────────────────────────────────────────────────── */
QTableWidget, QTreeWidget {{
    background-color: {MANTLE};
    border: 1px solid {SURFACE0};
    border-radius: 6px;
    gridline-color: {SURFACE0};
    outline: none;
}}

QTableWidget::item, QTreeWidget::item {{
    padding: 4px;
}}

QTableWidget::item:selected, QTreeWidget::item:selected {{
    background-color: {ACCENT};
    color: {CRUST};
}}

/* ── Separator ────────────────────────────────────────────────────────────── */
QFrame[frameShape="4"],   /* HLine */
QFrame[frameShape="5"] {{ /* VLine */
    background-color: {SURFACE0};
    border: none;
    max-height: 1px;
}}
"""


def apply_theme(app: QApplication) -> None:
    """Apply palette + full QSS stylesheet to the application."""
    apply_palette(app)
    app.setStyleSheet(STYLESHEET)
