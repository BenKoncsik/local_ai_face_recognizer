"""AI decision visualization window.

Shows a live view of the deep classifier's internal state for every face
that is processed during an AI scan:

  - The face crop (the actual image patch fed to the embedder)
  - Per-hidden-layer neuron activations (top-N neurons as horizontal bars)
  - Output probability distribution across all known persons
  - Gate results and the final decision

Updates in real-time via :meth:`update_info` called from the main window.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap, QPolygon
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.ui.i18n import t


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_TOP_N_NEURONS = 20   # neurons shown per layer
_TOP_N_PERSONS = 12   # persons shown in the output chart
_BAR_HEIGHT = 16
_BAR_GAP = 4
_LABEL_W = 140


class _BarChartWidget(QWidget):
    """Draws a ranked horizontal bar chart using QPainter.

    Each entry is ``(label: str, value: float, max_value: float)``.
    Bars are colored from dark-blue (zero) to bright-orange (one).
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._entries: List[Tuple[str, float, float]] = []
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

    def set_entries(self, entries: List[Tuple[str, float, float]]) -> None:
        self._entries = entries
        h = max(1, len(entries)) * (_BAR_HEIGHT + _BAR_GAP) + _BAR_GAP
        self.setMinimumHeight(h)
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        if not self._entries:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        font = QFont()
        font.setPointSize(9)
        p.setFont(font)

        w = self.width()
        bar_area = max(1, w - _LABEL_W - 50)

        for i, (label, value, max_val) in enumerate(self._entries):
            y = _BAR_GAP + i * (_BAR_HEIGHT + _BAR_GAP)
            frac = min(1.0, value / max_val) if max_val > 0 else 0.0

            # Background track
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(40, 40, 55))
            p.drawRect(_LABEL_W, y, bar_area, _BAR_HEIGHT)

            # Filled bar — blue→orange gradient approximated by lerp
            r = int(30 + frac * 225)
            g = int(100 + frac * 80)
            b = int(200 - frac * 160)
            p.setBrush(QColor(r, g, b))
            p.drawRect(_LABEL_W, y, int(bar_area * frac), _BAR_HEIGHT)

            # Label (left)
            p.setPen(QColor(180, 180, 200))
            p.drawText(0, y, _LABEL_W - 4, _BAR_HEIGHT, Qt.AlignRight | Qt.AlignVCenter, label)

            # Value (right of bar)
            p.setPen(QColor(220, 220, 220))
            p.drawText(
                _LABEL_W + bar_area + 4, y, 44, _BAR_HEIGHT,
                Qt.AlignLeft | Qt.AlignVCenter,
                f"{value:.3f}",
            )

        p.end()


_GATE_DISPLAY_NAMES = {
    "outlier":    "Outlier gate",
    "prob":       "Probability gate",
    "margin":     "Margin gate",
    "sim_floor":  "Sim-floor gate",
    "sim_margin": "Sim-margin gate",
}

_GATE_DESCRIPTIONS = {
    "outlier":    "Best global similarity ≥ threshold\n(must resemble someone we know)",
    "prob":       "Ensemble probability ≥ per-person threshold\n(MLP confident enough)",
    "margin":     "Probability gap to runner-up ≥ min margin\n(no ambiguity between persons)",
    "sim_floor":  "Max cosine sim to winner's examples ≥ floor\n(not just MLP overconfidence)",
    "sim_margin": "Winner sim − runner-up sim ≥ margin\n(winner clearly closer in embedding space)",
}

# Colors
_C_PASS   = QColor(80, 200, 100)
_C_FAIL   = QColor(240, 80, 80)
_C_SKIP   = QColor(90, 90, 110)
_C_BG     = QColor(30, 30, 46)
_C_START  = QColor(100, 140, 220)
_C_ASSIGN = QColor(60, 190, 120)
_C_TEXT   = QColor(220, 220, 235)
_C_DIM    = QColor(100, 100, 115)

# Layout constants
_NODE_W   = 240
_NODE_H   = 52
_NODE_R   = 8       # corner radius
_ROW_H    = 90      # vertical distance between gate node centres
_CX       = 180     # centre-x of the main (left) column
_REJ_CX   = 530     # centre-x of the rejection column
_ARROW_W  = 10      # arrowhead half-width


def _arrow(p: QPainter, x1: int, y1: int, x2: int, y2: int, color: QColor) -> None:
    """Draw a line with a filled arrowhead at (x2, y2)."""
    pen = QPen(color, 2)
    pen.setCapStyle(Qt.RoundCap)
    p.setPen(pen)
    p.setBrush(color)
    p.drawLine(x1, y1, x2, y2)
    # Arrowhead (triangle pointing downward / rightward)
    dx, dy = x2 - x1, y2 - y1
    length = max(1, (dx * dx + dy * dy) ** 0.5)
    ux, uy = dx / length, dy / length   # unit vector along arrow
    px, py = -uy, ux                     # perpendicular
    tip = QPoint(x2, y2)
    left  = QPoint(int(x2 - ux * 12 + px * _ARROW_W), int(y2 - uy * 12 + py * _ARROW_W))
    right = QPoint(int(x2 - ux * 12 - px * _ARROW_W), int(y2 - uy * 12 - py * _ARROW_W))
    p.setPen(Qt.NoPen)
    p.drawPolygon(QPolygon([tip, left, right]))


def _node(
    p: QPainter,
    cx: int, cy: int,
    text: str,
    sub: str,
    fill: QColor,
    border: QColor,
    text_color: QColor = _C_TEXT,
    bold_title: bool = True,
) -> None:
    """Draw a rounded-rectangle node centred at (cx, cy)."""
    x, y = cx - _NODE_W // 2, cy - _NODE_H // 2
    p.setPen(QPen(border, 2))
    p.setBrush(fill)
    p.drawRoundedRect(x, y, _NODE_W, _NODE_H, _NODE_R, _NODE_R)

    font = QFont()
    font.setPointSize(9)
    font.setBold(bold_title)
    p.setFont(font)
    p.setPen(text_color)

    if sub:
        p.drawText(QRect(x + 6, y + 4, _NODE_W - 12, _NODE_H // 2 - 2),
                   Qt.AlignCenter, text)
        font.setBold(False)
        font.setPointSize(8)
        p.setFont(font)
        p.setPen(text_color.darker(130))
        p.drawText(QRect(x + 6, y + _NODE_H // 2, _NODE_W - 12, _NODE_H // 2 - 4),
                   Qt.AlignCenter, sub)
    else:
        p.drawText(QRect(x + 6, y + 4, _NODE_W - 12, _NODE_H - 8), Qt.AlignCenter, text)


class _DecisionFlowWidget(QWidget):
    """Draws a vertical flowchart of the classifier's decision path.

    Main column (left):  Start → Gate1 → Gate2 → … → Assigned
    Right column:        Rejection terminal for each failed gate

    The path actually taken is drawn in full color; nodes not reached are
    dimmed to show the graph structure without distracting from the story.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._gates: list = []
        self._prediction = None
        self._mode: str = "empty"
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self._update_height()

    def set_data(self, gates, prediction, mode: str) -> None:
        self._gates = list(gates)
        self._prediction = prediction
        self._mode = mode
        self._update_height()
        self.update()

    def _update_height(self) -> None:
        n = max(1, len(self._gates))
        # Start + n gates + final terminal, with _REJ_CX needing room too
        h = 60 + n * _ROW_H + 80
        self.setMinimumHeight(h)
        self.setMinimumWidth(_REJ_CX + _NODE_W // 2 + 20)

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        if not self._gates and self._prediction is None:
            font = QFont()
            font.setPointSize(9)
            p.setFont(font)
            p.setPen(_C_DIM)
            p.drawText(self.rect(), Qt.AlignCenter, "—")
            p.end()
            return

        pred = self._prediction
        reason = pred.reason if pred else "untrained"
        assigned = reason == "assigned"

        # ── row centres ──────────────────────────────────────────────────
        start_y   = 44
        gate_ys   = [start_y + _ROW_H * (i + 1) for i in range(len(self._gates))]
        final_y   = (gate_ys[-1] if gate_ys else start_y) + _ROW_H

        # Which gate is the stopping point?
        stopping_idx = None
        for i, g in enumerate(self._gates):
            if not g.passed:
                stopping_idx = i
                break

        # ── START node ───────────────────────────────────────────────────
        _node(p, _CX, start_y,
              "Face embedding",
              f"norm {pred.similarity:.3f}" if pred else "",
              _C_BG.lighter(140), _C_START, _C_START)

        # ── Gate nodes and arrows ─────────────────────────────────────────
        prev_y = start_y
        for i, gate in enumerate(self._gates):
            gy = gate_ys[i]
            reached = stopping_idx is None or i <= stopping_idx
            passed  = gate.passed

            # Arrow: prev node → this gate
            arr_color = _C_PASS if (reached and (i == 0 or self._gates[i - 1].passed)) else _C_DIM
            _arrow(p, _CX, prev_y + _NODE_H // 2, _CX, gy - _NODE_H // 2, arr_color)

            # Gate node fill & border
            if not reached:
                fill   = _C_BG
                border = _C_SKIP
                tc     = _C_DIM
            elif passed:
                fill   = _C_PASS.darker(250)
                border = _C_PASS
                tc     = _C_TEXT
            else:
                fill   = _C_FAIL.darker(260)
                border = _C_FAIL
                tc     = _C_TEXT

            gate_name = _GATE_DISPLAY_NAMES.get(gate.name, gate.name)
            sub_text  = f"val={gate.value:.3f}  thr={gate.threshold:.3f}"
            _node(p, _CX, gy, gate_name, sub_text, fill, border, tc)

            # "PASS" / "FAIL" label on arrow
            label_color = (arr_color if not reached else (_C_PASS if passed else _C_FAIL))
            label_text = "✓ PASS" if passed else "✗ FAIL"
            font = QFont()
            font.setPointSize(8)
            font.setBold(True)
            p.setFont(font)
            p.setPen(label_color if reached else _C_DIM)
            p.drawText(_CX + _NODE_W // 2 + 4, prev_y + _NODE_H // 2 - 2,
                       60, 16, Qt.AlignLeft | Qt.AlignVCenter,
                       label_text if reached else "")

            # Rejection branch (right column) — only for the stopping gate
            if not passed and reached:
                rej_color = _C_FAIL
                rej_fill  = _C_FAIL.darker(260)
                rej_text  = f"REJECTED\n{gate.name}"
                # Horizontal arrow right
                _arrow(p, _CX + _NODE_W // 2, gy, _REJ_CX - _NODE_W // 2, gy, rej_color)
                # FAIL label on horizontal arrow
                mid_x = (_CX + _NODE_W // 2 + _REJ_CX - _NODE_W // 2) // 2 - 20
                font.setPointSize(8)
                p.setFont(font)
                p.setPen(rej_color)
                p.drawText(mid_x, gy - 16, 60, 14, Qt.AlignCenter, "✗ FAIL")
                _node(p, _REJ_CX, gy, rej_text, f"reason: {gate.name}", rej_fill, rej_color)

            prev_y = gy

        # ── Final arrow + terminal ────────────────────────────────────────
        if stopping_idx is None:
            # All gates passed → ASSIGNED
            arr_color = _C_ASSIGN
            final_text = f"ASSIGNED\n{pred.person_name or '?'}" if pred else "ASSIGNED"
            sub_text   = f"score={pred.score:.3f}  sim={pred.similarity:.3f}" if pred else ""
            _arrow(p, _CX, prev_y + _NODE_H // 2, _CX, final_y - _NODE_H // 2, arr_color)
            _node(p, _CX, final_y, final_text, sub_text,
                  _C_ASSIGN.darker(260), _C_ASSIGN, _C_ASSIGN)
        elif not self._gates:
            # No gates (untrained / no embedding)
            _arrow(p, _CX, prev_y + _NODE_H // 2, _CX, final_y - _NODE_H // 2, _C_FAIL)
            _node(p, _CX, final_y, f"REJECTED\n{reason}", "",
                  _C_FAIL.darker(260), _C_FAIL, _C_FAIL)

        p.end()


class _GatesWidget(QWidget):
    """Shows the gate results as a row of pass/fail badges."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._gates: List[Tuple[str, bool, float, float]] = []
        self.setMinimumHeight(28)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_gates(self, gates) -> None:
        self._gates = [(g.name, g.passed, g.value, g.threshold) for g in gates]
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        if not self._gates:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        p.setFont(font)

        x = 4
        for name, passed, value, thr in self._gates:
            color = QColor(80, 180, 80) if passed else QColor(230, 80, 80)
            symbol = "✓" if passed else "✗"
            label = f"{symbol} {name}  ({value:.3f} / {thr:.3f})"
            fm = p.fontMetrics()
            tw = fm.horizontalAdvance(label) + 16
            p.setPen(Qt.NoPen)
            p.setBrush(color.darker(180))
            p.drawRoundedRect(x, 4, tw, 20, 4, 4)
            p.setPen(color)
            p.drawText(x, 4, tw, 20, Qt.AlignCenter, label)
            x += tw + 6

        p.end()


# ──────────────────────────────────────────────────────────────────────────────
# Main window
# ──────────────────────────────────────────────────────────────────────────────

_HISTORY_MAX = 1000   # max faces kept in history


class AIVisualizationWindow(QDialog):
    """Non-modal window that shows deep-classifier internals in real time."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("debug_viz_title"))
        self.setMinimumWidth(700)
        self.resize(820, 860)
        self.setWindowFlags(
            self.windowFlags() | Qt.WindowStaysOnTopHint
        )
        self._pending = None          # most recent info not yet rendered
        self._history: List[object] = []   # DeepDebugInfo list
        self._view_idx: int = -1      # -1 = follow live (always latest)
        self._timer = QTimer(self)
        self._timer.setInterval(80)   # throttle to ~12 fps
        self._timer.timeout.connect(self._flush_pending)
        self._timer.start()
        self._build_ui()

    def _build_ui(self) -> None:
        from PySide6.QtWidgets import QPushButton

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # ── Navigation bar (fixed, outside scroll) ────────────────────────
        nav = QHBoxLayout()
        nav.setSpacing(4)

        self._btn_first = QPushButton("⏮")
        self._btn_prev  = QPushButton("◀")
        self._nav_label = QLabel("—")
        self._nav_label.setAlignment(Qt.AlignCenter)
        self._nav_label.setMinimumWidth(90)
        self._nav_label.setStyleSheet("font-family: monospace; font-size: 11px;")
        self._btn_next  = QPushButton("▶")
        self._btn_last  = QPushButton("⏭  Live")
        self._btn_last.setToolTip(t("debug_viz_nav_live_tooltip"))

        for btn in (self._btn_first, self._btn_prev, self._btn_next, self._btn_last):
            btn.setFixedWidth(64)
            btn.setEnabled(False)

        self._btn_first.clicked.connect(self._nav_first)
        self._btn_prev.clicked.connect(self._nav_prev)
        self._btn_next.clicked.connect(self._nav_next)
        self._btn_last.clicked.connect(self._nav_last)

        nav.addWidget(self._btn_first)
        nav.addWidget(self._btn_prev)
        nav.addWidget(self._nav_label)
        nav.addWidget(self._btn_next)
        nav.addWidget(self._btn_last)
        nav.addStretch()
        outer.addLayout(nav)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(10)

        # ── Top row: crop + decision summary ──────────────────────────────
        top = QHBoxLayout()
        top.setSpacing(12)

        self._crop_label = QLabel()
        self._crop_label.setFixedSize(128, 128)
        self._crop_label.setStyleSheet("background: #1e1e2e; border: 1px solid #45475a;")
        self._crop_label.setAlignment(Qt.AlignCenter)
        self._crop_label.setText("—")
        top.addWidget(self._crop_label)

        self._summary_label = QLabel()
        self._summary_label.setWordWrap(True)
        self._summary_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._summary_label.setStyleSheet("font-family: monospace; font-size: 11px;")
        self._summary_label.setMinimumWidth(400)
        top.addWidget(self._summary_label, stretch=1)
        layout.addLayout(top)

        # ── Decision flow graph ───────────────────────────────────────────
        flow_frame = QFrame()
        flow_frame.setFrameShape(QFrame.StyledPanel)
        flow_fl = QVBoxLayout(flow_frame)
        flow_fl.setContentsMargins(6, 4, 6, 4)
        flow_title = QLabel(f"<b>{t('debug_viz_flow_title')}</b>")
        flow_fl.addWidget(flow_title)
        self._flow_widget = _DecisionFlowWidget()
        flow_fl.addWidget(self._flow_widget)
        layout.addWidget(flow_frame)

        # ── Gate badges (compact summary row) ─────────────────────────────
        gate_frame = QFrame()
        gate_frame.setFrameShape(QFrame.StyledPanel)
        gate_fl = QVBoxLayout(gate_frame)
        gate_fl.setContentsMargins(6, 4, 6, 4)
        self._gates_widget = _GatesWidget()
        gate_fl.addWidget(self._gates_widget)
        layout.addWidget(gate_frame)

        # ── Hidden layer activations ───────────────────────────────────────
        self._layer_frames: List[QFrame] = []
        self._layer_charts: List[_BarChartWidget] = []
        self._layer_container = QWidget()
        self._layer_layout = QVBoxLayout(self._layer_container)
        self._layer_layout.setContentsMargins(0, 0, 0, 0)
        self._layer_layout.setSpacing(6)
        layout.addWidget(self._layer_container)

        # ── Output probabilities ───────────────────────────────────────────
        prob_frame = QFrame()
        prob_frame.setFrameShape(QFrame.StyledPanel)
        prob_fl = QVBoxLayout(prob_frame)
        prob_fl.setContentsMargins(6, 4, 6, 4)
        prob_title = QLabel(f"<b>{t('debug_viz_output_probs')}</b>")
        prob_fl.addWidget(prob_title)
        self._prob_chart = _BarChartWidget()
        prob_fl.addWidget(self._prob_chart)
        layout.addWidget(prob_frame)

        layout.addStretch()
        scroll.setWidget(container)
        outer.addWidget(scroll)

    # ── Public slot ───────────────────────────────────────────────────────

    def update_info(self, info: object) -> None:
        """Thread-safe: buffer the latest info; timer flushes it to the UI."""
        self._pending = info

    def _flush_pending(self) -> None:
        info = self._pending
        if info is None:
            return
        self._pending = None

        # Append to history
        if len(self._history) >= _HISTORY_MAX:
            self._history.pop(0)
            if self._view_idx > 0:
                self._view_idx -= 1
        self._history.append(info)

        # If following live (view_idx == -1), render and stay at tail
        if self._view_idx == -1:
            self._render(info)
            self._update_nav()

    # ── Navigation ────────────────────────────────────────────────────────

    def _nav_first(self) -> None:
        if not self._history:
            return
        self._view_idx = 0
        self._render(self._history[0])
        self._update_nav()

    def _nav_prev(self) -> None:
        if not self._history:
            return
        cur = len(self._history) - 1 if self._view_idx == -1 else self._view_idx
        new = max(0, cur - 1)
        self._view_idx = new
        self._render(self._history[new])
        self._update_nav()

    def _nav_next(self) -> None:
        if not self._history:
            return
        if self._view_idx == -1:
            return  # already at live
        new = self._view_idx + 1
        if new >= len(self._history):
            self._view_idx = -1   # caught up to live
        else:
            self._view_idx = new
        idx = len(self._history) - 1 if self._view_idx == -1 else self._view_idx
        self._render(self._history[idx])
        self._update_nav()

    def _nav_last(self) -> None:
        if not self._history:
            return
        self._view_idx = -1
        self._render(self._history[-1])
        self._update_nav()

    def _update_nav(self) -> None:
        n = len(self._history)
        live = self._view_idx == -1
        cur = n if live else self._view_idx + 1
        self._nav_label.setText(f"{cur} / {n}")
        at_first = (not live and self._view_idx == 0)
        self._btn_first.setEnabled(n > 0 and not at_first)
        self._btn_prev.setEnabled(n > 0 and not at_first)
        self._btn_next.setEnabled(n > 0 and not live)
        self._btn_last.setEnabled(n > 0 and not live)
        # Highlight live button when following
        self._btn_last.setStyleSheet(
            "color: #a6e3a1; font-weight: bold;" if live else ""
        )

    # ── Rendering ─────────────────────────────────────────────────────────

    def _render(self, info) -> None:
        from app.deep.debug_info import DeepDebugInfo
        if not isinstance(info, DeepDebugInfo):
            return

        pred = info.prediction

        # Face crop
        if info.crop_path:
            pix = QPixmap(info.crop_path)
            if not pix.isNull():
                self._crop_label.setPixmap(
                    pix.scaled(128, 128, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
                self._crop_label.setText("")
            else:
                self._crop_label.setText("?")
        else:
            self._crop_label.setText("—")

        # Decision summary
        color = "#a6e3a1" if pred.reason == "assigned" else "#f38ba8"
        reason_map = {
            "assigned": t("debug_viz_reason_assigned"),
            "outlier": t("debug_viz_reason_outlier"),
            "below_threshold": t("debug_viz_reason_threshold"),
            "margin": t("debug_viz_reason_margin"),
            "prototype": t("debug_viz_reason_prototype"),
            "untrained": t("debug_viz_reason_untrained"),
            "no_embedding": t("debug_viz_reason_no_embedding"),
        }
        reason_text = reason_map.get(pred.reason, pred.reason)
        name_text = pred.person_name or "—"
        face_text = f"Face #{info.face_id}" if info.face_id >= 0 else "—"
        summary = (
            f"<b style='color:{color}'>{reason_text}</b><br>"
            f"{face_text}<br>"
            f"<b>{t('debug_viz_person')}:</b> {name_text}<br>"
            f"<b>{t('debug_viz_score')}:</b> {pred.score:.4f}<br>"
            f"<b>{t('debug_viz_similarity')}:</b> {pred.similarity:.4f}<br>"
            f"<b>{t('debug_viz_probability')}:</b> {pred.probability:.4f}<br>"
            f"<b>{t('debug_viz_margin')}:</b> {pred.margin:.4f}<br>"
            f"<b>{t('debug_viz_mode')}:</b> {info.mode}<br>"
            f"<b>{t('debug_viz_emb_norm')}:</b> {info.embedding_norm:.4f}"
        )
        self._summary_label.setText(summary)

        # Decision flow graph
        self._flow_widget.set_data(info.gates, pred, info.mode)

        # Gate badges (compact row)
        self._gates_widget.set_gates(info.gates)
        self._gates_widget.setMinimumHeight(28 if not info.gates else 32)

        # Hidden layer activations
        n_layers = len(info.layer_activations)
        # Grow chart list if needed
        while len(self._layer_charts) < n_layers:
            idx = len(self._layer_charts)
            frame = QFrame()
            frame.setFrameShape(QFrame.StyledPanel)
            fl = QVBoxLayout(frame)
            fl.setContentsMargins(6, 4, 6, 4)
            lbl = QLabel()
            fl.addWidget(lbl)
            chart = _BarChartWidget()
            fl.addWidget(chart)
            self._layer_frames.append(frame)
            self._layer_charts.append(chart)
            self._layer_layout.addWidget(frame)
            frame.setProperty("layer_label", lbl)

        for i, activations in enumerate(info.layer_activations):
            frame = self._layer_frames[i]
            chart = self._layer_charts[i]
            lbl: QLabel = frame.property("layer_label")
            n = len(activations)
            lbl.setText(f"<b>{t('debug_viz_layer')} {i + 1}</b> ({n} {t('debug_viz_neurons')})")
            top_idx = np.argsort(activations)[::-1][:_TOP_N_NEURONS]
            max_val = float(activations[top_idx[0]]) if len(top_idx) else 1.0
            entries = [
                (f"#{int(j)}", float(activations[j]), max(max_val, 1e-6))
                for j in top_idx
            ]
            chart.set_entries(entries)
            frame.setVisible(True)

        # Hide unused frames
        for i in range(n_layers, len(self._layer_charts)):
            self._layer_frames[i].setVisible(False)

        # Output probabilities
        if info.output_probs:
            sorted_probs = sorted(info.output_probs.items(), key=lambda x: -x[1])
            top = sorted_probs[:_TOP_N_PERSONS]
            max_prob = top[0][1] if top else 1.0
            self._prob_chart.set_entries([
                (name, prob, max(max_prob, 1e-6))
                for name, prob in top
            ])
        else:
            # Prototype / untrained — show cosine similarities instead
            if info.all_similarities:
                sorted_sims = sorted(info.all_similarities.items(), key=lambda x: -x[1])
                top = sorted_sims[:_TOP_N_PERSONS]
                max_sim = top[0][1] if top else 1.0
                self._prob_chart.set_entries([
                    (name, sim, max(max_sim, 1e-6))
                    for name, sim in top
                ])
