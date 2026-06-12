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

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)
from PySide6.QtCore import QPoint, QRect, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPixmap, QPolygon
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
# Neural-network graph visualization — node/circle based
# ──────────────────────────────────────────────────────────────────────────────

_NN_NODE_R     = 14    # node circle radius (px)
_NN_NODE_DIA   = 2 * _NN_NODE_R
_NN_NODE_PITCH = _NN_NODE_DIA + 10   # vertical centre-to-centre spacing = 38px
_NN_TOP_N      = 15    # nodes shown per layer

_NN_IDX_W      = 46    # left index label (non-output: "d89", "n32")
_NN_VAL_W      = 48    # right value label ("0.254")
_NN_COL_W      = _NN_IDX_W + _NN_NODE_DIA + 6 + _NN_VAL_W   # ≈ 122px
_NN_OUT_NAME_W = 138   # output: person name label width
_NN_OUT_COL_W  = _NN_NODE_DIA + 8 + _NN_OUT_NAME_W + _NN_VAL_W   # ≈ 222px

_NN_CONN_W     = 90    # Bézier connection area between columns
_NN_HEAD_H     = 54    # column header height
_NN_FOOT_H     = 36    # column footer height
_NN_CONTENT_H  = _NN_TOP_N * _NN_NODE_PITCH + 10   # ≈ 580px
_NN_TOTAL_H    = _NN_HEAD_H + _NN_CONTENT_H + _NN_FOOT_H + 20
_NN_PAD        = 14    # left/right page margin
_NN_FAN        = 4     # max fan-out Bézier connections per source node

_C_NODE_BG     = QColor(28, 28, 44)   # inactive node fill
_C_NODE_BORDER = QColor(65, 65, 88)   # default node border


def _nn_color(frac: float, alpha: int = 255) -> QColor:
    """Map [0, 1] activation fraction to dark-blue → bright-orange."""
    frac = max(0.0, min(1.0, frac))
    r = int(30  + frac * 225)
    g = int(100 + frac * 80)
    b = int(200 - frac * 160)
    c = QColor(r, g, b)
    c.setAlpha(alpha)
    return c


class NeuralNetworkGraphWidget(QWidget):
    """Neural-network activation graph: circle nodes + Bézier edge connections.

    Columns (left → right):
      • Input embedding  — top-N active embedding dimensions
      • Hidden layer(s)  — top-N neurons per hidden layer (when available)
      • Output / Sim     — top-N output probabilities or cosine similarities

    Each node is a circle filled with the activation colour (dark = inactive,
    orange = highly active).  Bézier curves fan between adjacent columns,
    with opacity proportional to the source node's activation.
    Prototype mode (no MLP) shows cosine similarities as the output column.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        # Each entry: (title, vals_ndarray, labels_list, is_output)
        self._layers: List[tuple] = []
        self._winner: str = ""
        # Decision-path overlay: active labels per column + edges per boundary
        self._path_node_cols: List[set] = []
        self._path_edge_gaps: List[List[tuple]] = []
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumHeight(_NN_TOTAL_H)
        self.setMinimumWidth(400)

    def set_data(
        self,
        embedding_top_dims: List[tuple],
        layer_activations: List[np.ndarray],
        output_probs: Dict[str, float],
        all_similarities: Optional[Dict[str, float]] = None,
        embedding_dim: int = 192,
        winner: str = "",
        path: Optional[dict] = None,
    ) -> None:
        layers = []

        # ── Input: top-N active embedding dims ───────────────────────────
        top_dims = sorted(embedding_top_dims or [], key=lambda x: -abs(x[1]))[:_NN_TOP_N]
        if top_dims:
            labels = [f"d{idx}" for idx, _ in top_dims]
            vals   = np.array([abs(float(v)) for _, v in top_dims], dtype=np.float64)
            layers.append((f"Input ({embedding_dim})", vals, labels, False))

        # ── Hidden layers: top-N neurons by activation ───────────────────
        for i, acts in enumerate(layer_activations or []):
            a = np.abs(np.asarray(acts, dtype=np.float64))
            top_idx = np.argsort(a)[::-1][:_NN_TOP_N]
            labels  = [f"n{int(j)}" for j in top_idx]
            vals    = a[top_idx]
            layers.append((f"Hidden {i + 1}  ({len(a)})", vals, labels, False))

        # ── Output: probabilities; fallback → cosine similarities ────────
        if output_probs:
            top = sorted(output_probs.items(), key=lambda kv: -kv[1])[:_NN_TOP_N]
            labels = [n for n, _ in top]
            vals   = np.array([v for _, v in top], dtype=np.float64)
            layers.append((f"Output  ({len(output_probs)})", vals, labels, True))
        elif all_similarities:
            top = sorted(all_similarities.items(), key=lambda kv: -kv[1])[:_NN_TOP_N]
            labels = [n for n, _ in top]
            vals   = np.array([v for _, v in top], dtype=np.float64)
            layers.append((f"Cosine sim  ({len(all_similarities)})", vals, labels, True))

        self._layers = layers
        self._winner = winner
        # The overlay only applies when its column structure matches the
        # rendered one (input + hidden layers + output) exactly.
        node_cols = (path or {}).get("nodes") or []
        edge_gaps = (path or {}).get("edges") or []
        if len(node_cols) == len(layers) and len(edge_gaps) == len(layers) - 1:
            self._path_node_cols = [set(map(str, col)) for col in node_cols]
            self._path_edge_gaps = [
                [(str(e[0]), str(e[1]), float(e[2])) for e in gap if len(e) >= 3]
                for gap in edge_gaps
            ]
        else:
            self._path_node_cols = []
            self._path_edge_gaps = []

        n = len(layers)
        has_out = n > 0 and layers[-1][3]
        inner   = (n - 1) if has_out else n
        w = _NN_PAD + inner * _NN_COL_W + max(0, inner - 1) * _NN_CONN_W
        if has_out:
            w += _NN_CONN_W + _NN_OUT_COL_W
        w += _NN_PAD
        self.setMinimumSize(max(400, w), _NN_TOTAL_H)
        self.resize(max(400, w), _NN_TOTAL_H)
        self.update()

    # ── Geometry helpers ─────────────────────────────────────────────────

    def _col_x(self, i: int) -> int:
        """Left x of column i."""
        return _NN_PAD + i * (_NN_COL_W + _NN_CONN_W)

    def _node_cx(self, i: int) -> int:
        """Centre-x of the circle in column i."""
        is_out = bool(self._layers[i][3]) if i < len(self._layers) else False
        x = self._col_x(i)
        return x + (_NN_NODE_R + 4 if is_out else _NN_IDX_W + _NN_NODE_R + 2)

    def _node_cy(self, rank: int) -> int:
        """Centre-y of node at rank (0-based)."""
        return _NN_HEAD_H + 6 + rank * _NN_NODE_PITCH + _NN_NODE_R

    # ── Paint ─────────────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        if not self._layers:
            font = QFont()
            font.setPointSize(10)
            p.setFont(font)
            p.setPen(_C_DIM)
            p.drawText(self.rect(), Qt.AlignCenter, t("debug_nn_no_data"))
            p.end()
            return

        # 1. Bézier edge connections (drawn behind nodes)
        for i in range(len(self._layers) - 1):
            self._draw_connections(p, i)

        # 2. Decision-path edges in green, over the generic connections
        for i in range(len(self._layers) - 1):
            self._draw_path_edges(p, i)

        # 3. Node columns on top
        for i, (title, vals, labels, is_out) in enumerate(self._layers):
            self._draw_column(p, i, title, vals, labels, is_out)

        p.end()

    def _draw_column(
        self,
        p: QPainter,
        col_idx: int,
        title: str,
        vals: np.ndarray,
        labels: List[str],
        is_out: bool,
    ) -> None:
        n       = len(vals)
        max_val = float(vals.max()) if n and vals.max() > 0 else 1.0
        mean_v  = float(vals.mean()) if n else 0.0
        x       = self._col_x(col_idx)
        cx      = self._node_cx(col_idx)
        col_w   = _NN_OUT_COL_W if is_out else _NN_COL_W

        # ── Header ──────────────────────────────────────────────────────
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        p.setFont(font)
        p.setPen(_C_TEXT)
        header_x = x if is_out else x
        p.drawText(header_x, 4, col_w, _NN_HEAD_H // 2,
                   Qt.AlignHCenter | Qt.AlignVCenter, title)
        if not is_out and n:
            font.setBold(False)
            font.setPointSize(7)
            p.setFont(font)
            p.setPen(_C_DIM)
            p.drawText(header_x, _NN_HEAD_H // 2 + 2, col_w, _NN_HEAD_H // 2 - 4,
                       Qt.AlignHCenter | Qt.AlignVCenter,
                       f"top {n} aktív")

        # ── Nodes ────────────────────────────────────────────────────────
        for rank in range(n):
            cy   = self._node_cy(rank)
            val  = float(vals[rank])
            frac = val / max_val
            color = _nn_color(frac)
            is_win = is_out and (labels[rank] == self._winner)

            on_path = (
                col_idx < len(self._path_node_cols)
                and labels[rank] in self._path_node_cols[col_idx]
            )

            # Glow ring behind winner / decision-path nodes
            if is_win or on_path:
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(60, 190, 120, 45))
                p.drawEllipse(cx - _NN_NODE_R - 5, cy - _NN_NODE_R - 5,
                              _NN_NODE_DIA + 10, _NN_NODE_DIA + 10)

            # Node circle
            if is_win or on_path:
                border = _C_ASSIGN
            else:
                border = color if frac > 0.15 else _C_NODE_BORDER
            fill   = QColor(
                int(_C_NODE_BG.red()   + frac * (color.red()   - _C_NODE_BG.red())),
                int(_C_NODE_BG.green() + frac * (color.green() - _C_NODE_BG.green())),
                int(_C_NODE_BG.blue()  + frac * (color.blue()  - _C_NODE_BG.blue())),
            )
            p.setPen(QPen(border, 2.5 if (is_win or on_path) else 1.5))
            p.setBrush(fill)
            p.drawEllipse(cx - _NN_NODE_R, cy - _NN_NODE_R,
                          _NN_NODE_DIA, _NN_NODE_DIA)

            # Index label (left of node, non-output only)
            if not is_out:
                font.setPointSize(7)
                font.setBold(False)
                p.setFont(font)
                p.setPen(_C_DIM)
                p.drawText(x, cy - _NN_NODE_R, _NN_IDX_W - 4, _NN_NODE_DIA,
                           Qt.AlignRight | Qt.AlignVCenter, labels[rank])

            # Value label (right of node)
            label_x = cx + _NN_NODE_R + 5
            if is_out:
                # Name + value
                font.setPointSize(8)
                font.setBold(is_win)
                p.setFont(font)
                p.setPen(_C_ASSIGN if is_win else color)
                name_str = f"{labels[rank][:17]}  {val:.3f}"
                p.drawText(label_x, cy - _NN_NODE_R,
                           _NN_OUT_NAME_W + _NN_VAL_W - 4, _NN_NODE_DIA,
                           Qt.AlignLeft | Qt.AlignVCenter, name_str)
            else:
                font.setPointSize(7)
                font.setBold(False)
                p.setFont(font)
                p.setPen(color if frac > 0.2 else _C_DIM)
                p.drawText(label_x, cy - _NN_NODE_R, _NN_VAL_W, _NN_NODE_DIA,
                           Qt.AlignLeft | Qt.AlignVCenter, f"{val:.3f}")

        # ── Footer ───────────────────────────────────────────────────────
        font.setBold(False)
        font.setPointSize(7)
        p.setFont(font)
        p.setPen(_C_DIM)
        fy = _NN_HEAD_H + _NN_CONTENT_H + 4
        p.drawText(x, fy, col_w, _NN_FOOT_H - 4,
                   Qt.AlignHCenter | Qt.AlignTop,
                   f"max {max_val:.3f}   μ {mean_v:.3f}")

    def _draw_connections(self, p: QPainter, layer_idx: int) -> None:
        _, src_vals, _, _ = self._layers[layer_idx]
        _, dst_vals, _, _ = self._layers[layer_idx + 1]

        n_src = len(src_vals)
        n_dst = len(dst_vals)
        if not n_src or not n_dst:
            return

        max_src = float(src_vals.max()) if src_vals.max() > 0 else 1.0

        src_cx = self._node_cx(layer_idx)
        dst_cx = self._node_cx(layer_idx + 1)
        x1  = src_cx + _NN_NODE_R        # right edge of source node
        x2  = dst_cx - _NN_NODE_R        # left edge of destination node
        cx1 = x1 + (x2 - x1) // 3
        cx2 = x2 - (x2 - x1) // 3

        # Reproducible fan-out: active nodes connect to several destinations
        rng = np.random.default_rng(seed=layer_idx * 997 + n_src * 31 + n_dst)
        p.setBrush(Qt.NoBrush)

        for src_rank in range(n_src):
            frac = float(src_vals[src_rank]) / max_src
            if frac < 0.04:
                continue
            fan_count = max(1, min(int(frac * _NN_FAN + 0.5), n_dst))
            dst_ranks = rng.choice(n_dst, size=fan_count, replace=False)

            y1 = self._node_cy(src_rank)
            alpha = max(22, int(frac * 95))

            for dr in dst_ranks:
                y2  = self._node_cy(int(dr))
                pen = QPen(_nn_color(frac, alpha=alpha), 1)
                pen.setCapStyle(Qt.RoundCap)
                p.setPen(pen)
                path = QPainterPath()
                path.moveTo(x1, y1)
                path.cubicTo(cx1, y1, cx2, y2, x2, y2)
                p.drawPath(path)

    def _draw_path_edges(self, p: QPainter, layer_idx: int) -> None:
        """Overlay the decision-path edges between columns i and i+1 in green."""
        if layer_idx >= len(self._path_edge_gaps):
            return
        gap_edges = self._path_edge_gaps[layer_idx]
        if not gap_edges:
            return
        src_labels = self._layers[layer_idx][2]
        dst_labels = self._layers[layer_idx + 1][2]
        src_rank = {label: r for r, label in enumerate(src_labels)}
        dst_rank = {label: r for r, label in enumerate(dst_labels)}

        src_cx = self._node_cx(layer_idx)
        dst_cx = self._node_cx(layer_idx + 1)
        x1  = src_cx + _NN_NODE_R
        x2  = dst_cx - _NN_NODE_R
        cx1 = x1 + (x2 - x1) // 3
        cx2 = x2 - (x2 - x1) // 3

        p.setBrush(Qt.NoBrush)
        for src, dst, strength in gap_edges:
            if src not in src_rank or dst not in dst_rank:
                continue
            y1 = self._node_cy(src_rank[src])
            y2 = self._node_cy(dst_rank[dst])
            color = QColor(_C_ASSIGN)
            color.setAlpha(110 + int(max(0.0, min(1.0, strength)) * 145))
            pen = QPen(color, 1.5 + 1.5 * strength)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            path = QPainterPath()
            path.moveTo(x1, y1)
            path.cubicTo(cx1, y1, cx2, y2, x2, y2)
            p.drawPath(path)


_NN_CACHE_FILE = Path("data/nn_graph_last.json")


class NeuralNetworkGraphDialog(QDialog):
    """Standalone window with the full neural-network activation graph.

    Opened from the Debug tab (Settings) or from the AI Decision Visualizer
    toolbar.  Uses ``Qt.Window`` so it shows even when Settings is modal.

    Persists the last received activation data to *data/nn_graph_last.json*
    so closing and reopening the window restores the previous state.

    Call :meth:`update_from_info` to feed a
    :class:`~app.deep.debug_info.DeepDebugInfo` object.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("debug_nn_title"))
        self.setWindowFlags(Qt.Window | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.resize(1050, 660)
        self._build_ui()
        self._load_cache()   # restore last run data immediately

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._info_label = QLabel(t("debug_nn_waiting"))
        self._info_label.setStyleSheet(
            "font-family: monospace; font-size: 10px; color: #aaa; padding: 2px 4px;"
        )
        layout.addWidget(self._info_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(False)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.NoFrame)
        self._graph = NeuralNetworkGraphWidget()
        scroll.setWidget(self._graph)
        layout.addWidget(scroll, stretch=1)

    # ── Cache helpers ─────────────────────────────────────────────────────

    def _save_cache(self, data: dict) -> None:
        try:
            _NN_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with _NN_CACHE_FILE.open("w", encoding="utf-8") as fh:
                json.dump(data, fh)
        except Exception:
            log.debug("nn_graph: cache write failed", exc_info=True)

    def _load_cache(self) -> None:
        try:
            if not _NN_CACHE_FILE.exists():
                return
            with _NN_CACHE_FILE.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._render_from_dict(data)
        except Exception:
            log.debug("nn_graph: cache read failed", exc_info=True)

    def _render_from_dict(self, data: dict) -> None:
        face_id    = data.get("face_id", -1)
        mode       = data.get("mode", "?")
        decision   = data.get("decision", "?")
        person_name = data.get("person_name") or "?"
        winner     = data.get("winner", "")

        embedding_top_dims = [tuple(x) for x in data.get("embedding_top_dims") or []]
        layer_activations  = [np.array(a, dtype=np.float64) for a in data.get("layer_activations") or []]
        output_probs       = data.get("output_probs") or {}
        all_similarities   = data.get("all_similarities") or {}

        n_hidden   = len(layer_activations)
        n_out      = len(output_probs)
        face_text  = f"Face #{face_id}" if face_id >= 0 else "—"
        mode_note  = "  (prototype — MLP nem aktív, cosine sim látható)" if mode == "prototype" else ""
        self._info_label.setText(
            f"{face_text}   mode: {mode}{mode_note}   "
            f"hidden: {n_hidden}   output: {n_out}   "
            f"döntés: {decision}  →  {person_name}"
        )
        self._graph.set_data(
            embedding_top_dims=embedding_top_dims,
            layer_activations=layer_activations,
            output_probs=output_probs,
            all_similarities=all_similarities,
            winner=winner,
            path=data.get("decision_path"),
        )

    # ── Public update ─────────────────────────────────────────────────────

    def update_from_info(self, info: object) -> None:
        """Feed a DeepDebugInfo; silently ignores objects of a wrong type."""
        from app.deep.debug_info import DeepDebugInfo
        if not isinstance(info, DeepDebugInfo):
            return
        pred      = info.prediction
        winner    = pred.person_name if pred and pred.reason == "assigned" else ""

        cache = {
            "face_id":           info.face_id,
            "mode":              info.mode,
            "decision":          pred.reason,
            "person_name":       pred.person_name or "",
            "winner":            winner,
            "embedding_top_dims": [[int(i), float(v)] for i, v in (info.embedding_top_dims or [])],
            "layer_activations": [a.tolist() for a in (info.layer_activations or [])],
            "output_probs":      {k: float(v) for k, v in (info.output_probs or {}).items()},
            "all_similarities":  {k: float(v) for k, v in (info.all_similarities or {}).items()},
            "decision_path":     info.decision_path,
        }
        self._save_cache(cache)
        self._render_from_dict(cache)


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

        self._btn_nn_graph = QPushButton(t("debug_nn_open_btn"))
        self._btn_nn_graph.setToolTip(t("debug_nn_open_tooltip"))
        self._btn_nn_graph.clicked.connect(self._open_nn_graph)
        nav.addWidget(self._btn_nn_graph)

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

    # ── Neural graph ──────────────────────────────────────────────────────

    def _open_nn_graph(self) -> None:
        if not hasattr(self, "_nn_dlg") or self._nn_dlg is None:
            self._nn_dlg = NeuralNetworkGraphDialog(parent=self)
        idx = len(self._history) - 1 if self._view_idx == -1 else self._view_idx
        if 0 <= idx < len(self._history):
            self._nn_dlg.update_from_info(self._history[idx])
        self._nn_dlg.show()
        self._nn_dlg.raise_()

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

        # Live-feed open neural graph dialog
        if hasattr(self, "_nn_dlg") and self._nn_dlg is not None and self._nn_dlg.isVisible():
            self._nn_dlg.update_from_info(info)
