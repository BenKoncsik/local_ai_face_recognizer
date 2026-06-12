"""Per-suggestion decision graph + rich detail modal for the auto-merge review.

Two dialogs power the "why was this merge suggested?" workflow:

``MergeDecisionGraphDialog``
    Renders the *stored* decision graph for one auto-merged face using the very
    same flow-chart widget as the live AI debug view
    (:class:`app.ui.dialogs.ai_visualization_window._DecisionFlowWidget`).  The
    graph is reconstructed from the JSON captured at merge time
    (``Face.auto_merge_decision_json``), so it stays viewable long after the
    scan that produced it.  When a suggestion predates the decision capture it
    shows a friendly "no saved graph" message instead of failing.

``MergeReviewDetailDialog``
    The combined modal opened by clicking a suggestion's face crop: the original
    photo (zoom/pan, the suggested face highlighted), a "what was this based on?"
    explanation, the decision-graph button, and a comparison gallery of the
    target person's *confirmed* reference faces.  Every visual is built by
    reusing existing helpers from
    :mod:`app.ui.dialogs.suggestion_viewer` — no parallel viewer implementation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.deep.debug_info import GateResult
from app.ui.dialogs.ai_visualization_window import _DecisionFlowWidget
from app.ui.dialogs.suggestion_viewer import (
    _build_annotated_pixmap,
    _make_gallery_scroll,
    _WheelZoomScrollArea,
    get_other_bboxes,
)
from app.ui.i18n import t

log = logging.getLogger(__name__)

_SUGGESTED_CROP = 168
_COMPARE_COLS = 3


# ---------------------------------------------------------------------------
# Decision → graph adapter
# ---------------------------------------------------------------------------

class _MergePrediction:
    """Lightweight stand-in for ``DeepPrediction`` that ``_DecisionFlowWidget``
    reads (``reason`` / ``person_name`` / ``score`` / ``similarity``)."""

    def __init__(self, decision: Dict[str, Any]) -> None:
        outcome = decision.get("outcome")
        best = decision.get("best_score")
        self.reason = "assigned" if outcome == "auto_confirmed" else "pending"
        self.person_name = decision.get("target_person_name") or "?"
        self.score = float(best) if best is not None else 0.0
        self.similarity = float(best) if best is not None else 0.0


def _gates_from_decision(decision: Dict[str, Any]) -> List[GateResult]:
    """Build the flow-chart gate list from a stored decision dict.

    Adds a trailing failing gate when the face stayed pending despite every
    measured gate passing (e.g. auto-merge disabled, or no reference face), so
    the flow chart's terminal honestly reflects "not auto-confirmed".
    """
    gates: List[GateResult] = []
    for g in decision.get("gates", []) or []:
        try:
            gates.append(
                GateResult(
                    name=str(g.get("name", "?")),
                    passed=bool(g.get("passed", False)),
                    value=float(g.get("value", 0.0)),
                    threshold=float(g.get("threshold", 0.0)),
                )
            )
        except (TypeError, ValueError):
            continue

    # If the face stayed pending but every *measured* gate passed (or there were
    # none — e.g. no embedding to score), append an explicit failing terminal so
    # the flow chart honestly shows "not auto-confirmed" instead of "ASSIGNED".
    # (``all([])`` is True, so the empty-gates case is covered here too.)
    confirmed = decision.get("outcome") == "auto_confirmed"
    if not confirmed and all(g.passed for g in gates):
        if not decision.get("auto_merge_enabled", True):
            name = "auto_merge_off"
        elif decision.get("best_score") is None:
            name = "no_score"
        elif not decision.get("target_has_profile", True):
            name = "no_reference"
        else:
            name = "manual_review"
        gates.append(GateResult(name=name, passed=False, value=0.0, threshold=1.0))
    return gates


# ---------------------------------------------------------------------------
# MergeDecisionGraphDialog
# ---------------------------------------------------------------------------

class MergeDecisionGraphDialog(QDialog):
    """Show the stored decision graph for one auto-merge suggestion."""

    def __init__(
        self,
        decision: Optional[Dict[str, Any]],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("amerge_graph_title"))
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)
        self.resize(640, 720)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        if not decision:
            msg = QLabel(t("amerge_no_graph"))
            msg.setWordWrap(True)
            msg.setAlignment(Qt.AlignCenter)
            msg.setStyleSheet("color:#bbb; font-style:italic; padding:24px;")
            layout.addWidget(msg, stretch=1)
        else:
            header = QLabel(self._header_html(decision))
            header.setWordWrap(True)
            header.setStyleSheet("font-family: monospace; font-size: 11px; color:#ddd;")
            layout.addWidget(header)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            flow = _DecisionFlowWidget()
            flow.set_data(
                _gates_from_decision(decision),
                _MergePrediction(decision),
                mode="merge",
            )
            scroll.setWidget(flow)
            layout.addWidget(scroll, stretch=1)

            ranked = decision.get("ranked") or []
            if ranked:
                rank_lbl = QLabel(self._ranked_html(ranked, decision))
                rank_lbl.setWordWrap(True)
                rank_lbl.setStyleSheet(
                    "font-family: monospace; font-size: 11px; color:#bbb;"
                )
                layout.addWidget(rank_lbl)

        close_btn = QPushButton(t("close"))
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignRight)

    @staticmethod
    def _header_html(decision: Dict[str, Any]) -> str:
        target = decision.get("target_person_name") or "?"
        source = decision.get("source_person_name") or "?"
        best = decision.get("best_score")
        best_txt = f"{best:.3f}" if isinstance(best, (int, float)) else "—"
        return (
            f"<b>{t('amerge_why_target')}:</b> {target} &nbsp; "
            f"<b>{t('amerge_why_source')}:</b> {source}<br>"
            f"<b>{t('amerge_score')}:</b> {best_txt} &nbsp; "
            f"<b>{t('amerge_threshold')}:</b> {decision.get('threshold', 0):.2f} &nbsp; "
            f"<b>{t('amerge_margin')}:</b> {decision.get('min_margin', 0):.2f}"
        )

    @staticmethod
    def _ranked_html(ranked: List, decision: Dict[str, Any]) -> str:
        target_id = decision.get("target_person_id")
        rows = []
        for entry in ranked[:8]:
            try:
                pid, name, score = entry[0], entry[1], float(entry[2])
            except (TypeError, ValueError, IndexError):
                continue
            mark = " ◀" if pid == target_id else ""
            rows.append(f"&nbsp;&nbsp;{name}: {score:.3f}{mark}")
        if not rows:
            return ""
        return f"<b>{t('amerge_ranked')}:</b><br>" + "<br>".join(rows)


# ---------------------------------------------------------------------------
# Deep-engine decision graph (AI auto-grouping tab)
# ---------------------------------------------------------------------------

class _FlowPrediction:
    """Stand-in carrying the attributes ``_DecisionFlowWidget`` reads from a
    prediction (``reason`` / ``person_name`` / ``score`` / ``similarity``)."""

    def __init__(self, reason: str, person_name: str, score: float, similarity: float) -> None:
        self.reason = reason
        self.person_name = person_name
        self.score = score
        self.similarity = similarity


class DeepDecisionGraphDialog(QDialog):
    """Show the stored deep-engine decision graph for one auto-assignment.

    Two tabs:

    * **Decision flow** — the open-set rejection gate flow reconstructed from
      the JSON captured at recognition time, rendered with the very same
      widget as the live AI debug view (:class:`_DecisionFlowWidget`).
    * **Neural network** — the activation graph with the strongest
      contribution path to the winning output highlighted in green.  The
      per-neuron activations are not persisted, so this view is recomputed
      from the saved model on first open (needs *face_id* + *deep_config*).
    """

    def __init__(
        self,
        decision: Optional[Dict[str, Any]],
        parent: Optional[QWidget] = None,
        face_id: Optional[int] = None,
        deep_config=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("autoAssign.graph_title"))
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)
        self.resize(760, 720)
        self._face_id = face_id
        self._deep_config = deep_config
        self._nn_loaded = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        if not decision:
            msg = QLabel(t("amerge_no_graph"))
            msg.setWordWrap(True)
            msg.setAlignment(Qt.AlignCenter)
            msg.setStyleSheet("color:#bbb; font-style:italic; padding:24px;")
            layout.addWidget(msg, stretch=1)
        else:
            self._tabs = QTabWidget()
            self._tabs.addTab(
                self._build_flow_tab(decision), t("autoAssign.graph_tab_flow")
            )
            self._nn_tab = self._build_nn_tab_placeholder()
            self._tabs.addTab(self._nn_tab, t("autoAssign.graph_tab_nn"))
            self._tabs.currentChanged.connect(self._on_tab_changed)
            layout.addWidget(self._tabs, stretch=1)

        close_btn = QPushButton(t("close"))
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignRight)

    # -- tab 1: gate flow ------------------------------------------------------

    def _build_flow_tab(self, decision: Dict[str, Any]) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(8)

        header = QLabel(self._header_html(decision))
        header.setWordWrap(True)
        header.setStyleSheet("font-family: monospace; font-size: 11px; color:#ddd;")
        layout.addWidget(header)

        gates = [
            GateResult(
                name=str(g.get("name", "?")),
                passed=bool(g.get("passed", False)),
                value=float(g.get("value", 0.0)),
                threshold=float(g.get("threshold", 0.0)),
            )
            for g in decision.get("gates", []) or []
        ]
        pred = _FlowPrediction(
            reason=decision.get("reason", "assigned"),
            person_name=decision.get("person_name") or "?",
            score=float(decision.get("score", 0.0) or 0.0),
            similarity=float(decision.get("similarity", 0.0) or 0.0),
        )

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        flow = _DecisionFlowWidget()
        flow.set_data(gates, pred, mode=decision.get("mode", "ensemble"))
        scroll.setWidget(flow)
        layout.addWidget(scroll, stretch=1)

        persons = decision.get("top_persons") or []
        if persons:
            lbl = QLabel(self._persons_html(persons))
            lbl.setWordWrap(True)
            lbl.setStyleSheet("font-family: monospace; font-size: 11px; color:#bbb;")
            layout.addWidget(lbl)
        return tab

    # -- tab 2: neural network graph (lazy) ------------------------------------

    def _build_nn_tab_placeholder(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(8)
        loading = QLabel("⏳")
        loading.setAlignment(Qt.AlignCenter)
        loading.setStyleSheet("color:#888; font-size:24px;")
        layout.addWidget(loading, stretch=1)
        return tab

    def _on_tab_changed(self, index: int) -> None:
        if self._tabs.widget(index) is self._nn_tab and not self._nn_loaded:
            self._nn_loaded = True
            self._populate_nn_tab()

    def _clear_nn_tab(self) -> QVBoxLayout:
        layout = self._nn_tab.layout()
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        return layout

    def _populate_nn_tab(self) -> None:
        """Recompute the forward pass from the saved model and render it."""
        info = self._compute_debug_info()
        layout = self._clear_nn_tab()

        has_columns = info is not None and (
            getattr(info, "output_probs", None)
            or getattr(info, "all_similarities", None)
        )
        if not has_columns:
            msg = QLabel(t("autoAssign.graph_nn_unavailable"))
            msg.setWordWrap(True)
            msg.setAlignment(Qt.AlignCenter)
            msg.setStyleSheet("color:#bbb; font-style:italic; padding:24px;")
            layout.addWidget(msg, stretch=1)
            return

        from app.ui.dialogs.ai_visualization_window import NeuralNetworkGraphWidget

        pred = info.prediction
        winner = pred.person_name if pred and pred.person_name else ""

        legend = QLabel("🟢 " + t("autoAssign.graph_nn_legend"))
        legend.setWordWrap(True)
        legend.setStyleSheet("color:#aaa; font-size:11px;")
        layout.addWidget(legend)

        scroll = QScrollArea()
        scroll.setWidgetResizable(False)
        scroll.setFrameShape(QFrame.NoFrame)
        graph = NeuralNetworkGraphWidget()
        graph.set_data(
            embedding_top_dims=info.embedding_top_dims or [],
            layer_activations=info.layer_activations or [],
            output_probs=info.output_probs or {},
            all_similarities=info.all_similarities or {},
            winner=winner,
            path=info.decision_path,
        )
        scroll.setWidget(graph)
        layout.addWidget(scroll, stretch=1)

    def _compute_debug_info(self):
        if self._face_id is None:
            return None
        try:
            from app.db.database import session_scope
            from app.services.deep_recognition_service import DeepRecognitionService

            with session_scope() as session:
                return DeepRecognitionService(
                    session, self._deep_config
                ).debug_info_for_face(self._face_id)
        except Exception:  # noqa: BLE001
            log.warning("Could not compute NN graph for face %s", self._face_id)
            return None

    @staticmethod
    def _header_html(decision: Dict[str, Any]) -> str:
        name = decision.get("person_name") or "?"
        score = decision.get("score")
        sim = decision.get("similarity")
        prob = decision.get("probability")
        score_txt = f"{score:.3f}" if isinstance(score, (int, float)) else "—"
        sim_txt = f"{sim:.3f}" if isinstance(sim, (int, float)) else "—"
        prob_txt = f"{prob:.3f}" if isinstance(prob, (int, float)) else "—"
        return (
            f"<b>{t('amerge_why_target')}:</b> {name}<br>"
            f"<b>{t('amerge_score')}:</b> {score_txt} &nbsp; "
            f"sim: {sim_txt} &nbsp; prob: {prob_txt} &nbsp; "
            f"<b>{t('debug_viz_mode')}:</b> {decision.get('mode', '—')}"
        )

    @staticmethod
    def _persons_html(persons: List) -> str:
        rows = []
        for entry in persons[:8]:
            try:
                name, prob = entry[0], float(entry[1])
            except (TypeError, ValueError, IndexError):
                continue
            rows.append(f"&nbsp;&nbsp;{name}: {prob:.3f}")
        if not rows:
            return ""
        return f"<b>{t('amerge_ranked')}:</b><br>" + "<br>".join(rows)


# ---------------------------------------------------------------------------
# MergeReviewDetailDialog
# ---------------------------------------------------------------------------

class MergeReviewDetailDialog(QDialog):
    """Combined detail/compare modal for one pending auto-merge suggestion.

    *row* is a :class:`app.services.unknown_merge_service.PendingAutoMerge`.
    """

    def __init__(self, row, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._row = row
        self.setWindowTitle(t("amerge_detail_title", name=row.person_name or "?"))
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)
        self.resize(1100, 720)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_original_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([650, 450])
        layout.addWidget(splitter, stretch=1)

        close_btn = QPushButton(t("close"))
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignRight)

    # -- left: original image -------------------------------------------------

    def _build_original_panel(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(self._section_title(t("amerge_original_image")))

        pixmap = None
        if self._row.image_path:
            other = get_other_bboxes(self._row.image_path, self._row.face_id)
            pixmap = _build_annotated_pixmap(
                self._row.image_path, self._row.bbox, other
            )

        if pixmap is not None and not pixmap.isNull():
            scroll = _WheelZoomScrollArea(pixmap)
            scroll.setWidgetResizable(False)
            lay.addWidget(scroll, stretch=1)
            path_lbl = QLabel(self._row.image_path or "")
            path_lbl.setStyleSheet("color:#888; font-size:10px;")
            path_lbl.setWordWrap(True)
            lay.addWidget(path_lbl)
        else:
            # Friendly fallback: the largest available crop + a clear notice.
            lay.addWidget(self._fallback_crop(), stretch=1)
            warn = QLabel("⚠ " + t("amerge_full_image_missing"))
            warn.setStyleSheet("color:#e0a060; font-size:11px;")
            warn.setWordWrap(True)
            lay.addWidget(warn)
        return panel

    def _fallback_crop(self) -> QWidget:
        lbl = QLabel()
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("background:#1a1a1a; border:1px solid #333;")
        pix = self._load_pixmap(self._row.crop_path)
        if pix is not None:
            lbl.setPixmap(
                pix.scaled(420, 420, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        else:
            lbl.setText("?")
            lbl.setStyleSheet("background:#1a1a1a; color:#777; font-size:40px;")
        return lbl

    # -- right: suggested face + explanation + compare ------------------------

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        # Suggested face crop.
        lay.addWidget(self._section_title(t("amerge_suggested_face")))
        crop = QLabel()
        crop.setFixedSize(_SUGGESTED_CROP, _SUGGESTED_CROP)
        crop.setAlignment(Qt.AlignCenter)
        crop.setStyleSheet("background:#1a1a1a; border:1px solid #3a7a3a;")
        pix = self._load_pixmap(self._row.crop_path)
        if pix is not None:
            crop.setPixmap(
                pix.scaled(
                    _SUGGESTED_CROP, _SUGGESTED_CROP,
                    Qt.KeepAspectRatio, Qt.SmoothTransformation,
                )
            )
        else:
            crop.setText("?")
        lay.addWidget(crop, alignment=Qt.AlignLeft)

        # "What was this based on?" explanation.
        why = QLabel(self._explanation_html())
        why.setWordWrap(True)
        why.setTextInteractionFlags(Qt.TextSelectableByMouse)
        why.setStyleSheet(
            "background:#1c1c1c; border:1px solid #333; border-radius:4px;"
            "padding:8px; color:#ddd; font-size:11px;"
        )
        lay.addWidget(why)

        graph_btn = QPushButton("📊 " + t("amerge_graph_btn"))
        graph_btn.clicked.connect(self._open_graph)
        lay.addWidget(graph_btn, alignment=Qt.AlignLeft)

        # Target's confirmed reference faces.
        lay.addWidget(self._section_title(t("amerge_target_known")))
        if self._row.person_id is not None:
            gallery = _make_gallery_scroll(
                self._row.person_id, _COMPARE_COLS, panel, exclude_pending=True
            )
            lay.addWidget(gallery, stretch=1)
        else:
            lay.addStretch(1)
        return panel

    # -- helpers --------------------------------------------------------------

    def _explanation_html(self) -> str:
        row = self._row
        decision = row.decision or {}
        lines: List[str] = [f"<b>{t('amerge_why_title')}</b>"]
        lines.append(f"{t('amerge_why_target')}: <b>{row.person_name or '?'}</b>")
        lines.append(f"{t('amerge_why_source')}: {row.source_person_name}")

        best = row.confidence if row.confidence is not None else decision.get("best_score")
        if isinstance(best, (int, float)):
            lines.append(f"{t('amerge_score')}: <b>{best:.0%}</b>")
        if row.image_path:
            lines.append(f"{t('amerge_source_image')}: {Path(row.image_path).name}")
        if row.assigned_at is not None:
            lines.append(
                f"{t('amerge_when')}: {row.assigned_at.strftime('%Y-%m-%d %H:%M')}"
            )
        rule = decision.get("rule")
        if rule:
            thr = decision.get("threshold")
            mar = decision.get("min_margin")
            extra = ""
            if isinstance(thr, (int, float)) and isinstance(mar, (int, float)):
                extra = f" ({t('amerge_threshold')} {thr:.2f}, {t('amerge_margin')} {mar:.2f})"
            lines.append(f"{t('amerge_why_rule')}: {t('amerge_rule_scatter')}{extra}")
        return "<br>".join(lines)

    def _open_graph(self) -> None:
        MergeDecisionGraphDialog(self._row.decision, self).exec()

    @staticmethod
    def _section_title(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("font-weight:bold; color:#eee; font-size:12px;")
        return lbl

    @staticmethod
    def _load_pixmap(path: Optional[str]) -> Optional[QPixmap]:
        if not path or not Path(path).exists():
            return None
        pix = QPixmap(path)
        return None if pix.isNull() else pix
