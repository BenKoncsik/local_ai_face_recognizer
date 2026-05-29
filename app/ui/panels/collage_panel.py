"""Collage viewer panel.

Shows a Picasa collage rendered on a zoomable canvas.  Each node's
boundary is drawn as an overlay; clicking a node opens an info popup
with filename, album data, and any recognised persons.

Layout
------
    CollagePanel (QWidget)
    ├── toolbar (QHBoxLayout)
    │   ├── collage selector (QComboBox)
    │   ├── zoom-in / zoom-out / fit buttons
    │   └── export button
    └── QGraphicsView
        └── QGraphicsScene
            ├── QGraphicsPixmapItem  ← rendered collage image
            └── _NodeOverlay items   ← clickable node rectangles
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from PySide6.QtCore import (
    Qt, QRectF, Signal, Slot, QPointF,
)
from PySide6.QtGui import (
    QBrush, QColor, QFont, QPainter, QPen, QPixmap, QImage,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QGraphicsItem,
    QGraphicsRectItem, QGraphicsScene, QGraphicsTextItem,
    QGraphicsView, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QSizePolicy, QToolTip, QVBoxLayout, QWidget,
)

from app.db.database import session_scope
from app.db.models import Collage, CollageNode
from app.services.collage_service import CollageService
from app.ui.i18n import t

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
_BORDER_COLOR = QColor(80, 160, 255, 180)       # node border
_BORDER_HOVER = QColor(255, 200, 60, 220)       # node border on hover
_FACE_COLOR   = QColor(50, 220, 50, 200)        # face bbox
_MISSING_COLOR = QColor(200, 60, 60, 160)       # missing-file overlay
_LABEL_BG     = QColor(0, 0, 0, 160)


# ---------------------------------------------------------------------------
# Clickable node overlay item
# ---------------------------------------------------------------------------

class _NodeOverlayItem(QGraphicsRectItem):
    """Transparent QGraphicsRectItem that highlights on hover and emits a signal
    when clicked by forwarding to the parent panel."""

    def __init__(
        self,
        rect: QRectF,
        node_id: int,
        panel: "CollagePanel",
        missing: bool = False,
    ) -> None:
        super().__init__(rect)
        self._node_id = node_id
        self._panel = panel
        self._missing = missing

        pen = QPen(_MISSING_COLOR if missing else _BORDER_COLOR, 1.5)
        self.setPen(pen)
        self.setBrush(QBrush(Qt.transparent))
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self.setCursor(Qt.PointingHandCursor)

    def hoverEnterEvent(self, event) -> None:
        pen = QPen(_BORDER_HOVER, 2.5)
        self.setPen(pen)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        pen = QPen(_MISSING_COLOR if self._missing else _BORDER_COLOR, 1.5)
        self.setPen(pen)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._panel.node_clicked.emit(self._node_id)
        super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# Face overlay item (small coloured rectangle)
# ---------------------------------------------------------------------------

class _FaceOverlayItem(QGraphicsRectItem):
    def __init__(self, rect: QRectF, person_name: Optional[str]) -> None:
        super().__init__(rect)
        self._person_name = person_name or ""
        pen = QPen(_FACE_COLOR, 1.5)
        self.setPen(pen)
        self.setBrush(QBrush(Qt.transparent))
        self.setToolTip(self._person_name or t("unknown"))
        self.setAcceptHoverEvents(True)


# ---------------------------------------------------------------------------
# Zoomable QGraphicsView
# ---------------------------------------------------------------------------

class _ZoomableView(QGraphicsView):
    def __init__(self, scene: QGraphicsScene, parent=None) -> None:
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self._zoom_factor = 1.0

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        factor = 1.12 if delta > 0 else 1 / 1.12
        self._zoom_factor *= factor
        self._zoom_factor = max(0.05, min(self._zoom_factor, 20.0))
        self.scale(factor, factor)

    def fit(self) -> None:
        self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
        self._zoom_factor = 1.0

    def zoom_in(self) -> None:
        self.scale(1.25, 1.25)
        self._zoom_factor *= 1.25

    def zoom_out(self) -> None:
        self.scale(1 / 1.25, 1 / 1.25)
        self._zoom_factor /= 1.25


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

class CollagePanel(QWidget):
    """Widget that renders a Picasa collage and exposes node-click events."""

    node_clicked = Signal(int)   # node_id

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._current_collage_id: Optional[int] = None
        self._node_map: Dict[int, CollageNode] = {}   # node_id → ORM (detached)
        self._render_w = 0
        self._render_h = 0

        self._scene = QGraphicsScene(self)
        self._view = _ZoomableView(self._scene, self)

        self._build_ui()
        self.node_clicked.connect(self._on_node_clicked)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # --- toolbar ---
        toolbar = QHBoxLayout()

        self._collage_combo = QComboBox()
        self._collage_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._collage_combo.setMinimumWidth(200)
        self._collage_combo.currentIndexChanged.connect(self._on_combo_changed)
        toolbar.addWidget(QLabel(t("collage_label")))
        toolbar.addWidget(self._collage_combo)

        self._fit_btn = QPushButton(f"⊡ {t('fit')}")
        self._fit_btn.clicked.connect(self._view.fit)
        self._fit_btn.setFixedWidth(110)
        toolbar.addWidget(self._fit_btn)

        self._zoom_in_btn = QPushButton("🔍+")
        self._zoom_in_btn.clicked.connect(self._view.zoom_in)
        self._zoom_in_btn.setFixedWidth(46)
        toolbar.addWidget(self._zoom_in_btn)

        self._zoom_out_btn = QPushButton("🔍−")
        self._zoom_out_btn.clicked.connect(self._view.zoom_out)
        self._zoom_out_btn.setFixedWidth(46)
        toolbar.addWidget(self._zoom_out_btn)

        self._faces_btn = QPushButton(f"👤 {t('faces')}")
        self._faces_btn.setCheckable(True)
        self._faces_btn.setChecked(True)
        self._faces_btn.clicked.connect(self._reload_current)
        self._faces_btn.setFixedWidth(90)
        toolbar.addWidget(self._faces_btn)

        self._export_btn = QPushButton(f"📤 {t('export')}")
        self._export_btn.clicked.connect(self._on_export)
        self._export_btn.setFixedWidth(90)
        toolbar.addWidget(self._export_btn)

        layout.addLayout(toolbar)
        layout.addWidget(self._view, 1)

        # --- info bar ---
        self._info_label = QLabel(t("select_collage"))
        self._info_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._info_label)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh_collage_list(self) -> None:
        """Reload the combo box from the database."""
        prev_id = self._current_collage_id
        self._collage_combo.blockSignals(True)
        self._collage_combo.clear()

        with session_scope() as session:
            svc = CollageService(session)
            collages = svc.list_collages()
            items = []
            for c in collages:
                label = c.album_title or f"{t('tab_collage')} #{c.id}"
                if c.album_date:
                    label += f"  ({c.album_date})"
                items.append((label, c.id))

        for label, cid in items:
            self._collage_combo.addItem(label, cid)

        self._collage_combo.blockSignals(False)

        # Restore selection
        if prev_id is not None:
            for i in range(self._collage_combo.count()):
                if self._collage_combo.itemData(i) == prev_id:
                    self._collage_combo.setCurrentIndex(i)
                    return
        if self._collage_combo.count() > 0:
            self._collage_combo.setCurrentIndex(0)
            self._load_collage(self._collage_combo.itemData(0))

    def show_collage(self, collage_id: int) -> None:
        """Display a specific collage by id."""
        for i in range(self._collage_combo.count()):
            if self._collage_combo.itemData(i) == collage_id:
                self._collage_combo.setCurrentIndex(i)
                return
        self._load_collage(collage_id)

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    @Slot(int)
    def _on_combo_changed(self, index: int) -> None:
        if index < 0:
            return
        collage_id = self._collage_combo.itemData(index)
        if collage_id is not None:
            self._load_collage(collage_id)

    def _reload_current(self) -> None:
        if self._current_collage_id is not None:
            self._load_collage(self._current_collage_id)

    def _load_collage(self, collage_id: int) -> None:
        self._current_collage_id = collage_id
        self._scene.clear()
        self._node_map.clear()

        draw_faces = self._faces_btn.isChecked()

        with session_scope() as session:
            svc = CollageService(session)
            collage = svc.get_collage(collage_id)
            if collage is None:
                self._info_label.setText(t("collage_not_found"))
                return

            # Determine canvas size
            cw = collage.format_width or 2858
            ch = collage.format_height or 1000

            # Render to at most 2000px wide for performance
            max_w = 2000
            scale = min(max_w / cw, 1.0)
            render_w = int(cw * scale)
            render_h = int(ch * scale)
            self._render_w = render_w
            self._render_h = render_h

            # Render image
            canvas = svc.render_collage_image(
                collage, render_h=render_h,
                draw_borders=False,   # we draw borders as overlay items
                draw_faces=False,     # we draw face overlays as items
            )

            # Detach nodes for use outside session
            nodes_data = []
            for node in collage.nodes:
                nodes_data.append({
                    "id": node.id,
                    "rel_x": node.rel_x, "rel_y": node.rel_y,
                    "rel_w": node.rel_w, "rel_h": node.rel_h,
                    "src_missing": node.src_missing,
                    "src_raw": node.src_raw,
                    "src_resolved": node.src_resolved,
                    "node_uid": node.node_uid,
                    "theme": node.theme,
                    "image_id": node.image_id,
                })

            face_data = svc.projected_faces(collage, render_w, render_h) if draw_faces else []

            info_txt = (
                f"{collage.album_title or t('untitled')}  |  "
                f"{collage.album_date or ''}  |  "
                f"{t('items_count', n=len(collage.nodes))}"
            )

        # Build scene outside of session
        if canvas is not None:
            import numpy as np
            rgb = cv2_to_qimage(canvas)
            pixmap = QPixmap.fromImage(rgb)
            self._scene.addPixmap(pixmap)
            self._scene.setSceneRect(QRectF(0, 0, render_w, render_h))

        # Node overlays
        for nd in nodes_data:
            px = nd["rel_x"] * render_w
            py = nd["rel_y"] * render_h
            pw = nd["rel_w"] * render_w
            ph = nd["rel_h"] * render_h
            rect = QRectF(px, py, pw, ph)
            item = _NodeOverlayItem(rect, nd["id"], self, missing=nd["src_missing"])
            self._scene.addItem(item)
            self._node_map[nd["id"]] = nd  # store dict (no ORM)

        # Face overlays
        for fd in face_data:
            fx, fy, fw, fh = fd["bbox_collage"]
            face_rect = QRectF(fx, fy, fw, fh)
            face_item = _FaceOverlayItem(face_rect, fd.get("person_name"))
            self._scene.addItem(face_item)

        self._info_label.setText(info_txt)
        self._view.fit()

    # ------------------------------------------------------------------
    # Node click handler
    # ------------------------------------------------------------------

    @Slot(int)
    def _on_node_clicked(self, node_id: int) -> None:
        nd = self._node_map.get(node_id)
        if nd is None:
            return

        with session_scope() as session:
            svc = CollageService(session)
            node = session.get(CollageNode, node_id)
            if node is None:
                return

            collage = node.collage
            faces = svc.get_faces_for_node(node)

            person_names = []
            for face in faces:
                if face.person_id:
                    from app.db.models import Person
                    p = session.get(Person, face.person_id)
                    if p:
                        person_names.append(p.name)

        # Build info text
        lines = []
        src_name = ""
        if nd.get("src_raw"):
            from pathlib import Path as _P
            src_name = _P(nd["src_raw"].replace("\\", "/")).name
        lines.append(f"<b>{t('filename')}:</b> {src_name or '—'}")
        if nd.get("src_resolved"):
            lines.append(f"<b>{t('path')}:</b> {nd['src_resolved']}")
        elif nd.get("src_raw"):
            lines.append(f"<b>{t('raw_path')}:</b> {nd['src_raw']}")
        if nd.get("src_missing"):
            lines.append(f"<b style='color:#e57373;'>{t('source_file_missing')}</b>")
        lines.append(f"<b>Node UID:</b> {nd.get('node_uid') or '—'}")
        if person_names:
            lines.append(f"<b>{t('persons')}:</b> {', '.join(person_names)}")
        else:
            lines.append(f"<b>{t('persons')}:</b> {t('no_recognized_face')}")

        from app.ui.dialogs.collage_node_dialog import CollageNodeDialog
        dlg = CollageNodeDialog(node_id=node_id, info_lines=lines, parent=self)
        dlg.metadata_saved.connect(self._reload_current)
        dlg.exec()

    # ------------------------------------------------------------------
    # Export slot
    # ------------------------------------------------------------------

    @Slot()
    def _on_export(self) -> None:
        if self._current_collage_id is None:
            QMessageBox.warning(self, t("export"), t("no_collage_selected"))
            return

        from app.app_settings import app_qsettings
        settings = app_qsettings()
        start_dir = settings.value("paths/last_export", "", type=str)
        target = QFileDialog.getExistingDirectory(
            self, t("select_export_folder"), start_dir
        )
        if not target:
            return
        settings.setValue("paths/last_export", target)

        try:
            with session_scope() as session:
                svc = CollageService(session)
                collage = svc.get_collage(self._current_collage_id)
                if collage is None:
                    return
                # Load nodes within session
                _ = list(collage.nodes)
                out = svc.export_annotated_collage(collage, target)

            QMessageBox.information(
                self, t("export"),
                t("collage_exported", path=out)
            )
        except Exception as exc:
            log.exception("Collage export failed")
            QMessageBox.critical(self, t("export_error"), str(exc))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def cv2_to_qimage(bgr_array) -> QImage:
    """Convert a BGR numpy array to QImage (RGB888)."""
    import numpy as np
    rgb = bgr_array[:, :, ::-1].copy()
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
