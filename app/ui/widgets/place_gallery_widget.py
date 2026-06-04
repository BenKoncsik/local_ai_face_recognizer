"""Image gallery widget for a Place: large preview + scrollable thumbnail strip."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, QThreadPool, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.ui.i18n import t
from app.workers.thumbnail_worker import ThumbnailRunnable

log = logging.getLogger(__name__)

_THUMB_SIZE = 80
_PREVIEW_MAX_H = 260
_PREVIEW_MAX_W = 420
_LARGE_FULL_MAX = 1400


class _ThumbLabel(QLabel):
    """Clickable thumbnail with selection highlight."""

    clicked = Signal(str)   # file_path

    def __init__(self, file_path: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._path = file_path
        self.setFixedSize(_THUMB_SIZE, _THUMB_SIZE)
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.PointingHandCursor)
        self.setText("…")
        self.setStyleSheet(self._style(False))

    @staticmethod
    def _style(selected: bool) -> str:
        border = "#89B4FA" if selected else "#45475A"
        return (
            f"border:2px solid {border};background:#181825;"
            "color:#6C7086;font-size:10px;"
        )

    def set_selected(self, selected: bool) -> None:
        self.setStyleSheet(self._style(selected))

    def set_image(self, img: QImage) -> None:
        pix = QPixmap.fromImage(img)
        self.setPixmap(
            pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self.setText("")

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self._path)
        super().mousePressEvent(event)

    @property
    def path(self) -> str:
        return self._path


class _PreviewLabel(QLabel):
    """Large preview label that emits doubleClicked when double-clicked."""

    doubleClicked = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: ANN001
        if event.button() == Qt.LeftButton:
            self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)


class _ClickableLabel(QLabel):
    """Label that emits clicked on left mouse press — used in full-size dialog."""

    clicked = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class _FullSizeDialog(QDialog):
    """Modal dialog showing a full-resolution image, click or Esc to close."""

    def __init__(self, file_path: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(Path(file_path).name)
        self.setModal(True)

        screen = self.screen()
        if screen is not None:
            avail = screen.availableGeometry()
            max_w = int(avail.width() * 0.9)
            max_h = int(avail.height() * 0.9)
        else:
            max_w, max_h = _LARGE_FULL_MAX, _LARGE_FULL_MAX

        label = _ClickableLabel()
        label.setAlignment(Qt.AlignCenter)
        label.setCursor(Qt.PointingHandCursor)
        label.setToolTip(t("places_gallery_click_to_close"))
        label.clicked.connect(self.accept)

        pix = QPixmap(file_path)
        if not pix.isNull():
            pix = pix.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pix)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(label)

        self.adjustSize()

    def keyPressEvent(self, event) -> None:  # noqa: ANN001
        if event.key() == Qt.Key_Escape:
            self.accept()
        super().keyPressEvent(event)


class PlaceGalleryWidget(QWidget):
    """Visual gallery: large preview + scrollable thumbnail strip below."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._paths: List[str] = []
        self._thumb_labels: Dict[str, _ThumbLabel] = {}  # path → label
        self._current_path: Optional[str] = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Large preview — uses _PreviewLabel subclass to properly override mouseDoubleClickEvent
        self._preview = _PreviewLabel()
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setFixedHeight(_PREVIEW_MAX_H)
        self._preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._preview.setStyleSheet("background:#181825;color:#6C7086;border-radius:4px;")
        self._preview.setToolTip(t("places_gallery_dbl_click"))
        self._preview.doubleClicked.connect(self._on_preview_dbl_click)
        layout.addWidget(self._preview)

        # Thumbnail strip
        scroll = QScrollArea()
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedHeight(_THUMB_SIZE + 18)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background:#181825;border:none;")

        self._thumb_container = QWidget()
        self._thumb_layout = QHBoxLayout(self._thumb_container)
        self._thumb_layout.setContentsMargins(4, 4, 4, 4)
        self._thumb_layout.setSpacing(4)
        self._thumb_layout.addStretch()
        scroll.setWidget(self._thumb_container)
        layout.addWidget(scroll)

        self._clear_preview()

    def set_images(self, paths: List[str]) -> None:
        """Replace the gallery with images at *paths* (absolute file paths)."""
        self._cancel_pending()
        self._paths = list(paths)
        self._thumb_labels.clear()
        self._current_path = None

        # Clear thumb layout (keep the trailing stretch)
        while self._thumb_layout.count() > 1:
            item = self._thumb_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

        if not paths:
            self._clear_preview()
            return

        for path in paths:
            lbl = _ThumbLabel(path)
            lbl.clicked.connect(self._on_thumb_click)
            self._thumb_layout.insertWidget(self._thumb_layout.count() - 1, lbl)
            self._thumb_labels[path] = lbl

            key = path
            worker = ThumbnailRunnable(path, key, size=_THUMB_SIZE)
            worker.signals.ready.connect(self._on_thumb_ready)
            worker.signals.failed.connect(self._on_thumb_failed)
            QThreadPool.globalInstance().start(worker)

        # Select first image immediately (load preview async too)
        self._select_path(paths[0])

    def _select_path(self, path: str) -> None:
        if self._current_path and self._current_path in self._thumb_labels:
            self._thumb_labels[self._current_path].set_selected(False)
        self._current_path = path
        if path in self._thumb_labels:
            self._thumb_labels[path].set_selected(True)
        self._load_preview(path)

    def _load_preview(self, path: str) -> None:
        pix = QPixmap(path)
        if pix.isNull():
            self._clear_preview()
            return
        pix = pix.scaled(
            _PREVIEW_MAX_W, _PREVIEW_MAX_H,
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self._preview.setPixmap(pix)
        self._preview.setText("")

    def _clear_preview(self) -> None:
        self._preview.setPixmap(QPixmap())
        self._preview.setText(t("places_gallery_no_images"))

    def _on_thumb_click(self, path: str) -> None:
        self._select_path(path)

    def _on_preview_dbl_click(self) -> None:
        if self._current_path and Path(self._current_path).exists():
            dlg = _FullSizeDialog(self._current_path, self)
            dlg.exec()

    def _on_thumb_ready(self, key: str, img: QImage) -> None:
        lbl = self._thumb_labels.get(key)
        if lbl is not None:
            lbl.set_image(img)

    def _on_thumb_failed(self, key: str) -> None:
        lbl = self._thumb_labels.get(key)
        if lbl is not None:
            lbl.setText("✗")

    def _cancel_pending(self) -> None:
        # ThumbnailRunnable workers are fire-and-forget; stale results are
        # ignored because _thumb_labels is rebuilt on every set_images() call.
        pass

    def clear(self) -> None:
        self.set_images([])
