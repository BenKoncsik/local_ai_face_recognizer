"""Deoldified on-image compare divider in the image browser preview."""

from __future__ import annotations

import numpy as np
import pytest

from app.db.database import init_db
from app.ui.panels.image_browser_panel import ImageBrowserPanel
from app.utils.image_utils import save_image_bgr


@pytest.fixture()
def db(tmp_path):
    init_db(tmp_path / "browser.db")


def test_compare_composite_left_bw_right_color(db, qtbot):
    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)

    h, w = 10, 20
    panel._deol_bw_bgr = np.zeros((h, w, 3), dtype=np.uint8)        # black
    panel._deol_color_bgr = np.full((h, w, 3), 255, dtype=np.uint8)  # white

    panel._deol_split = 50
    out = panel._deol_composite()
    assert out is not None
    assert (out[:, :10] == 0).all(), "left half must be the B&W side"
    assert (out[:, 10:] == 255).all(), "right half must be the color side"

    panel._deol_split = 0
    assert (panel._deol_composite() == 255).all(), "split=0 → all color"

    panel._deol_split = 100
    assert (panel._deol_composite() == 0).all(), "split=100 → all B&W"


def test_ensure_pair_bgr_resizes_color_to_bw_shape(db, qtbot, tmp_path):
    bw_path = tmp_path / "bw.jpg"
    color_path = tmp_path / "color.jpg"
    save_image_bgr(bw_path, np.zeros((10, 20, 3), dtype=np.uint8))
    # Different resolution on the colorized side — must be resized to match.
    save_image_bgr(color_path, np.full((20, 40, 3), 200, dtype=np.uint8))

    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)
    panel._deol_pair_orig_id = None
    panel._current_path = str(bw_path)
    panel._deol_pair_color_path = str(color_path)

    assert panel._deol_ensure_pair_bgr() is True
    assert panel._deol_bw_bgr.shape[:2] == (10, 20)
    assert panel._deol_color_bgr.shape[:2] == (10, 20)


def test_clear_for_new_image_keeps_remembered_mode_and_split(db, qtbot):
    """Switching images drops cached pixels but remembers the chosen mode."""
    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)

    panel._deol_mode = "compare"
    panel._deol_split = 30
    panel._deol_compare = True
    panel._deol_bw_bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    panel._deol_color_bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    panel._btn_view_compare.setChecked(True)

    panel._deol_clear_for_new_image()

    # Per-image state is cleared …
    assert panel._deol_compare is False
    assert panel._deol_bw_bgr is None
    assert panel._deol_color_bgr is None
    assert panel._btn_view_compare.isChecked() is False
    # … but the remembered choice persists for the next image.
    assert panel._deol_mode == "compare"
    assert panel._deol_split == 30


def test_compare_dragged_updates_split_and_recomposites(db, qtbot):
    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)

    panel._deol_bw_bgr = np.zeros((10, 20, 3), dtype=np.uint8)
    panel._deol_color_bgr = np.full((10, 20, 3), 255, dtype=np.uint8)
    panel._deol_compare = True
    panel._orig_img_bgr = panel._deol_color_bgr.copy()

    panel._on_compare_dragged(25)
    assert panel._deol_split == 25
    # left quarter (5 px) is B&W, the rest color
    assert (panel._orig_img_bgr[:, :5] == 0).all()
    assert (panel._orig_img_bgr[:, 5:] == 255).all()


def test_drag_while_not_in_compare_only_records_split(db, qtbot):
    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)
    panel._deol_compare = False
    panel._on_compare_dragged(72)
    assert panel._deol_split == 72


def test_label_compare_divider_x_requires_pixmap(db, qtbot):
    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)
    label = panel._image_label
    label.set_compare_mode(True, 50)
    # No source pixmap yet → no divider position.
    assert label._compare_divider_x() is None
    assert label._compare is True
    label.set_compare_mode(False)
    assert label._compare is False
