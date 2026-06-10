"""Deoldified on-image compare divider in the image browser preview."""

from __future__ import annotations

import numpy as np
import pytest

from app.db.database import init_db
from app.services.deoldified_pairing_service import ComparisonMember
from app.ui.panels.image_browser_panel import ImageBrowserPanel
from app.utils.image_utils import save_image_bgr


@pytest.fixture()
def db(tmp_path):
    init_db(tmp_path / "browser.db")


def test_compare_composite_left_right(db, qtbot):
    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)

    h, w = 10, 20
    panel._deol_left_bgr = np.zeros((h, w, 3), dtype=np.uint8)        # black
    panel._deol_right_bgr = np.full((h, w, 3), 255, dtype=np.uint8)   # white

    panel._deol_split = 50
    out = panel._deol_composite()
    assert out is not None
    assert (out[:, :10] == 0).all(), "left half must be the left side"
    assert (out[:, 10:] == 255).all(), "right half must be the right side"

    panel._deol_split = 0
    assert (panel._deol_composite() == 255).all(), "split=0 → all right side"

    panel._deol_split = 100
    assert (panel._deol_composite() == 0).all(), "split=100 → all left side"


def test_ensure_compare_bgr_resizes_right_to_left_shape(db, qtbot, tmp_path):
    left_path = tmp_path / "bw.jpg"
    right_path = tmp_path / "color.jpg"
    save_image_bgr(left_path, np.zeros((10, 20, 3), dtype=np.uint8))
    # Different resolution on the right side — must be resized to match left.
    save_image_bgr(right_path, np.full((20, 40, 3), 200, dtype=np.uint8))

    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)
    panel._deol_group = [
        ComparisonMember(1, str(left_path), "", True),
        ComparisonMember(2, str(right_path), "(artistic)", False),
    ]
    panel._deol_left_idx = 0
    panel._deol_right_idx = 1

    assert panel._deol_ensure_compare_bgr() is True
    assert panel._deol_left_bgr.shape[:2] == (10, 20)
    assert panel._deol_right_bgr.shape[:2] == (10, 20)


def test_picker_change_recomposes_with_chosen_variant(db, qtbot, tmp_path):
    """Selecting another variant on a side recomposes from that variant's pixels."""
    bw_path = tmp_path / "bw.jpg"
    artistic = tmp_path / "art.jpg"
    stable = tmp_path / "stable.jpg"
    save_image_bgr(bw_path, np.zeros((10, 20, 3), dtype=np.uint8))          # black
    save_image_bgr(artistic, np.full((10, 20, 3), 128, dtype=np.uint8))    # grey
    save_image_bgr(stable, np.full((10, 20, 3), 255, dtype=np.uint8))      # white

    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)
    panel._deol_group = [
        ComparisonMember(1, str(bw_path), "", True),
        ComparisonMember(2, str(artistic), "(artistic)", False),
        ComparisonMember(3, str(stable), "(stable)", False),
    ]
    panel._deol_left_idx, panel._deol_right_idx = 0, 1
    panel._deol_compare = True
    panel._deol_split = 0  # show the whole right side

    assert panel._deol_ensure_compare_bgr() is True
    panel._orig_img_bgr = panel._deol_composite()
    assert (panel._orig_img_bgr == 128).all(), "right side = artistic (grey)"

    # Switch the right picker to the 'stable' (white) variant.
    panel._on_deol_right_changed(2)
    assert panel._deol_right_idx == 2
    assert (panel._orig_img_bgr == 255).all(), "right side now = stable (white)"

    # Two colorized variants can be compared with each other.
    panel._deol_split = 100  # show the whole left side
    panel._on_deol_left_changed(1)  # left = artistic (grey)
    assert panel._deol_left_idx == 1
    assert (panel._orig_img_bgr == 128).all(), "left side = artistic (grey)"


def test_combos_visible_only_in_compare_with_three_members(db, qtbot, tmp_path):
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    c = tmp_path / "c.jpg"
    for p in (a, b, c):
        save_image_bgr(p, np.zeros((8, 8, 3), dtype=np.uint8))

    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)
    panel._deol_group = [
        ComparisonMember(1, str(a), "", True),
        ComparisonMember(2, str(b), "(artistic)", False),
        ComparisonMember(3, str(c), "(stable)", False),
    ]
    panel._populate_deol_combos()
    bar = panel._deoldified_bar  # combos live inside the (otherwise hidden) bar

    # Hidden outside compare mode.
    panel._deol_compare = False
    panel._deol_update_combo_visibility()
    assert panel._deol_left_combo.isVisibleTo(bar) is False

    # Visible in compare mode with 3 members.
    panel._deol_compare = True
    panel._deol_update_combo_visibility()
    assert panel._deol_left_combo.isVisibleTo(bar) is True
    assert panel._deol_left_combo.count() == 3

    # Two members → still hidden even in compare mode (simple 2-image case).
    panel._deol_group = panel._deol_group[:2]
    panel._deol_update_combo_visibility()
    assert panel._deol_left_combo.isVisibleTo(bar) is False


def test_clear_for_new_image_keeps_remembered_mode_and_split(db, qtbot):
    """Switching images drops cached pixels but remembers the chosen mode."""
    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)

    panel._deol_mode = "compare"
    panel._deol_split = 30
    panel._deol_compare = True
    panel._deol_left_bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    panel._deol_right_bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    panel._deol_group = [ComparisonMember(1, "x", "", True)]
    panel._btn_view_compare.setChecked(True)

    panel._deol_clear_for_new_image()

    # Per-image state is cleared …
    assert panel._deol_compare is False
    assert panel._deol_left_bgr is None
    assert panel._deol_right_bgr is None
    assert panel._deol_group == []
    assert panel._btn_view_compare.isChecked() is False
    # … but the remembered choice persists for the next image.
    assert panel._deol_mode == "compare"
    assert panel._deol_split == 30


def test_compare_dragged_updates_split_and_recomposites(db, qtbot):
    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)

    panel._deol_left_bgr = np.zeros((10, 20, 3), dtype=np.uint8)
    panel._deol_right_bgr = np.full((10, 20, 3), 255, dtype=np.uint8)
    panel._deol_compare = True
    panel._orig_img_bgr = panel._deol_right_bgr.copy()

    panel._on_compare_dragged(25)
    assert panel._deol_split == 25
    # left quarter (5 px) is the left side, the rest right side
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
