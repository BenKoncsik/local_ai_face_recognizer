"""Smoke tests for the missing-image review dialog."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QCheckBox, QComboBox

from app.services.image_path_matcher import (
    Candidate,
    MatchProposal,
    MatchReport,
    MissingImage,
)
from app.ui.dialogs.image_path_repair_dialog import (
    _COL_FILE,
    _COL_MATCH,
    _COL_USE,
    ImagePathRepairDialog,
)


def _proposal(image_id: int, *, hash_match=None, extra_candidate=False):
    missing = MissingImage(
        image_id=image_id,
        file_path=f"/old/root/sub/photo{image_id}.jpg",
        relative_path=f"sub/photo{image_id}.jpg",
        file_hash="abc" if hash_match is not None else None,
        resolved_path=f"/new/root/sub/photo{image_id}.jpg",
    )
    candidates = [
        Candidate(
            path=f"/found/sub/photo{image_id}.jpg",
            score=1.0 if hash_match else 0.55,
            depth=2,
            hash_match=hash_match,
        )
    ]
    if extra_candidate:
        candidates.append(
            Candidate(path=f"/other/photo{image_id}.jpg", score=0.4, depth=1)
        )
    return MatchProposal(missing=missing, candidates=candidates)


@pytest.fixture()
def dialog(qtbot):
    dlg = ImagePathRepairDialog(search_roots=["/some/folder"])
    qtbot.addWidget(dlg)
    return dlg


def test_folders_are_prefilled_and_deduplicated(dialog):
    dialog._add_folder("/some/folder")
    dialog._add_folder("/another")
    assert dialog._search_roots() == ["/some/folder", "/another"]


def test_confident_rows_are_pre_checked_and_others_are_not(dialog):
    report = MatchReport(
        proposals=[
            _proposal(1, hash_match=True),                      # proof
            _proposal(2, hash_match=False, extra_candidate=True),  # demoted
            MatchProposal(missing=_proposal(3).missing),           # no candidate
        ],
        missing_total=3,
        scanned_files=10,
    )
    dialog._on_scan_done(report)

    assert dialog._table.rowCount() == 3
    checks = [
        dialog._table.cellWidget(row, _COL_USE).findChild(QCheckBox)
        for row in range(3)
    ]
    assert checks[0].isChecked()
    assert not checks[1].isChecked()
    assert not checks[2].isEnabled()  # nothing to accept
    assert dialog._apply_btn.isEnabled()


def test_decisions_only_include_checked_rows_with_a_candidate(dialog):
    dialog._on_scan_done(
        MatchReport(
            proposals=[_proposal(1, hash_match=True), _proposal(2)],
            missing_total=2,
        )
    )
    # Row 1 (id 2) has a candidate but is not confident — accept it by hand.
    dialog._set_row_checked(1, True)

    decisions = dialog._decisions()
    assert decisions == {
        1: "/found/sub/photo1.jpg",
        2: "/found/sub/photo2.jpg",
    }

    # Skipping a row drops it from the decisions.
    combo = dialog._table.cellWidget(1, _COL_MATCH)
    assert isinstance(combo, QComboBox)
    combo.setCurrentIndex(0)  # "— skip —"
    assert dialog._decisions() == {1: "/found/sub/photo1.jpg"}


def test_select_helpers_toggle_rows(dialog):
    dialog._on_scan_done(
        MatchReport(
            proposals=[_proposal(1, hash_match=True), _proposal(2, hash_match=False)],
            missing_total=2,
        )
    )

    dialog._set_all_checked(confident_only=False)
    assert len(dialog._decisions()) == 2

    dialog._clear_all_checked()
    assert dialog._decisions() == {}

    dialog._set_all_checked(confident_only=True)
    assert list(dialog._decisions()) == [1]


def test_rows_carry_their_image_id(dialog):
    dialog._on_scan_done(
        MatchReport(proposals=[_proposal(7, hash_match=True)], missing_total=1)
    )
    from PySide6.QtCore import Qt

    item = dialog._table.item(0, _COL_FILE)
    assert item.data(Qt.UserRole) == 7
    assert dialog._proposal_for_row(0).missing.image_id == 7
