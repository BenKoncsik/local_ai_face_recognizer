"""Tests for UnknownMergeService — reviewable auto-merge from Unknown clusters."""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.config import RecognitionConfig
from app.db.database import init_db, session_scope
from app.db.models import Face, Image, Person
from app.services.identity_service import IdentityService
from app.services.unknown_merge_service import (
    REVIEW_PENDING,
    SOURCE_AUTO_MERGE,
    UnknownMergeService,
)


@pytest.fixture()
def db(tmp_path):
    init_db(tmp_path / "umerge.db")
    return tmp_path


def _img(session, path: str) -> Image:
    image = Image(file_path=path, file_hash=f"h_{path}", file_mtime=0.0)
    session.add(image)
    session.flush()
    return image


def _person(session, name: str, *, is_auto_named=False, is_protected=False) -> Person:
    p = Person(name=name, is_auto_named=is_auto_named, is_protected=is_protected)
    session.add(p)
    session.flush()
    return p


def _face(session, image, person, *, embedding=None, source=None) -> Face:
    f = Face(
        image_id=image.id,
        person_id=person.id,
        bbox_x=1, bbox_y=2, bbox_w=3, bbox_h=4,
        confidence=0.9,
        detector_backend="cpu",
        assignment_source=source,
    )
    if embedding is not None:
        f.set_embedding(np.asarray(embedding, dtype=np.float32))
    session.add(f)
    session.flush()
    return f


# 8-dim one-hot embeddings: E_A and E_B are orthogonal (cosine 0).
E_A = [1, 0, 0, 0, 0, 0, 0, 0]
E_B = [0, 1, 0, 0, 0, 0, 0, 0]


# ---------------------------------------------------------------------------
# 1. Scatter: siblings move along, selected stays a plain manual assignment
# ---------------------------------------------------------------------------

def test_scatter_moves_siblings_keeps_selected_manual(db):
    with session_scope() as session:
        img = _img(session, "/tmp/a.jpg")
        unknown = _person(session, "Unknown 7", is_auto_named=True)
        target = _person(session, "Anikó")
        faces = [_face(session, img, unknown) for _ in range(3)]  # no embeddings
        selected_id = faces[0].id
        target_id = target.id

    with session_scope() as session:
        result = UnknownMergeService(session).assign_unknown_face(
            selected_id, target_id
        )

    assert result.was_unknown_cluster
    assert result.n_moved_siblings == 2

    with session_scope() as session:
        # All three now belong to the target.
        assert session.query(Face).filter(Face.person_id == target_id).count() == 3
        selected = session.get(Face, selected_id)
        assert selected.assignment_source == "manual"
        assert not selected.auto_merged_from_unknown
        assert selected.auto_merge_review_status is None
        # Siblings are flagged pending.
        siblings = (
            session.query(Face)
            .filter(Face.id != selected_id, Face.person_id == target_id)
            .all()
        )
        assert len(siblings) == 2
        for s in siblings:
            assert s.auto_merged_from_unknown
            assert s.auto_merge_review_status == REVIEW_PENDING
            assert s.assignment_source == SOURCE_AUTO_MERGE
            assert s.auto_merge_confirmed_by_user is False


# ---------------------------------------------------------------------------
# 2. source_person_id recorded; emptied Unknown cleaned up
# ---------------------------------------------------------------------------

def test_pending_records_source_and_cleans_unknown(db):
    with session_scope() as session:
        img = _img(session, "/tmp/a.jpg")
        unknown = _person(session, "Unknown 9", is_auto_named=True)
        target = _person(session, "Béla")
        faces = [_face(session, img, unknown) for _ in range(2)]
        selected_id, target_id, unknown_id = faces[0].id, target.id, unknown.id

    with session_scope() as session:
        UnknownMergeService(session).assign_unknown_face(selected_id, target_id)

    with session_scope() as session:
        # Source cluster drained → deleted.
        assert session.get(Person, unknown_id) is None
        sibling = (
            session.query(Face)
            .filter(Face.id != selected_id, Face.person_id == target_id)
            .one()
        )
        assert sibling.auto_merge_source_person_id == unknown_id


# ---------------------------------------------------------------------------
# 3. Non-Unknown source → no scatter (plain reassign)
# ---------------------------------------------------------------------------

def test_named_source_does_not_scatter(db):
    with session_scope() as session:
        img = _img(session, "/tmp/a.jpg")
        roni = _person(session, "Roni")          # named, not Unknown
        target = _person(session, "Anikó")
        f0 = _face(session, img, roni, source="manual")
        f1 = _face(session, img, roni, source="manual")
        sel_id, roni_id, target_id = f0.id, roni.id, target.id

    with session_scope() as session:
        result = UnknownMergeService(session).assign_unknown_face(sel_id, target_id)

    assert not result.was_unknown_cluster
    assert result.n_moved_siblings == 0
    with session_scope() as session:
        # Only the selected face moved; the sibling stays on Roni unflagged.
        assert session.query(Face).filter(Face.person_id == target_id).count() == 1
        other = session.query(Face).filter(Face.person_id == roni_id).one()
        assert not other.auto_merged_from_unknown


def test_protected_catchall_does_not_scatter(db):
    with session_scope() as session:
        img = _img(session, "/tmp/a.jpg")
        catchall = _person(session, "Ismeretlen", is_auto_named=True, is_protected=True)
        target = _person(session, "Anikó")
        faces = [_face(session, img, catchall) for _ in range(3)]
        sel_id, target_id = faces[0].id, target.id

    with session_scope() as session:
        result = UnknownMergeService(session).assign_unknown_face(sel_id, target_id)

    assert not result.was_unknown_cluster
    with session_scope() as session:
        # Catch-all is never scattered.
        assert session.query(Face).filter(Face.person_id == target_id).count() == 1


# ---------------------------------------------------------------------------
# 4. Intelligent auto-confirm
# ---------------------------------------------------------------------------

def test_auto_confirm_high_similarity_clears_only_matching(db):
    with session_scope() as session:
        img = _img(session, "/tmp/a.jpg")
        unknown = _person(session, "Unknown 3", is_auto_named=True)
        target = _person(session, "Anikó")
        # Reference face already confirmed on the target.
        _face(session, img, target, embedding=E_A, source="manual")
        selected = _face(session, img, unknown, embedding=E_A)
        sib_same = _face(session, img, unknown, embedding=E_A)   # → auto-confirm
        sib_diff = _face(session, img, unknown, embedding=E_B)   # → stays pending
        sel_id, target_id = selected.id, target.id
        same_id, diff_id = sib_same.id, sib_diff.id

    with session_scope() as session:
        result = UnknownMergeService(session).assign_unknown_face(sel_id, target_id)

    assert result.n_auto_confirmed == 1
    assert result.n_pending == 1
    with session_scope() as session:
        same = session.get(Face, same_id)
        assert same.auto_merge_review_status is None
        assert same.auto_merge_confirmed_at is not None
        assert same.auto_merge_confirmed_by_user is False
        diff = session.get(Face, diff_id)
        assert diff.auto_merge_review_status == REVIEW_PENDING


def test_auto_confirm_needs_reference_face(db):
    """Without a confirmed reference on the target, faces stay pending."""
    with session_scope() as session:
        img = _img(session, "/tmp/a.jpg")
        target = _person(session, "Anikó")  # no faces yet
        f1 = _face(session, img, target, embedding=E_A, source=SOURCE_AUTO_MERGE)
        f1.auto_merged_from_unknown = True
        f1.auto_merge_review_status = REVIEW_PENDING
        session.flush()
        fid, target_id = f1.id, target.id

    with session_scope() as session:
        # Target has only the pending face (excluded from profiles) → no reference.
        confirmed = UnknownMergeService(session).maybe_auto_confirm([fid], target_id)

    assert confirmed == 0
    with session_scope() as session:
        assert session.get(Face, fid).auto_merge_review_status == REVIEW_PENDING


def test_auto_confirm_disabled_keeps_pending(db):
    with session_scope() as session:
        img = _img(session, "/tmp/a.jpg")
        unknown = _person(session, "Unknown 1", is_auto_named=True)
        target = _person(session, "Anikó")
        _face(session, img, target, embedding=E_A, source="manual")
        selected = _face(session, img, unknown, embedding=E_A)
        sib = _face(session, img, unknown, embedding=E_A)
        sel_id, target_id, sib_id = selected.id, target.id, sib.id

    cfg = RecognitionConfig(unknown_auto_merge_enabled=False)
    with session_scope() as session:
        result = UnknownMergeService(session, cfg).assign_unknown_face(sel_id, target_id)

    assert result.n_auto_confirmed == 0
    with session_scope() as session:
        assert session.get(Face, sib_id).auto_merge_review_status == REVIEW_PENDING


# ---------------------------------------------------------------------------
# 5. confirm / move clear the markers
# ---------------------------------------------------------------------------

def test_confirm_auto_merge_clears_markers(db):
    with session_scope() as session:
        img = _img(session, "/tmp/a.jpg")
        unknown = _person(session, "Unknown 2", is_auto_named=True)
        target = _person(session, "Anikó")
        faces = [_face(session, img, unknown) for _ in range(2)]
        sel_id, target_id = faces[0].id, target.id

    with session_scope() as session:
        svc = UnknownMergeService(session)
        svc.assign_unknown_face(sel_id, target_id)
        sib_id = svc.pending_face_ids()[0]

    with session_scope() as session:
        UnknownMergeService(session).confirm_auto_merge(sib_id)

    with session_scope() as session:
        f = session.get(Face, sib_id)
        assert not f.auto_merged_from_unknown
        assert f.auto_merge_review_status is None
        assert f.auto_merge_confirmed_by_user is True
        assert f.assignment_source == "manual"


def test_move_auto_merge_clears_markers(db):
    with session_scope() as session:
        img = _img(session, "/tmp/a.jpg")
        unknown = _person(session, "Unknown 4", is_auto_named=True)
        target = _person(session, "Anikó")
        other = _person(session, "Béla")
        faces = [_face(session, img, unknown) for _ in range(2)]
        sel_id, target_id, other_id = faces[0].id, target.id, other.id

    with session_scope() as session:
        svc = UnknownMergeService(session)
        svc.assign_unknown_face(sel_id, target_id)
        sib_id = svc.pending_face_ids()[0]

    with session_scope() as session:
        UnknownMergeService(session).move_auto_merge(sib_id, other_id)

    with session_scope() as session:
        f = session.get(Face, sib_id)
        assert f.person_id == other_id
        assert f.auto_merge_review_status is None
        assert f.assignment_source == "manual"


# ---------------------------------------------------------------------------
# 6. Bulk reassign override clears pending markers
# ---------------------------------------------------------------------------

def test_bulk_reassign_clears_pending(db):
    with session_scope() as session:
        img = _img(session, "/tmp/a.jpg")
        unknown = _person(session, "Unknown 6", is_auto_named=True)
        target = _person(session, "Anikó")
        other = _person(session, "Béla")
        faces = [_face(session, img, unknown) for _ in range(3)]
        sel_id, target_id, other_id = faces[0].id, target.id, other.id

    with session_scope() as session:
        svc = UnknownMergeService(session)
        svc.assign_unknown_face(sel_id, target_id)
        pending_ids = svc.pending_face_ids()

    assert len(pending_ids) == 2
    with session_scope() as session:
        IdentityService(session).reassign_faces_bulk(pending_ids, other_id)

    with session_scope() as session:
        for fid in pending_ids:
            f = session.get(Face, fid)
            assert f.person_id == other_id
            assert not f.auto_merged_from_unknown
            assert f.auto_merge_review_status is None


# ---------------------------------------------------------------------------
# 7. Listing / counting
# ---------------------------------------------------------------------------

def test_list_and_count_pending(db):
    with session_scope() as session:
        img = _img(session, "/tmp/a.jpg")
        unknown = _person(session, "Unknown 8", is_auto_named=True)
        target = _person(session, "Anikó")
        faces = [_face(session, img, unknown) for _ in range(3)]
        sel_id, target_id, unknown_id = faces[0].id, target.id, unknown.id

    with session_scope() as session:
        svc = UnknownMergeService(session)
        svc.assign_unknown_face(sel_id, target_id)

    with session_scope() as session:
        svc = UnknownMergeService(session)
        assert svc.count_pending() == 2
        rows = svc.list_pending()
        assert len(rows) == 2
        for r in rows:
            assert r.person_id == target_id
            assert r.person_name == "Anikó"
            assert r.source_person_id == unknown_id
            # Emptied Unknown is gone → synthetic fallback label.
            assert r.source_person_name == f"Unknown #{unknown_id}"
            assert r.image_path == "/tmp/a.jpg"


# ---------------------------------------------------------------------------
# 8. Decision graph capture (per-suggestion, serializable)
# ---------------------------------------------------------------------------

def test_pending_face_gets_decision_graph(db):
    """Every dragged-along face stores a parseable decision graph."""
    with session_scope() as session:
        img = _img(session, "/tmp/a.jpg")
        unknown = _person(session, "Unknown 11", is_auto_named=True)
        target = _person(session, "Anikó")
        _face(session, img, target, embedding=E_A, source="manual")  # reference
        selected = _face(session, img, unknown, embedding=E_A)
        sib_diff = _face(session, img, unknown, embedding=E_B)  # stays pending
        sel_id, target_id, diff_id = selected.id, target.id, sib_diff.id

    with session_scope() as session:
        UnknownMergeService(session).assign_unknown_face(sel_id, target_id)

    with session_scope() as session:
        diff = session.get(Face, diff_id)
        assert diff.auto_merge_decision_json is not None
        decision = json.loads(diff.auto_merge_decision_json)
        assert decision["version"] == 1
        assert decision["rule"] == "unknown_cluster_scatter"
        assert decision["target_person_id"] == target_id
        assert decision["selected_face_id"] == sel_id
        assert decision["outcome"] == "pending"
        assert isinstance(decision["gates"], list) and decision["gates"]
        assert isinstance(decision["ranked"], list)


def test_list_pending_exposes_decision_and_bbox(db):
    with session_scope() as session:
        img = _img(session, "/tmp/a.jpg")
        unknown = _person(session, "Unknown 12", is_auto_named=True)
        target = _person(session, "Anikó")
        faces = [_face(session, img, unknown) for _ in range(2)]
        sel_id, target_id = faces[0].id, target.id

    with session_scope() as session:
        UnknownMergeService(session).assign_unknown_face(sel_id, target_id)

    with session_scope() as session:
        rows = UnknownMergeService(session).list_pending()
        assert len(rows) == 1
        row = rows[0]
        assert row.bbox == (1, 2, 3, 4)
        assert row.person_id == target_id  # person_id IS the target
        # No embeddings → empty ranking, but a decision is still recorded.
        assert row.decision is not None
        assert row.decision["outcome"] == "pending"


def test_decision_recorded_when_auto_merge_disabled(db):
    cfg = RecognitionConfig(unknown_auto_merge_enabled=False)
    with session_scope() as session:
        img = _img(session, "/tmp/a.jpg")
        unknown = _person(session, "Unknown 13", is_auto_named=True)
        target = _person(session, "Anikó")
        _face(session, img, target, embedding=E_A, source="manual")
        selected = _face(session, img, unknown, embedding=E_A)
        sib = _face(session, img, unknown, embedding=E_A)
        sel_id, target_id, sib_id = selected.id, target.id, sib.id

    with session_scope() as session:
        UnknownMergeService(session, cfg).assign_unknown_face(sel_id, target_id)

    with session_scope() as session:
        sib = session.get(Face, sib_id)
        assert sib.auto_merge_review_status == REVIEW_PENDING
        decision = json.loads(sib.auto_merge_decision_json)
        assert decision["auto_merge_enabled"] is False
        assert decision["outcome"] == "pending"


def test_old_row_without_decision_is_tolerated(db):
    with session_scope() as session:
        img = _img(session, "/tmp/a.jpg")
        unknown = _person(session, "Unknown 14", is_auto_named=True)
        target = _person(session, "Anikó")
        f = _face(session, img, target, source=SOURCE_AUTO_MERGE)
        f.auto_merged_from_unknown = True
        f.auto_merge_review_status = REVIEW_PENDING
        f.auto_merge_decision_json = None  # legacy row
        session.flush()

    with session_scope() as session:
        rows = UnknownMergeService(session).list_pending()
        assert rows[0].decision is None
        assert rows[0].confidence is None


def test_delete_face_removes_row(db):
    with session_scope() as session:
        img = _img(session, "/tmp/a.jpg")
        unknown = _person(session, "Unknown 10", is_auto_named=True)
        target = _person(session, "Anikó")
        faces = [_face(session, img, unknown) for _ in range(2)]
        sel_id, target_id = faces[0].id, target.id

    with session_scope() as session:
        svc = UnknownMergeService(session)
        svc.assign_unknown_face(sel_id, target_id)
        sib_id = svc.pending_face_ids()[0]

    with session_scope() as session:
        assert UnknownMergeService(session).delete_face(sib_id) is True

    with session_scope() as session:
        assert session.get(Face, sib_id) is None
