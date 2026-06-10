"""Tests for the "already decided" suggestion history (MergeDecision log)."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from app.config import MatchingConfig
from app.db.database import init_db, session_scope
from app.db.models import (
    MERGE_DECISION_ACCEPTED,
    MERGE_DECISION_DISMISSED,
    MERGE_DECISION_REJECTED,
    MERGE_DECISION_SOURCE_AUTO,
    MERGE_DECISION_SOURCE_MANUAL,
    Face,
    Image,
    MergeDecision,
    MergeSuggestion,
    Person,
)
from app.services.merge_suggestion_service import MergeSuggestionService

DIM = 64


@pytest.fixture()
def tmp_db(tmp_path):
    init_db(tmp_path / "test.db")
    return tmp_path


def _axis_vec(index: int, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[index] = 1.0
    if noise:
        v += np.random.default_rng(seed).normal(0, noise, DIM).astype(np.float32)
    return v.astype(np.float32)


def _add_person(session, name, *, auto) -> int:
    p = Person(name=name, is_auto_named=auto)
    session.add(p)
    session.flush()
    return p.id


def _add_face(session, person_id, vec) -> int:
    img = Image(
        file_path=f"/img/{np.random.rand()}.jpg", file_hash="h", file_mtime=0.0
    )
    session.add(img)
    session.flush()
    f = Face(
        image_id=img.id,
        person_id=person_id,
        bbox_x=0, bbox_y=0, bbox_w=100, bbox_h=100,
        confidence=0.9,
        crop_path=f"/crop/{np.random.rand()}.jpg",
    )
    f.set_embedding(vec)
    session.add(f)
    session.flush()
    return f.id


def _seed_suggestion(axis=0, named_name="Anna", auto_name="Unknown 1") -> int:
    """Create a look-alike pair and persist one pending suggestion."""
    with session_scope() as session:
        named = _add_person(session, named_name, auto=False)
        auto = _add_person(session, auto_name, auto=True)
        for s in range(3):
            _add_face(session, named, _axis_vec(axis, noise=0.02, seed=s))
            _add_face(session, auto, _axis_vec(axis, noise=0.02, seed=s + 10))
    with session_scope() as session:
        svc = MergeSuggestionService(session, MatchingConfig())
        results = svc.score_candidates(
            svc.load_candidates(), svc.load_targets(), svc.load_suppressed_pairs()
        )
        assert results, "expected a look-alike suggestion"
        svc.persist_results(results, "job-1", datetime.utcnow() + timedelta(days=1))
    with session_scope() as session:
        return session.query(MergeSuggestion).one().id


class TestDecisionLogging:
    def test_reject_logs_decision(self, tmp_db):
        sid = _seed_suggestion()
        with session_scope() as session:
            MergeSuggestionService(session, MatchingConfig()).reject(sid)

        with session_scope() as session:
            row = session.query(MergeDecision).one()
            assert row.decision == MERGE_DECISION_REJECTED
            assert row.source == MERGE_DECISION_SOURCE_MANUAL
            assert row.candidate_name == "Unknown 1"
            assert row.target_name == "Anna"
            assert row.candidate_crop_path
            assert row.decided_at is not None

    def test_accept_logs_and_survives_merge_cascade(self, tmp_db):
        """The suggestion row dies with the merged person; the log must not."""
        sid = _seed_suggestion()
        with session_scope() as session:
            MergeSuggestionService(session, MatchingConfig()).accept(sid)

        with session_scope() as session:
            # The merge deleted the candidate person and the suggestion row…
            assert session.query(MergeSuggestion).count() == 0
            # …but the decided list still shows what happened.
            row = session.query(MergeDecision).one()
            assert row.decision == MERGE_DECISION_ACCEPTED
            assert row.candidate_name == "Unknown 1"
            assert row.target_name == "Anna"

    def test_dismiss_logs_decision(self, tmp_db):
        sid = _seed_suggestion()
        with session_scope() as session:
            MergeSuggestionService(session, MatchingConfig()).dismiss(sid)

        with session_scope() as session:
            row = session.query(MergeDecision).one()
            assert row.decision == MERGE_DECISION_DISMISSED

    def test_defer_does_not_log(self, tmp_db):
        """'Decide later' is not a decision — it stays in the open list."""
        sid = _seed_suggestion()
        with session_scope() as session:
            MergeSuggestionService(session, MatchingConfig()).defer(sid)

        with session_scope() as session:
            assert session.query(MergeDecision).count() == 0

    def test_auto_merge_logs_with_auto_source(self, tmp_db):
        _seed_suggestion()
        with session_scope() as session:
            merged = MergeSuggestionService(
                session, MatchingConfig()
            ).auto_merge_suggestions(min_confidence=0.5, max_unknown_faces=10)
            assert merged == 1

        with session_scope() as session:
            row = session.query(MergeDecision).one()
            assert row.decision == MERGE_DECISION_ACCEPTED
            assert row.source == MERGE_DECISION_SOURCE_AUTO


class TestDecidedListing:
    def test_list_decided_newest_first(self, tmp_db):
        first = _seed_suggestion(axis=0, named_name="Anna", auto_name="U1")
        second = _seed_suggestion(axis=20, named_name="Béla", auto_name="U2")

        with session_scope() as session:
            svc = MergeSuggestionService(session, MatchingConfig())
            svc.reject(first)
            svc.reject(second)

        with session_scope() as session:
            svc = MergeSuggestionService(session, MatchingConfig())
            assert svc.count_decided() == 2
            decided = svc.list_decided()
            assert len(decided) == 2
            # Newest decision first.
            assert decided[0].candidate_name == "U2"
            assert decided[1].candidate_name == "U1"
            assert all(d.decision == MERGE_DECISION_REJECTED for d in decided)

    def test_list_decided_respects_limit(self, tmp_db):
        for i in range(3):
            sid = _seed_suggestion(
                axis=i * 15, named_name=f"P{i}", auto_name=f"U{i}"
            )
            with session_scope() as session:
                MergeSuggestionService(session, MatchingConfig()).reject(sid)

        with session_scope() as session:
            svc = MergeSuggestionService(session, MatchingConfig())
            assert len(svc.list_decided(limit=2)) == 2
            assert svc.count_decided() == 3

    def test_empty_history(self, tmp_db):
        with session_scope() as session:
            svc = MergeSuggestionService(session, MatchingConfig())
            assert svc.count_decided() == 0
            assert svc.list_decided() == []
