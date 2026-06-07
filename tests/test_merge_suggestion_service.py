"""Tests for the background merge-suggestion engine."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from app.config import MatchingConfig
from app.db.database import init_db, session_scope
from app.db.models import (
    MERGE_STATUS_DISMISSED,
    MERGE_STATUS_PENDING,
    Face,
    Image,
    MergeSuggestion,
    Person,
)
from app.services.merge_suggestion_service import (
    MergeSuggestionService,
    compute_confidence,
)

DIM = 64


@pytest.fixture()
def tmp_db(tmp_path):
    init_db(tmp_path / "test.db")
    return tmp_path


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _axis_vec(index: int, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[index] = 1.0
    if noise:
        v += np.random.default_rng(seed).normal(0, noise, DIM).astype(np.float32)
    return v.astype(np.float32)


def _add_person(session, name, *, auto, nickname=None) -> int:
    p = Person(name=name, is_auto_named=auto, nickname=nickname)
    session.add(p)
    session.flush()
    return p.id


def _add_face(session, person_id, vec) -> int:
    img = Image(file_path=f"/img/{np.random.rand()}.jpg", file_hash="h", file_mtime=0.0)
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


def _seed_match(session, *, axis=0, named_name="Anna", auto_name="Unknown 1"):
    """A named person and a look-alike auto-named person on the same axis."""
    named = _add_person(session, named_name, auto=False)
    auto = _add_person(session, auto_name, auto=True)
    for s in range(3):
        _add_face(session, named, _axis_vec(axis, noise=0.02, seed=s))
        _add_face(session, auto, _axis_vec(axis, noise=0.02, seed=s + 10))
    session.flush()
    return named, auto


# ---------------------------------------------------------------------------
# compute_confidence
# ---------------------------------------------------------------------------

class TestComputeConfidence:
    def test_face_only(self):
        assert compute_confidence(0.8, None) == pytest.approx(0.8)

    def test_name_only_is_capped(self):
        # A perfect name with no face evidence must stay low.
        assert compute_confidence(None, 1.0) <= 0.5

    def test_both_face_dominates(self):
        both = compute_confidence(0.8, 1.0)
        assert both > compute_confidence(0.8, None)
        # Face still dominates the blend.
        assert both < compute_confidence(1.0, None)

    def test_neither(self):
        assert compute_confidence(None, None) == 0.0

    def test_clamped(self):
        assert 0.0 <= compute_confidence(0.99, 0.99) <= 1.0


# ---------------------------------------------------------------------------
# Scoring + persistence
# ---------------------------------------------------------------------------

class TestScoringAndPersistence:
    def test_generates_suggestion_for_lookalike(self, tmp_db):
        with session_scope() as session:
            _seed_match(session)
        with session_scope() as session:
            svc = MergeSuggestionService(session, MatchingConfig())
            results = svc.score_candidates(
                svc.load_candidates(), svc.load_targets(), svc.load_suppressed_pairs()
            )
            assert len(results) == 1
            assert results[0].face_similarity is not None
            svc.persist_results(results, "job-1", datetime.utcnow() + timedelta(days=1))
        with session_scope() as session:
            rows = session.query(MergeSuggestion).all()
            assert len(rows) == 1
            assert rows[0].status == MERGE_STATUS_PENDING

    def test_no_duplicate_on_reruns_and_parallel_chunks(self, tmp_db):
        # Two distinct look-alike pairs on different axes.
        with session_scope() as session:
            _seed_match(session, axis=0, named_name="Anna", auto_name="U1")
            _seed_match(session, axis=1, named_name="Béla", auto_name="U2")
        future = datetime.utcnow() + timedelta(days=1)

        def run_once():
            with session_scope() as session:
                svc = MergeSuggestionService(session, MatchingConfig())
                cands = svc.load_candidates()
                targets = svc.load_targets()
                supp = svc.load_suppressed_pairs()
                # Simulate two parallel chunks, one candidate each.
                for chunk in ([cands[0]], [cands[1]]):
                    res = svc.score_candidates(chunk, targets, supp)
                    svc.persist_results(res, "job", future)

        run_once()
        run_once()  # rerun must not duplicate
        with session_scope() as session:
            rows = session.query(MergeSuggestion).all()
            assert len(rows) == 2
            pairs = {(r.source_person_id, r.target_person_id) for r in rows}
            assert len(pairs) == 2

    def test_manual_edit_not_overwritten(self, tmp_db):
        """A person edited after the job snapshot must not get a suggestion."""
        with session_scope() as session:
            named, auto = _seed_match(session)
        job_started = datetime.utcnow()

        # User edits the auto person AFTER the job's snapshot time.
        with session_scope() as session:
            p = session.get(Person, auto)
            p.updated_at = job_started + timedelta(minutes=5)
            session.commit()

        with session_scope() as session:
            svc = MergeSuggestionService(session, MatchingConfig())
            results = svc.score_candidates(
                svc.load_candidates(), svc.load_targets(), svc.load_suppressed_pairs()
            )
            written = svc.persist_results(results, "job-stale", job_started)
            assert written == 0
        with session_scope() as session:
            assert session.query(MergeSuggestion).count() == 0

    def test_candidate_named_by_user_is_skipped(self, tmp_db):
        with session_scope() as session:
            named, auto = _seed_match(session)
            results_holder = {}
        future = datetime.utcnow() + timedelta(days=1)
        with session_scope() as session:
            svc = MergeSuggestionService(session, MatchingConfig())
            results = svc.score_candidates(
                svc.load_candidates(), svc.load_targets(), svc.load_suppressed_pairs()
            )
        # User names the candidate before persistence runs.
        with session_scope() as session:
            p = session.get(Person, auto)
            p.is_auto_named = False
            session.commit()
        with session_scope() as session:
            svc = MergeSuggestionService(session, MatchingConfig())
            written = svc.persist_results(results, "job", future)
            assert written == 0

    def test_suppressed_pair_not_resuggested(self, tmp_db):
        with session_scope() as session:
            named, auto = _seed_match(session)
        future = datetime.utcnow() + timedelta(days=1)
        with session_scope() as session:
            svc = MergeSuggestionService(session, MatchingConfig())
            res = svc.score_candidates(
                svc.load_candidates(), svc.load_targets(), svc.load_suppressed_pairs()
            )
            svc.persist_results(res, "job", future)
            row = session.query(MergeSuggestion).one()
            svc.dismiss(row.id)
        # Next run must respect the dismissal.
        with session_scope() as session:
            svc = MergeSuggestionService(session, MatchingConfig())
            supp = svc.load_suppressed_pairs()
            res = svc.score_candidates(svc.load_candidates(), svc.load_targets(), supp)
            assert res == []
            rows = session.query(MergeSuggestion).all()
            assert len(rows) == 1
            assert rows[0].status == MERGE_STATUS_DISMISSED


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

class TestDecisions:
    def test_accept_merges_persons(self, tmp_db):
        with session_scope() as session:
            named, auto = _seed_match(session)
        future = datetime.utcnow() + timedelta(days=1)
        with session_scope() as session:
            svc = MergeSuggestionService(session, MatchingConfig())
            res = svc.score_candidates(
                svc.load_candidates(), svc.load_targets(), svc.load_suppressed_pairs()
            )
            svc.persist_results(res, "job", future)
            row = session.query(MergeSuggestion).one()
            svc.accept(row.id)
        with session_scope() as session:
            # Auto person merged away; only the named survivor remains.
            assert session.get(Person, auto) is None
            survivor = session.get(Person, named)
            assert survivor is not None
            assert len(survivor.faces) == 6

    def test_list_open(self, tmp_db):
        with session_scope() as session:
            _seed_match(session)
        future = datetime.utcnow() + timedelta(days=1)
        with session_scope() as session:
            svc = MergeSuggestionService(session, MatchingConfig())
            res = svc.score_candidates(
                svc.load_candidates(), svc.load_targets(), svc.load_suppressed_pairs()
            )
            svc.persist_results(res, "job", future)
            dtos = svc.list_open()
            assert len(dtos) == 1
            dto = dtos[0]
            assert dto.candidate_name == "Unknown 1"
            assert dto.target_name == "Anna"
            assert dto.confidence > 0


# ---------------------------------------------------------------------------
# Merge exclusion ("exclude from merge" workflow)
# ---------------------------------------------------------------------------

class TestMergeExclusion:
    def test_excluded_face_dropped_from_profile(self, tmp_db):
        from app.services.identity_service import IdentityService

        with session_scope() as session:
            _, auto = _seed_match(session)
            auto_face_ids = [
                f.id for f in session.get(Person, auto).faces
            ]
        # Exclude one of the auto person's faces from the merge.
        with session_scope() as session:
            IdentityService(session).set_face_merge_excluded(
                auto_face_ids[0], True
            )
        with session_scope() as session:
            svc = MergeSuggestionService(session, MatchingConfig())
            profiles = {p.person_id: p for p in svc.load_candidates()}
            # The candidate now has only 2 usable faces (3 minus the excluded).
            assert profiles[auto].face_count == 2

    def test_accept_keeps_excluded_face_in_source(self, tmp_db):
        from app.db.models import MERGE_STATUS_ACCEPTED
        from app.services.identity_service import IdentityService

        with session_scope() as session:
            named, auto = _seed_match(session)
            excluded_id = session.get(Person, auto).faces[0].id
        with session_scope() as session:
            IdentityService(session).set_face_merge_excluded(excluded_id, True)

        future = datetime.utcnow() + timedelta(days=1)
        with session_scope() as session:
            svc = MergeSuggestionService(session, MatchingConfig())
            res = svc.score_candidates(
                svc.load_candidates(), svc.load_targets(), svc.load_suppressed_pairs()
            )
            svc.persist_results(res, "job", future)
            row = session.query(MergeSuggestion).one()
            svc.accept(row.id)

        with session_scope() as session:
            # Source survives, holding only the excluded face (flag cleared).
            source = session.get(Person, auto)
            assert source is not None
            assert [f.id for f in source.faces] == [excluded_id]
            assert source.faces[0].is_merge_excluded is False
            # Target received the 3 named + 2 non-excluded auto faces.
            survivor = session.get(Person, named)
            assert len(survivor.faces) == 5
            # The suggestion is resolved, not still pending.
            resolved = session.get(MergeSuggestion, row.id)
            assert resolved is not None
            assert resolved.status == MERGE_STATUS_ACCEPTED

    def test_accept_all_excluded_raises(self, tmp_db):
        from app.services.identity_service import IdentityService

        with session_scope() as session:
            named, auto = _seed_match(session)
            all_ids = [f.id for f in session.get(Person, auto).faces]
        future = datetime.utcnow() + timedelta(days=1)
        with session_scope() as session:
            svc = MergeSuggestionService(session, MatchingConfig())
            res = svc.score_candidates(
                svc.load_candidates(), svc.load_targets(), svc.load_suppressed_pairs()
            )
            svc.persist_results(res, "job", future)
            row_id = session.query(MergeSuggestion).one().id
        # Exclude every auto face.
        with session_scope() as session:
            ident = IdentityService(session)
            for fid in all_ids:
                ident.set_face_merge_excluded(fid, True)
        with session_scope() as session:
            svc = MergeSuggestionService(session, MatchingConfig())
            with pytest.raises(ValueError):
                svc.accept(row_id)
        # Both persons untouched.
        with session_scope() as session:
            assert session.get(Person, named) is not None
            assert session.get(Person, auto) is not None

    def test_toggle_persists(self, tmp_db):
        from app.services.identity_service import IdentityService

        with session_scope() as session:
            _, auto = _seed_match(session)
            fid = session.get(Person, auto).faces[0].id
        with session_scope() as session:
            IdentityService(session).set_face_merge_excluded(fid, True)
        with session_scope() as session:
            assert session.get(Face, fid).is_merge_excluded is True
        with session_scope() as session:
            IdentityService(session).set_face_merge_excluded(fid, False)
        with session_scope() as session:
            assert session.get(Face, fid).is_merge_excluded is False
