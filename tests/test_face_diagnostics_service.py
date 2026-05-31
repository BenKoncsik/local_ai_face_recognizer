"""Tests for the per-face recognition diagnostics service."""

from __future__ import annotations

import numpy as np
import pytest

from app.config import RecognitionConfig
from app.db.database import init_db, session_scope
from app.db.models import Face, Image, Person
from app.services.face_diagnostics_service import FaceDiagnosticsService


@pytest.fixture()
def tmp_db(tmp_path):
    init_db(tmp_path / "t.db")


def _unit(index: int, tilt: float = 0.0, dim: int = 64) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[index] = 1.0
    if tilt:
        v[(index + 1) % dim] = tilt
    return (v / np.linalg.norm(v)).astype(np.float32)


def _img(s, p="/i.jpg"):
    im = Image(file_path=p, file_hash=p, file_mtime=0.0)
    s.add(im); s.flush()
    return im.id


def _face(s, img, pid, emb, source=None, w=80, h=80):
    f = Face(image_id=img, person_id=pid, bbox_x=0, bbox_y=0, bbox_w=w, bbox_h=h,
             confidence=1.0, detector_backend="cpu", assignment_source=source)
    f.set_embedding(emb)
    s.add(f); s.flush()
    return f.id


def test_explain_missing_face_returns_none(tmp_db):
    with session_scope() as s:
        assert FaceDiagnosticsService(s).explain(999) is None


def test_no_embedding_verdict(tmp_db):
    with session_scope() as s:
        img = _img(s)
        f = Face(image_id=img, bbox_x=0, bbox_y=0, bbox_w=20, bbox_h=20,
                 confidence=1.0, detector_backend="cpu")
        s.add(f); s.flush()
        fid = f.id
    with session_scope() as s:
        diag = FaceDiagnosticsService(s).explain(fid)
        assert diag is not None
        assert not diag.has_embedding
        assert "No embedding" in diag.verdict


def test_strong_named_match_is_eligible(tmp_db):
    with session_scope() as s:
        img = _img(s)
        alice = Person(name="Alice", is_auto_named=False)
        s.add(alice); s.flush()
        # Training faces for Alice.
        _face(s, img, alice.id, _unit(0), source="manual")
        _face(s, img, alice.id, _unit(0, 0.02), source="manual")
        # Unassigned look-alike face.
        cand = _face(s, _img(s, "/b.jpg"), None, _unit(0, 0.03))
    with session_scope() as s:
        diag = FaceDiagnosticsService(s).explain(cand)
        assert diag.top_named
        assert diag.top_named[0].name == "Alice"
        assert diag.top_named[0].similarity > 0.9
        assert "eligible" in diag.verdict.lower()


def test_weak_match_left_for_clustering(tmp_db):
    with session_scope() as s:
        img = _img(s)
        alice = Person(name="Alice", is_auto_named=False)
        s.add(alice); s.flush()
        _face(s, img, alice.id, _unit(0), source="manual")
        _face(s, img, alice.id, _unit(0, 0.02), source="manual")
        # A clearly different person.
        cand = _face(s, _img(s, "/b.jpg"), None, _unit(40))
    with session_scope() as s:
        diag = FaceDiagnosticsService(s).explain(cand)
        assert diag.top_named[0].similarity < diag.adaptive_threshold
        assert "below" in diag.verdict.lower() or "Unknown" in diag.verdict


def test_unknown_centroids_are_scored(tmp_db):
    with session_scope() as s:
        img = _img(s)
        u = Person(name="Unknown 1", is_auto_named=True)
        s.add(u); s.flush()
        _face(s, img, u.id, _unit(5))
        cand = _face(s, _img(s, "/b.jpg"), None, _unit(5, 0.02))
    with session_scope() as s:
        diag = FaceDiagnosticsService(s).explain(cand)
        assert diag.top_unknown
        assert diag.top_unknown[0].name == "Unknown 1"
        assert diag.top_unknown[0].is_auto_named


def test_adaptive_threshold_lower_for_small_face(tmp_db):
    cfg = RecognitionConfig()
    with session_scope() as s:
        img = _img(s)
        big = _face(s, img, None, _unit(0), w=200, h=200)
        small = _face(s, img, None, _unit(0), w=30, h=30)
    with session_scope() as s:
        svc = FaceDiagnosticsService(s, config=cfg)
        d_big = svc.explain(big)
        d_small = svc.explain(small)
        assert d_small.adaptive_threshold <= d_big.adaptive_threshold
