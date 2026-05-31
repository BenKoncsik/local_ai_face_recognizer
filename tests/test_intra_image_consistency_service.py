"""Unit tests for the same-image identity consistency pass."""

from __future__ import annotations

import numpy as np
import pytest

from app.config import IntraImageConsistencyConfig
from app.db.database import init_db, session_scope
from app.db.models import Face, FaceCorrection, Image, Person
from app.services.intra_image_consistency_service import (
    IntraImageConsistencyService,
)


@pytest.fixture()
def tmp_db(tmp_path):
    db_file = tmp_path / "test.db"
    init_db(db_file)
    return db_file


def _unit(dim: int, index: int, tilt: float = 0.0, tilt_index: int = 1) -> np.ndarray:
    """A near-axis unit vector; *tilt* nudges it toward another axis."""
    v = np.zeros(dim, dtype=np.float32)
    v[index] = 1.0
    if tilt:
        v[tilt_index] = tilt
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32)


def _add_image(session, path: str) -> int:
    img = Image(file_path=path, file_hash=path, file_mtime=0.0)
    session.add(img)
    session.flush()
    return img.id


def _add_face(session, image_id: int, embedding, person_id=None, source=None) -> int:
    face = Face(
        image_id=image_id,
        person_id=person_id,
        bbox_x=0, bbox_y=0, bbox_w=20, bbox_h=20,
        confidence=1.0,
        detector_backend="cpu",
        assignment_source=source,
    )
    face.set_embedding(embedding)
    session.add(face)
    session.flush()
    return face.id


class TestSplitUnknownHealing:
    def test_two_unknowns_on_one_image_are_merged(self, tmp_db):
        """The core bug: same person, same image, two Unknown identities."""
        dim = 64
        with session_scope() as s:
            img = _add_image(s, "/photo.jpg")
            p98 = Person(name="Unknown 98", is_auto_named=True)
            p155 = Person(name="Unknown 155", is_auto_named=True)
            s.add_all([p98, p155])
            s.flush()
            # Two near-identical faces, but split across two Unknown persons.
            _add_face(s, img, _unit(dim, 0, tilt=0.02), person_id=p98.id,
                      source="clustering")
            _add_face(s, img, _unit(dim, 0, tilt=0.05), person_id=p155.id,
                      source="clustering")

        with session_scope() as s:
            stats = IntraImageConsistencyService(s).run()

        assert stats.n_faces_reassigned >= 1
        assert stats.n_persons_removed == 1
        with session_scope() as s:
            faces = s.query(Face).all()
            assert len({f.person_id for f in faces}) == 1
            # Only one Unknown person remains.
            assert s.query(Person).count() == 1

    def test_whole_losing_unknown_is_absorbed(self, tmp_db):
        """Faces of the losing Unknown on *other* images follow the merge."""
        dim = 64
        with session_scope() as s:
            img1 = _add_image(s, "/a.jpg")
            img2 = _add_image(s, "/b.jpg")
            winner = Person(name="Unknown 1", is_auto_named=True)
            loser = Person(name="Unknown 2", is_auto_named=True)
            s.add_all([winner, loser])
            s.flush()
            # winner has 2 faces (so it wins the tie-break), loser has 1 here…
            _add_face(s, img1, _unit(dim, 0), person_id=winner.id)
            _add_face(s, img2, _unit(dim, 0, tilt=0.01), person_id=winner.id)
            # …plus the co-occurring split face on img1 and an off-image face.
            _add_face(s, img1, _unit(dim, 0, tilt=0.03), person_id=loser.id)
            off = _add_face(s, img2, _unit(dim, 5), person_id=loser.id)

        with session_scope() as s:
            IntraImageConsistencyService(s).run()

        with session_scope() as s:
            # The loser is gone and even its unrelated face moved to the winner.
            assert s.query(Person).count() == 1
            moved = s.get(Face, off)
            assert moved.person_id is not None
            survivor = s.query(Person).one()
            assert moved.person_id == survivor.id


class TestSafety:
    def test_different_people_not_merged(self, tmp_db):
        """Two clearly different faces on one image keep separate identities."""
        dim = 64
        with session_scope() as s:
            img = _add_image(s, "/group.jpg")
            pa = Person(name="Unknown 1", is_auto_named=True)
            pb = Person(name="Unknown 2", is_auto_named=True)
            s.add_all([pa, pb])
            s.flush()
            _add_face(s, img, _unit(dim, 0), person_id=pa.id)
            _add_face(s, img, _unit(dim, 30), person_id=pb.id)

        with session_scope() as s:
            stats = IntraImageConsistencyService(s).run()

        assert stats.n_faces_reassigned == 0
        with session_scope() as s:
            assert s.query(Person).count() == 2

    def test_diff_correction_blocks_merge(self, tmp_db):
        """A user 'not same person' correction prevents merging look-alikes."""
        dim = 64
        with session_scope() as s:
            img = _add_image(s, "/twins.jpg")
            pa = Person(name="Unknown 1", is_auto_named=True)
            pb = Person(name="Unknown 2", is_auto_named=True)
            s.add_all([pa, pb])
            s.flush()
            fa = _add_face(s, img, _unit(dim, 0, tilt=0.02), person_id=pa.id)
            fb = _add_face(s, img, _unit(dim, 0, tilt=0.04), person_id=pb.id)
            lo, hi = sorted((fa, fb))
            s.add(FaceCorrection(face_id_a=lo, face_id_b=hi, same_person=False))

        with session_scope() as s:
            stats = IntraImageConsistencyService(s).run()

        assert stats.n_faces_reassigned == 0
        with session_scope() as s:
            assert s.query(Person).count() == 2

    def test_two_named_persons_conflict_skipped(self, tmp_db):
        """Look-alike faces under two *named* persons are never auto-merged."""
        dim = 64
        with session_scope() as s:
            img = _add_image(s, "/pair.jpg")
            alice = Person(name="Alice", is_auto_named=False)
            bob = Person(name="Bob", is_auto_named=False)
            s.add_all([alice, bob])
            s.flush()
            _add_face(s, img, _unit(dim, 0, tilt=0.02), person_id=alice.id,
                      source="manual")
            _add_face(s, img, _unit(dim, 0, tilt=0.03), person_id=bob.id,
                      source="manual")

        with session_scope() as s:
            stats = IntraImageConsistencyService(s).run()

        assert stats.n_conflicts_skipped == 1
        assert stats.n_faces_reassigned == 0
        with session_scope() as s:
            assert s.query(Person).count() == 2

    def test_unknown_pulled_into_named_anchor(self, tmp_db):
        """A split Unknown face matching a named face joins the named person…"""
        dim = 64
        with session_scope() as s:
            img = _add_image(s, "/anchor.jpg")
            alice = Person(name="Alice", is_auto_named=False)
            unknown = Person(name="Unknown 5", is_auto_named=True)
            s.add_all([alice, unknown])
            s.flush()
            _add_face(s, img, _unit(dim, 0, tilt=0.01), person_id=alice.id,
                      source="manual")
            split = _add_face(s, img, _unit(dim, 0, tilt=0.03),
                              person_id=unknown.id, source="clustering")
            # …but a *different* Unknown face on another image must NOT follow.
            img2 = _add_image(s, "/other.jpg")
            other = _add_face(s, img2, _unit(dim, 9), person_id=unknown.id)

        with session_scope() as s:
            IntraImageConsistencyService(s).run()

        with session_scope() as s:
            alice = s.query(Person).filter_by(name="Alice").one()
            moved = s.get(Face, split)
            assert moved.person_id == alice.id
            assert moved.assignment_source == "intra_image"
            # The unrelated Unknown face stays put (named anchors only absorb
            # the co-occurring face, not the whole Unknown identity).
            unrelated = s.get(Face, other)
            assert unrelated.person_id != alice.id

    def test_disabled_config_is_noop(self, tmp_db):
        dim = 64
        with session_scope() as s:
            img = _add_image(s, "/p.jpg")
            pa = Person(name="Unknown 1", is_auto_named=True)
            pb = Person(name="Unknown 2", is_auto_named=True)
            s.add_all([pa, pb])
            s.flush()
            _add_face(s, img, _unit(dim, 0), person_id=pa.id)
            _add_face(s, img, _unit(dim, 0, tilt=0.02), person_id=pb.id)

        cfg = IntraImageConsistencyConfig(enabled=False)
        with session_scope() as s:
            stats = IntraImageConsistencyService(s, cfg).run()

        assert stats.n_faces_reassigned == 0
        with session_scope() as s:
            assert s.query(Person).count() == 2
