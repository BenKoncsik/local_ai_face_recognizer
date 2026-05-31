"""Unit tests for the Identity Repair Scan."""

from __future__ import annotations

import numpy as np
import pytest

from app.config import IdentityRepairConfig
from app.db.database import init_db, session_scope
from app.db.models import Face, FaceCorrection, Image, Person
from app.services.identity_repair_service import IdentityRepairService


@pytest.fixture()
def tmp_db(tmp_path):
    db_file = tmp_path / "test.db"
    init_db(db_file)
    return db_file


def _unit(index: int, tilt: float = 0.0, dim: int = 64) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[index] = 1.0
    if tilt:
        v[(index + 1) % dim] = tilt
    return (v / np.linalg.norm(v)).astype(np.float32)


def _add_image(session, path: str) -> int:
    img = Image(file_path=path, file_hash=path, file_mtime=0.0)
    session.add(img)
    session.flush()
    return img.id


def _add_person(session, name, auto=True, protected=False) -> int:
    p = Person(name=name, is_auto_named=auto, is_protected=protected)
    session.add(p)
    session.flush()
    return p.id


def _add_face(session, image_id, person_id, emb) -> int:
    f = Face(
        image_id=image_id, person_id=person_id,
        bbox_x=0, bbox_y=0, bbox_w=20, bbox_h=20,
        confidence=1.0, detector_backend="cpu",
    )
    f.set_embedding(emb)
    session.add(f)
    session.flush()
    return f.id


class TestScan:
    def test_fragments_of_same_person_are_proposed(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s, "/i.jpg")
            p98 = _add_person(s, "Unknown 98")
            p155 = _add_person(s, "Unknown 155")
            other = _add_person(s, "Unknown 200")
            _add_face(s, img, p98, _unit(0, 0.01))
            _add_face(s, img, p98, _unit(0, 0.02))
            _add_face(s, img, p155, _unit(0, 0.03))
            _add_face(s, img, other, _unit(40))

        with session_scope() as s:
            cands = IdentityRepairService(s).scan()

        assert len(cands) == 1
        c = cands[0]
        assert {c.person_a_id, c.person_b_id} == set(
            pid for pid in [c.person_a_id, c.person_b_id]
        )
        assert c.confidence > 0.9
        names = {c.name_a, c.name_b}
        assert names == {"Unknown 98", "Unknown 155"}

    def test_dissimilar_persons_not_proposed(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s, "/i.jpg")
            a = _add_person(s, "Unknown 1")
            b = _add_person(s, "Unknown 2")
            _add_face(s, img, a, _unit(0))
            _add_face(s, img, b, _unit(30))

        with session_scope() as s:
            assert IdentityRepairService(s).scan() == []

    def test_named_and_protected_persons_excluded(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s, "/i.jpg")
            named = _add_person(s, "Alice", auto=False)
            prot = _add_person(s, "??", auto=True, protected=True)
            _add_face(s, img, named, _unit(0))
            _add_face(s, img, prot, _unit(0, 0.01))

        with session_scope() as s:
            assert IdentityRepairService(s).scan() == []

    def test_diff_correction_suppresses_pair(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s, "/i.jpg")
            a = _add_person(s, "Unknown 1")
            b = _add_person(s, "Unknown 2")
            fa = _add_face(s, img, a, _unit(0, 0.01))
            fb = _add_face(s, img, b, _unit(0, 0.02))
            lo, hi = sorted((fa, fb))
            s.add(FaceCorrection(face_id_a=lo, face_id_b=hi, same_person=False))

        with session_scope() as s:
            assert IdentityRepairService(s).scan() == []


class TestApply:
    def test_apply_merges_transitive_group(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s, "/i.jpg")
            p98 = _add_person(s, "Unknown 98")
            p155 = _add_person(s, "Unknown 155")
            p184 = _add_person(s, "Unknown 184")
            # p98 has the most faces → should survive.
            _add_face(s, img, p98, _unit(0))
            _add_face(s, img, p98, _unit(0, 0.01))
            _add_face(s, img, p155, _unit(0, 0.02))
            _add_face(s, img, p184, _unit(0, 0.03))
            ids = (p98, p155, p184)

        with session_scope() as s:
            res = IdentityRepairService(s).apply(
                [(ids[0], ids[1]), (ids[1], ids[2])]
            )

        assert res.groups_consolidated == 1
        assert res.persons_merged_away == 2
        with session_scope() as s:
            persons = s.query(Person).all()
            assert len(persons) == 1
            assert persons[0].name == "Unknown 98"
            assert len(persons[0].faces) == 4

    def test_apply_ignores_unknown_person_ids(self, tmp_db):
        with session_scope() as s:
            img = _add_image(s, "/i.jpg")
            a = _add_person(s, "Unknown 1")
            _add_face(s, img, a, _unit(0))
            real_id = a

        with session_scope() as s:
            res = IdentityRepairService(s).apply([(real_id, 9999)])

        assert res.groups_consolidated == 0
        assert res.persons_merged_away == 0


def test_scan_then_apply_roundtrip(tmp_db):
    with session_scope() as s:
        img = _add_image(s, "/i.jpg")
        p1 = _add_person(s, "Unknown 1")
        p2 = _add_person(s, "Unknown 2")
        _add_face(s, img, p1, _unit(5, 0.01))
        _add_face(s, img, p2, _unit(5, 0.02))

    with session_scope() as s:
        svc = IdentityRepairService(s)
        cands = svc.scan()
        assert len(cands) == 1
        pairs = [(c.person_a_id, c.person_b_id) for c in cands]
        res = svc.apply(pairs)
        assert res.persons_merged_away == 1

    with session_scope() as s:
        assert s.query(Person).count() == 1
