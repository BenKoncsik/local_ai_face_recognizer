"""Unit tests for the stored-face geometry cleanup service."""

from __future__ import annotations

from typing import Optional

from app.db.database import init_db, session_scope
from app.db.models import Face, FaceBlob, Image, Person
from app.services.face_geometry_cleanup_service import FaceGeometryCleanupService

# Plausible frontal constellation inside a 100×100 box (YuNet order).
_GOOD = [[32, 40], [68, 40], [50, 58], [38, 75], [62, 75]]
# Collinear "scrambled" points — what a knot / hand / card produces.
_BAD = [[40, 10], [40, 20], [40, 30], [40, 40], [40, 50]]


def _image(session) -> int:
    img = Image(file_path="/x/a.jpg", file_hash="h", file_mtime=0.0)
    session.add(img)
    session.flush()
    return img.id


def _person(session, name: str = "Anna") -> int:
    person = Person(name=name)
    session.add(person)
    session.flush()
    return person.id


def _face(
    session,
    image_id: int,
    landmarks,
    *,
    person_id: Optional[int] = None,
    source: Optional[str] = None,
    backend: str = "cpu",
) -> int:
    face = Face(
        image_id=image_id,
        bbox_x=0, bbox_y=0, bbox_w=100, bbox_h=100,
        confidence=0.7,
        detector_backend=backend,
        person_id=person_id,
        assignment_source=source,
    )
    session.add(face)
    session.flush()
    if landmarks is not None:
        face.set_landmarks(landmarks)
    session.flush()
    return face.id


def test_plausible_face_is_kept(tmp_path):
    init_db(tmp_path / "faces.db")
    with session_scope() as s:
        img = _image(s)
        _face(s, img, _GOOD)

    with session_scope() as s:
        report = FaceGeometryCleanupService(s).scan()

    assert report.scanned == 1
    assert report.droppable_count == 0
    assert report.flagged_count == 0


def test_impossible_unassigned_is_droppable(tmp_path):
    init_db(tmp_path / "faces.db")
    with session_scope() as s:
        img = _image(s)
        fid = _face(s, img, _BAD)

    with session_scope() as s:
        report = FaceGeometryCleanupService(s).scan()

    assert report.droppable_ids == [fid]
    assert report.flagged_count == 0
    assert report.droppable[0].reason  # a concrete failing-check name


def test_impossible_auto_assigned_is_droppable(tmp_path):
    init_db(tmp_path / "faces.db")
    with session_scope() as s:
        img = _image(s)
        pid = _person(s)
        fid = _face(s, img, _BAD, person_id=pid, source="clustering")

    with session_scope() as s:
        report = FaceGeometryCleanupService(s).scan()

    assert report.droppable_ids == [fid]


def test_impossible_manual_assignment_is_flagged_not_dropped(tmp_path):
    init_db(tmp_path / "faces.db")
    with session_scope() as s:
        img = _image(s)
        pid = _person(s)
        fid = _face(s, img, _BAD, person_id=pid, source="manual")

    with session_scope() as s:
        report = FaceGeometryCleanupService(s).scan()

    assert report.droppable_count == 0
    assert [c.face_id for c in report.flagged] == [fid]


def test_impossible_legacy_null_source_assignment_is_flagged(tmp_path):
    """A None assignment_source on an assigned face is trusted (legacy data)."""
    init_db(tmp_path / "faces.db")
    with session_scope() as s:
        img = _image(s)
        pid = _person(s)
        fid = _face(s, img, _BAD, person_id=pid, source=None)

    with session_scope() as s:
        report = FaceGeometryCleanupService(s).scan()

    assert report.droppable_count == 0
    assert [c.face_id for c in report.flagged] == [fid]


def test_manual_backend_face_is_never_dropped(tmp_path):
    init_db(tmp_path / "faces.db")
    with session_scope() as s:
        img = _image(s)
        fid = _face(s, img, _BAD, backend="manual")

    with session_scope() as s:
        report = FaceGeometryCleanupService(s).scan()

    assert report.droppable_count == 0
    assert [c.face_id for c in report.flagged] == [fid]


def test_landmark_less_face_is_not_scanned(tmp_path):
    init_db(tmp_path / "faces.db")
    with session_scope() as s:
        img = _image(s)
        _face(s, img, None)  # no landmarks → fail-open, ignored

    with session_scope() as s:
        report = FaceGeometryCleanupService(s).scan()

    assert report.scanned == 0
    assert report.droppable_count == 0
    assert report.flagged_count == 0


def test_delete_faces_removes_rows_and_blobs(tmp_path):
    init_db(tmp_path / "faces.db")
    with session_scope() as s:
        img = _image(s)
        keep = _face(s, img, _GOOD)
        drop = _face(s, img, _BAD)

    with session_scope() as s:
        svc = FaceGeometryCleanupService(s)
        report = svc.scan()
        removed = svc.delete_faces(report.droppable_ids)

    assert removed == 1
    with session_scope() as s:
        remaining = {f.id for f in s.query(Face).all()}
        assert remaining == {keep}
        # The blob side-row of the deleted face is gone (cascade).
        assert s.query(FaceBlob).filter_by(face_id=drop).count() == 0


def test_scan_and_delete_convenience(tmp_path):
    init_db(tmp_path / "faces.db")
    with session_scope() as s:
        img = _image(s)
        pid = _person(s)
        _face(s, img, _GOOD)
        _face(s, img, _BAD)
        _face(s, img, _BAD, person_id=pid, source="manual")  # flagged, kept

    with session_scope() as s:
        report, deleted = FaceGeometryCleanupService(s).scan_and_delete()

    assert deleted == 1
    assert report.flagged_count == 1
    with session_scope() as s:
        assert s.query(Face).count() == 2
