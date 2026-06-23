"""Tests for the retroactive multi-stage face-verification sweep.

Focus: the confidence-exemption rule (do not re-verify — and never delete —
high-confidence faces the live detection gate trusts) and the conservative
droppable rule (human-confirmed / manual faces are flagged, not deleted).
"""

from __future__ import annotations

import cv2
import numpy as np

from app.config import AppConfig
from app.db.database import init_db, session_scope
from app.db.models import Face, Image
from app.services.retroactive_verification_service import (
    RetroactiveVerificationService,
)


def _img_file(tmp_path) -> str:
    path = tmp_path / "photo.jpg"
    cv2.imwrite(str(path), np.full((480, 640, 3), 128, np.uint8))
    return str(path)


def _add_image(session, path: str) -> int:
    image = Image(
        file_path=path, file_hash=path, file_mtime=0.0,
        detection_done=True, embedding_done=True,
    )
    session.add(image)
    session.flush()
    return image.id


def _add_face(session, image_id, *, conf, source=None, person_id=None, backend="yunet") -> int:
    face = Face(
        image_id=image_id, person_id=person_id,
        bbox_x=100, bbox_y=100, bbox_w=80, bbox_h=80,
        confidence=conf, detector_backend=backend, assignment_source=source,
    )
    session.add(face)
    session.flush()
    return face.id


class _RejectAll:
    """Fake gate: claims nothing is a face (verify always returns None)."""

    available = True

    def verify(self, img_bgr, det):
        return None


def _service(session, **inject):
    svc = RetroactiveVerificationService(session=session, config=AppConfig())
    svc._verifier = _RejectAll()
    return svc


def test_high_confidence_faces_are_exempt(tmp_path):
    """A face at/above the exemption confidence is never scanned or deleted,
    even when the verifier would reject it — it matches the live gate."""
    init_db(tmp_path / "t.db")
    exempt = AppConfig().detection.verification_confidence_exempt
    with session_scope() as s:
        img = _add_image(s, _img_file(tmp_path))
        high = _add_face(s, img, conf=exempt + 0.05)   # trusted
        low = _add_face(s, img, conf=exempt - 0.20)    # uncertain

        svc = _service(s)
        report = svc.scan()

        assert high not in report.droppable_ids
        # Only the uncertain face was verified and (rejected →) droppable.
        assert low in report.droppable_ids
        assert report.scanned == 1


def test_verify_all_drops_exemption(tmp_path):
    """With verification_verify_all, even high-confidence faces are re-checked."""
    import dataclasses
    from app.config import AppConfig

    init_db(tmp_path / "t.db")
    cfg = AppConfig()
    cfg.detection = dataclasses.replace(cfg.detection, verification_verify_all=True)
    exempt = cfg.detection.verification_confidence_exempt
    with session_scope() as s:
        img = _add_image(s, _img_file(tmp_path))
        high = _add_face(s, img, conf=exempt + 0.1)  # would be exempt normally

        svc = RetroactiveVerificationService(session=s, config=cfg)
        svc._verifier = _RejectAll()
        report = svc.scan()

        # No exemption: the high-confidence face was verified and dropped.
        assert report.scanned == 1
        assert high in report.droppable_ids


def test_manual_and_assigned_faces_are_flagged_not_dropped(tmp_path):
    """Conservative rule survives the multi-stage path: human-confirmed faces
    that fail verification are flagged, never auto-deleted."""
    init_db(tmp_path / "t.db")
    exempt = AppConfig().detection.verification_confidence_exempt
    with session_scope() as s:
        img = _add_image(s, _img_file(tmp_path))
        person = None  # assignment without a person still counts as automatic
        auto = _add_face(s, img, conf=exempt - 0.2, source="recognition")
        human = _add_face(
            s, img, conf=exempt - 0.2, source="manual", person_id=_person(s)
        )

        report = _service(s).scan()

        assert auto in report.droppable_ids
        assert human not in report.droppable_ids
        assert human in [c.face_id for c in report.flagged]


def _person(session) -> int:
    from app.db.models import Person
    p = Person(name="Anna")
    session.add(p)
    session.flush()
    return p.id
