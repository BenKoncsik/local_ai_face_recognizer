"""Tests for FaceDateResolver source priority and caching."""

from __future__ import annotations

from types import SimpleNamespace

from app.services.face_date_service import FaceDateResolver
from app.utils.fuzzy_date import Precision


def _img(iid, photo_date=None, estimated_date=None):
    return SimpleNamespace(id=iid, photo_date=photo_date, estimated_date=estimated_date)


def _face(iid, photo_date=None, estimated_date=None):
    return SimpleNamespace(image=_img(iid, photo_date, estimated_date))


def test_photo_date_takes_priority():
    r = FaceDateResolver()
    fd = r.for_face(_face(1, photo_date="1954", estimated_date="1900"))
    assert fd.earliest.year == 1954
    assert fd.is_estimated is False


def test_falls_back_to_estimated_and_flags_it():
    r = FaceDateResolver()
    fd = r.for_face(_face(2, photo_date=None, estimated_date="1930"))
    assert fd.earliest.year == 1930
    assert fd.precision is Precision.YEAR
    assert fd.is_estimated is True          # forced even though "1930" looks exact


def test_unknown_when_no_source():
    r = FaceDateResolver()
    fd = r.for_face(_face(3))
    assert fd.precision is Precision.UNKNOWN


def test_cache_returns_same_value_per_image_id():
    r = FaceDateResolver()
    face = _face(4, photo_date="1960")
    first = r.for_face(face)
    # Mutate the underlying string; cached result should be unchanged.
    face.image.photo_date = "1999"
    second = r.for_face(face)
    assert first is second
    assert second.earliest.year == 1960
