"""Tests for the lock-tolerant EXIF GPS/date writers in app.utils.exif."""

from __future__ import annotations

from datetime import datetime

import piexif
from PIL import Image as PilImage

from app.utils.exif import read_exif_gps, write_exif_date, write_exif_gps


def _make_jpeg(path, *, exif_bytes: bytes | None = None) -> None:
    img = PilImage.new("RGB", (120, 90), (30, 60, 90))
    if exif_bytes:
        img.save(path, format="JPEG", exif=exif_bytes)
    else:
        img.save(path, format="JPEG")


def test_write_gps_and_date_roundtrip(tmp_path):
    jpg = tmp_path / "p.jpg"
    _make_jpeg(jpg)

    assert write_exif_gps(jpg, 47.5, 19.05) is True
    assert write_exif_date(jpg, datetime(1985, 7, 20, 0, 0, 0)) is True

    lat, lon = read_exif_gps(jpg)
    assert round(lat, 3) == 47.5
    assert round(lon, 3) == 19.05

    ex = piexif.load(str(jpg))
    assert ex["Exif"][piexif.ExifIFD.DateTimeOriginal] == b"1985:07:20 00:00:00"


def test_gps_and_date_coexist(tmp_path):
    """Writing the date must not wipe the GPS written earlier (and vice versa)."""
    jpg = tmp_path / "p.jpg"
    _make_jpeg(jpg)

    write_exif_gps(jpg, 47.5, 19.05)
    write_exif_date(jpg, datetime(2001, 2, 3))

    assert read_exif_gps(jpg) is not None
    ex = piexif.load(str(jpg))
    assert ex["Exif"][piexif.ExifIFD.DateTimeOriginal] == b"2001:02:03 00:00:00"


def test_write_gps_survives_windows_lock(tmp_path, monkeypatch):
    """Simulate Windows os.replace PermissionError → in-place fallback succeeds."""
    jpg = tmp_path / "locked.jpg"
    _make_jpeg(jpg)

    import app.utils.image_metadata as meta

    real_replace = meta.os.replace

    def fake_replace(src, dst):
        if str(dst).endswith("locked.jpg"):
            raise PermissionError("simulated Windows lock")
        return real_replace(src, dst)

    monkeypatch.setattr(meta.os, "replace", fake_replace)
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)

    assert write_exif_gps(jpg, 47.5, 19.05) is True
    assert read_exif_gps(jpg) is not None
    # No leftover temp file.
    assert not (tmp_path / "locked.jpg.facelocal.tmp").exists()


def test_write_gps_quality_preserved_no_recompress_blowup(tmp_path):
    """A repeated GPS write should not keep shrinking/inflating the JPEG wildly."""
    jpg = tmp_path / "p.jpg"
    _make_jpeg(jpg)
    write_exif_gps(jpg, 47.5, 19.05)
    size1 = jpg.stat().st_size
    write_exif_gps(jpg, 47.6, 19.06)
    size2 = jpg.stat().st_size
    # quality="keep" → near-identical recompression, not a big drift.
    assert abs(size2 - size1) < size1 * 0.5


def test_missing_file_returns_false(tmp_path):
    assert write_exif_gps(tmp_path / "nope.jpg", 1.0, 2.0) is False
    assert write_exif_date(tmp_path / "nope.jpg", datetime(2000, 1, 1)) is False
