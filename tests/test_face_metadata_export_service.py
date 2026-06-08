"""Tests for FaceMetadataExportService and the image_metadata helper.

Covers payload generation (known/unknown persons, notes, name toggle),
embed/sidecar write modes, idempotent re-export, EXIF preservation, and
controlled failure handling.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import piexif
import pytest
from PIL import Image as PilImage

from app.db.database import init_db, session_scope
from app.db.models import Face, Image, Person
from app.services.face_metadata_export_service import (
    FaceMetadataExportOptions,
    FaceMetadataExportService,
)
from app.utils import image_metadata as meta


@pytest.fixture()
def db(tmp_path):
    init_db(tmp_path / "test_face_meta.db")
    return tmp_path


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _make_image(session, file_path, **kwargs) -> Image:
    img = Image(
        file_path=str(file_path),
        file_hash="hash",
        file_mtime=0.0,
        detection_done=True,
        **kwargs,
    )
    session.add(img)
    session.flush()
    return img


def _make_person(session, name, **kwargs) -> Person:
    p = Person(name=name, is_auto_named=False, **kwargs)
    session.add(p)
    session.flush()
    return p


def _make_face(session, image, person=None, *, x=10, y=20, w=30, h=40,
               confidence=0.9, **kwargs) -> Face:
    f = Face(
        image_id=image.id,
        person_id=person.id if person else None,
        bbox_x=x, bbox_y=y, bbox_w=w, bbox_h=h,
        confidence=confidence,
        detector_backend="cpu",
        is_excluded=False,
        **kwargs,
    )
    session.add(f)
    session.flush()
    return f


def _write_jpeg(path: Path, *, exif: bytes | None = None) -> None:
    img = PilImage.new("RGB", (200, 150), (40, 80, 120))
    if exif:
        img.save(path, format="JPEG", exif=exif)
    else:
        img.save(path, format="JPEG")


# ---------------------------------------------------------------------------
# 1 + 5 — payload generation, known person, name toggle
# ---------------------------------------------------------------------------

def test_payload_known_person(db):
    with session_scope() as session:
        alice = _make_person(session, "Alice", notes="bal oldalon")
        img = _make_image(session, db / "p.jpg", relative_path="album/p.jpg")
        _make_face(session, img, alice, x=120, y=80, w=64, h=72,
                   assignment_source="manual")

        payload = FaceMetadataExportService(session).build_payload(img)

    assert payload["schema"] == meta.FACELOCAL_SCHEMA
    assert payload["image_id"] == img.id
    assert payload["relative_path"] == "album/p.jpg"
    assert "exported_at" in payload
    assert len(payload["faces"]) == 1
    face = payload["faces"][0]
    assert face["person_name"] == "Alice"
    assert face["person_id"] == alice.id
    assert face["box"] == {"x": 120, "y": 80, "width": 64, "height": 72}
    assert face["assignment_source"] == "manual"
    assert "confidence" in face


def test_payload_excludes_name_when_disabled(db):
    with session_scope() as session:
        alice = _make_person(session, "Alice")
        img = _make_image(session, db / "p.jpg")
        _make_face(session, img, alice)

        opts = FaceMetadataExportOptions(include_person_name=False)
        payload = FaceMetadataExportService(session).build_payload(img, opts)

    face = payload["faces"][0]
    assert "person_name" not in face
    assert face["person_id"] == alice.id  # id still present


# ---------------------------------------------------------------------------
# 2 — unknown person
# ---------------------------------------------------------------------------

def test_payload_unknown_person(db):
    with session_scope() as session:
        img = _make_image(session, db / "p.jpg")
        _make_face(session, img, None, x=120, y=80, w=64, h=72, confidence=0.42)

        payload = FaceMetadataExportService(session).build_payload(img)

    face = payload["faces"][0]
    assert face["unknown"] is True
    assert "person_name" not in face
    assert "person_id" not in face
    assert face["box"] == {"x": 120, "y": 80, "width": 64, "height": 72}
    assert face["assignment_source"] == "unknown"
    assert face["confidence"] == 0.42


def test_protected_person_treated_as_unknown(db):
    with session_scope() as session:
        unknown_p = _make_person(session, "Ismeretlen", is_protected=True)
        img = _make_image(session, db / "p.jpg")
        _make_face(session, img, unknown_p)

        payload = FaceMetadataExportService(session).build_payload(img)

    assert payload["faces"][0]["unknown"] is True
    assert "person_name" not in payload["faces"][0]


# ---------------------------------------------------------------------------
# 3 + 4 — notes inclusion / exclusion
# ---------------------------------------------------------------------------

def test_notes_included_when_enabled(db):
    with session_scope() as session:
        alice = _make_person(session, "Alice", notes="megjegyzés")
        img = _make_image(session, db / "p.jpg")
        _make_face(session, img, alice)
        payload = FaceMetadataExportService(session).build_payload(img)
    assert payload["faces"][0]["notes"] == "megjegyzés"


def test_notes_excluded_when_disabled(db):
    with session_scope() as session:
        alice = _make_person(session, "Alice", notes="megjegyzés")
        img = _make_image(session, db / "p.jpg")
        _make_face(session, img, alice)
        opts = FaceMetadataExportOptions(include_notes=False)
        payload = FaceMetadataExportService(session).build_payload(img, opts)
    assert "notes" not in payload["faces"][0]


# ---------------------------------------------------------------------------
# 8 — empty face list still yields a valid payload
# ---------------------------------------------------------------------------

def test_empty_face_list_valid_payload(db):
    with session_scope() as session:
        img = _make_image(session, db / "p.jpg")
        payload = FaceMetadataExportService(session).build_payload(img)
    assert payload["schema"] == meta.FACELOCAL_SCHEMA
    assert payload["faces"] == []


# ---------------------------------------------------------------------------
# Embedded write (XMP) + 6 — idempotent re-export
# ---------------------------------------------------------------------------

def test_export_embeds_exif_comment_and_is_idempotent(db):
    jpg = db / "photo.jpg"
    _write_jpeg(jpg)

    with session_scope() as session:
        alice = _make_person(session, "Alice")
        img = _make_image(session, jpg)
        _make_face(session, img, alice)
        svc = FaceMetadataExportService(session)
        r1 = svc.export_image(img.id)
        r2 = svc.export_image(img.id)

    # JPEG now embeds the JSON in the EXIF UserComment ("comment") by default.
    assert r1.success and r1.write_mode == meta.WRITE_MODE_EXIF_USER_COMMENT
    assert r2.success and r2.write_mode == meta.WRITE_MODE_EXIF_USER_COMMENT

    # The UserComment is overwritten in place — a re-export does not duplicate.
    import piexif
    with PilImage.open(jpg) as im:
        exif = piexif.load(im.info["exif"])
    comment = exif["Exif"][piexif.ExifIFD.UserComment]
    assert comment.count(b'"schema"') == 1

    # And the data round-trips.
    payload = meta.read_face_metadata(jpg)
    assert payload is not None
    assert payload["faces"][0]["person_name"] == "Alice"


def test_export_xmp_when_exif_comment_disabled(db):
    jpg = db / "photo.jpg"
    _write_jpeg(jpg)

    with session_scope() as session:
        alice = _make_person(session, "Alice")
        img = _make_image(session, jpg)
        _make_face(session, img, alice)
        opts = FaceMetadataExportOptions(embed_in_exif_comment=False)
        result = FaceMetadataExportService(session).export_image(img.id, opts)

    assert result.write_mode == meta.WRITE_MODE_XMP
    payload = meta.read_face_metadata(jpg)
    assert payload["faces"][0]["person_name"] == "Alice"


# ---------------------------------------------------------------------------
# 9 — existing EXIF GPS / date survives an embedded write
# ---------------------------------------------------------------------------

def test_existing_exif_preserved(db):
    jpg = db / "gps.jpg"
    exif_dict = {
        "0th": {piexif.ImageIFD.DateTime: b"2019:05:01 12:00:00"},
        "Exif": {piexif.ExifIFD.DateTimeOriginal: b"2019:05:01 12:00:00"},
        "GPS": {
            piexif.GPSIFD.GPSLatitudeRef: b"N",
            piexif.GPSIFD.GPSLatitude: [(47, 1), (30, 1), (0, 1)],
            piexif.GPSIFD.GPSLongitudeRef: b"E",
            piexif.GPSIFD.GPSLongitude: [(19, 1), (3, 1), (0, 1)],
        },
        "1st": {}, "thumbnail": None,
    }
    _write_jpeg(jpg, exif=piexif.dump(exif_dict))

    with session_scope() as session:
        alice = _make_person(session, "Alice")
        img = _make_image(session, jpg)
        _make_face(session, img, alice)
        FaceMetadataExportService(session).export_image(img.id)

    with PilImage.open(jpg) as im:
        after = piexif.load(im.info["exif"])
    assert after["GPS"][piexif.GPSIFD.GPSLatitudeRef] == b"N"
    assert after["GPS"][piexif.GPSIFD.GPSLatitude] == ((47, 1), (30, 1), (0, 1))
    assert after["0th"][piexif.ImageIFD.DateTime] == b"2019:05:01 12:00:00"


# ---------------------------------------------------------------------------
# 7 — unwritable / non-embeddable image → sidecar JSON
# ---------------------------------------------------------------------------

def test_raw_format_falls_back_to_sidecar(db):
    raw = db / "photo.cr2"
    raw.write_bytes(b"not a real raw but unsupported ext")

    with session_scope() as session:
        alice = _make_person(session, "Alice")
        img = _make_image(session, raw)
        _make_face(session, img, alice)
        result = FaceMetadataExportService(session).export_image(img.id)

    assert result.success
    assert result.write_mode == meta.WRITE_MODE_SIDECAR
    sidecar = Path(result.sidecar_path)
    assert sidecar.exists()
    assert sidecar.name == "photo.cr2.facelocal.json"
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["faces"][0]["person_name"] == "Alice"


def test_prefer_sidecar_only_never_touches_image(db):
    jpg = db / "photo.jpg"
    _write_jpeg(jpg)
    before = jpg.read_bytes()

    with session_scope() as session:
        alice = _make_person(session, "Alice")
        img = _make_image(session, jpg)
        _make_face(session, img, alice)
        opts = FaceMetadataExportOptions(prefer_sidecar_only=True)
        result = FaceMetadataExportService(session).export_image(img.id, opts)

    assert result.write_mode == meta.WRITE_MODE_SIDECAR
    assert jpg.read_bytes() == before  # image untouched
    assert meta.sidecar_path_for(jpg).exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission semantics")
def test_readonly_image_falls_back_to_sidecar(db):
    jpg = db / "ro.jpg"
    _write_jpeg(jpg)
    os.chmod(jpg, stat.S_IRUSR)
    try:
        with session_scope() as session:
            alice = _make_person(session, "Alice")
            img = _make_image(session, jpg)
            _make_face(session, img, alice)
            result = FaceMetadataExportService(session).export_image(img.id)
    finally:
        os.chmod(jpg, stat.S_IRUSR | stat.S_IWUSR)

    assert result.success
    assert result.write_mode == meta.WRITE_MODE_SIDECAR


# ---------------------------------------------------------------------------
# 10 — controlled failure for a bad path / missing image
# ---------------------------------------------------------------------------

def test_missing_image_record_returns_failed(db):
    with session_scope() as session:
        result = FaceMetadataExportService(session).export_image(99999)
    assert not result.success
    assert result.write_mode == meta.WRITE_MODE_FAILED
    assert result.error_message


def test_missing_file_on_disk_returns_failed(db):
    with session_scope() as session:
        alice = _make_person(session, "Alice")
        img = _make_image(session, db / "does_not_exist.jpg")
        _make_face(session, img, alice)
        result = FaceMetadataExportService(session).export_image(img.id)
    assert not result.success
    assert result.write_mode == meta.WRITE_MODE_FAILED


# ---------------------------------------------------------------------------
# Batch summary + do-not-overwrite
# ---------------------------------------------------------------------------

def test_batch_summary_counts(db):
    jpg = db / "a.jpg"
    _write_jpeg(jpg)
    raw = db / "b.cr2"
    raw.write_bytes(b"raw")

    with session_scope() as session:
        alice = _make_person(session, "Alice")
        img_a = _make_image(session, jpg)
        _make_face(session, img_a, alice)
        img_b = _make_image(session, raw)
        _make_face(session, img_b, alice)
        summary = FaceMetadataExportService(session).export_images([img_a.id, img_b.id])

    assert summary.total == 2
    assert summary.embedded_count == 1
    assert summary.sidecar_count == 1
    assert summary.failed_count == 0


def test_no_overwrite_skips_when_block_exists(db):
    jpg = db / "photo.jpg"
    _write_jpeg(jpg)

    with session_scope() as session:
        alice = _make_person(session, "Alice")
        img = _make_image(session, jpg)
        _make_face(session, img, alice)
        svc = FaceMetadataExportService(session)
        svc.export_image(img.id)  # first write
        opts = FaceMetadataExportOptions(overwrite_existing_facelocal_metadata=False)
        result = svc.export_image(img.id, opts)

    assert result.write_mode == meta.WRITE_MODE_SKIPPED
    assert result.success


def test_windows_lock_on_replace_still_embeds(db, monkeypatch):
    """Simulate the Windows case where os.replace fails on a locked target.

    The embed must succeed via the in-place fallback, NOT degrade to a sidecar.
    """
    jpg = db / "locked.jpg"
    _write_jpeg(jpg)

    real_replace = os.replace

    def fake_replace(src, dst):
        # Only the image target is "locked"; the sidecar (a new file) is fine.
        if str(dst).endswith("locked.jpg"):
            raise PermissionError("simulated Windows lock")
        return real_replace(src, dst)

    monkeypatch.setattr(meta.os, "replace", fake_replace)
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)

    with session_scope() as session:
        alice = _make_person(session, "Alice")
        img = _make_image(session, jpg)
        _make_face(session, img, alice)
        result = FaceMetadataExportService(session).export_image(img.id)

    assert result.success
    assert result.write_mode == meta.WRITE_MODE_EXIF_USER_COMMENT  # embedded, not sidecar
    assert not meta.sidecar_path_for(jpg).exists()
    payload = meta.read_face_metadata(jpg)
    assert payload["faces"][0]["person_name"] == "Alice"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS Finder comment")
def test_macos_finder_comment_writes_xattr(tmp_path, monkeypatch):
    # The helper is a no-op under pytest (avoids Finder prompts); clear the
    # marker so this one test exercises the real macOS write path.
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    import subprocess

    jpg = tmp_path / "fc.jpg"
    _write_jpeg(jpg)
    ok = meta.write_macos_finder_comment(jpg, '{"schema":"facelocal.faces.v1"}')
    assert ok
    out = subprocess.run(
        ["xattr", "-p", "com.apple.metadata:kMDItemFinderComment", str(jpg)],
        capture_output=True, text=True,
    )
    assert "facelocal.faces.v1" in out.stdout


def test_png_xmp_roundtrip(db):
    png = db / "photo.png"
    PilImage.new("RGB", (50, 50), (1, 2, 3)).save(png)

    with session_scope() as session:
        alice = _make_person(session, "Alice")
        img = _make_image(session, png)
        _make_face(session, img, alice)
        result = FaceMetadataExportService(session).export_image(img.id)

    assert result.write_mode == meta.WRITE_MODE_XMP
    payload = meta.read_face_metadata(png)
    assert payload["faces"][0]["person_name"] == "Alice"


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------

def test_cancel_stops_before_next_image(db):
    """A cancel requested mid-batch stops further processing cleanly."""
    from app.jobs.cancellation import CancellationToken

    paths = [db / f"img_{i}.jpg" for i in range(4)]
    for p in paths:
        _write_jpeg(p)

    token = CancellationToken()

    with session_scope() as session:
        alice = _make_person(session, "Alice")
        ids = []
        for p in paths:
            img = _make_image(session, p)
            _make_face(session, img, alice)
            ids.append(img.id)

        # Cancel after the second image has finished.
        def cb(done, total, name):
            if done == 2:
                token.cancel()

        summary = FaceMetadataExportService(session).export_images(
            ids, progress_cb=cb, cancel_token=token
        )

    assert summary.cancelled is True
    assert summary.requested_total == 4
    assert summary.total == 2            # only two were processed
    assert summary.remaining_count == 2  # the rest were left untouched
    assert summary.failed_count == 0     # cancellation is not a failure
    # The two unprocessed images must have no metadata written.
    assert meta.read_face_metadata(paths[0]) is not None
    assert meta.read_face_metadata(paths[3]) is None


def test_no_cancel_processes_all(db):
    """Without cancellation everything is processed and cancelled stays False."""
    from app.jobs.cancellation import CancellationToken

    paths = [db / f"all_{i}.jpg" for i in range(3)]
    for p in paths:
        _write_jpeg(p)

    seen = []

    with session_scope() as session:
        alice = _make_person(session, "Alice")
        ids = []
        for p in paths:
            img = _make_image(session, p)
            _make_face(session, img, alice)
            ids.append(img.id)

        summary = FaceMetadataExportService(session).export_images(
            ids,
            progress_cb=lambda done, total, name: seen.append((done, total, name)),
            cancel_token=CancellationToken(),
        )

    assert summary.cancelled is False
    assert summary.total == 3
    assert summary.remaining_count == 0
    # Progress reports the filename of the image being processed.
    names = {name for _, _, name in seen if name}
    assert names == {p.name for p in paths}
