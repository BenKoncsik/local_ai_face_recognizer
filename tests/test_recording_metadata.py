"""Tests for the crash-safe recording-metadata writer."""

from __future__ import annotations

import json

from app.services.recording_metadata import (
    METADATA_FILENAME,
    RecordingMetadata,
    RecordingMetadataWriter,
)


def _read(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_begin_writes_pessimistic_record(tmp_path) -> None:
    meta = RecordingMetadata(started_at="2026-06-02T10:00:00+00:00", fps=18)
    writer = RecordingMetadataWriter(tmp_path / METADATA_FILENAME, meta)
    path = writer.begin()

    assert path.exists()
    data = _read(path)
    # Until finalized, the session is assumed crashed / not cleanly closed.
    assert data["crashed"] is True
    assert data["clean_exit"] is False
    assert data["fps"] == 18
    assert data["ended_at"] is None


def test_finalize_clean_exit(tmp_path) -> None:
    meta = RecordingMetadata(started_at="2026-06-02T10:00:00+00:00", fps=15)
    writer = RecordingMetadataWriter(tmp_path / METADATA_FILENAME, meta)
    writer.begin()
    writer.finalize("2026-06-02T10:05:30+00:00", 330.0)

    data = _read(tmp_path / METADATA_FILENAME)
    assert data["crashed"] is False
    assert data["clean_exit"] is True
    assert data["ended_at"] == "2026-06-02T10:05:30+00:00"
    assert data["duration_seconds"] == 330.0


def test_finalize_crashed_keeps_flag_and_errors(tmp_path) -> None:
    meta = RecordingMetadata(started_at="2026-06-02T10:00:00+00:00")
    writer = RecordingMetadataWriter(tmp_path / METADATA_FILENAME, meta)
    writer.begin()
    writer.add_error("ffmpeg exited unexpectedly")
    writer.finalize("2026-06-02T10:01:00+00:00", 60.0, crashed=True)

    data = _read(tmp_path / METADATA_FILENAME)
    assert data["crashed"] is True
    assert data["clean_exit"] is False
    assert "ffmpeg exited unexpectedly" in data["errors"]


def test_resolution_omitted_when_unknown(tmp_path) -> None:
    meta = RecordingMetadata(started_at="x")
    writer = RecordingMetadataWriter(tmp_path / METADATA_FILENAME, meta)
    writer.begin()
    assert _read(tmp_path / METADATA_FILENAME)["resolution"] is None

    meta2 = RecordingMetadata(started_at="x", width=1920, height=1080)
    writer2 = RecordingMetadataWriter(tmp_path / "m2.json", meta2)
    writer2.begin()
    res = _read(tmp_path / "m2.json")["resolution"]
    assert res == {"width": 1920, "height": 1080}


def test_monitors_and_layout_round_trip(tmp_path) -> None:
    meta = RecordingMetadata(
        started_at="x",
        display_mode="selected",
        monitors=[{"id": "A", "width": 2560, "height": 1440, "is_primary": True}],
        layout={"x": 0, "y": 0, "width": 2560, "height": 1440},
    )
    writer = RecordingMetadataWriter(tmp_path / METADATA_FILENAME, meta)
    writer.finalize("x2", 10.0)
    data = _read(tmp_path / METADATA_FILENAME)
    assert data["display_mode"] == "selected"
    assert data["monitors"][0]["id"] == "A"
    assert data["layout"]["width"] == 2560
