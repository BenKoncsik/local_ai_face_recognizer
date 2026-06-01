"""Tests for the screen-recording image-file timeline log."""

from __future__ import annotations

from app.services.recording_timeline_log import (
    RecordingTimelineLog,
    format_timestamp,
)


def test_format_timestamp() -> None:
    assert format_timestamp(0) == "00:00:00"
    assert format_timestamp(12) == "00:00:12"
    assert format_timestamp(68) == "00:01:08"
    assert format_timestamp(3661) == "01:01:01"
    assert format_timestamp(-5) == "00:00:00"


def test_basic_timeline(tmp_path) -> None:
    log = RecordingTimelineLog(tmp_path / "timeline.txt")
    log.note_active("1984_PICT0346.JPG", "Panni", 0)
    log.note_active("1984_PICT0347.JPG", None, 12)
    log.note_active("1984_PICT0348.JPG", "Kati", 31)
    out = log.finalize(68)

    text = out.read_text(encoding="utf-8")
    assert text == (
        "00:00:00 - 00:00:12 | 1984_PICT0346.JPG | kijelölt személy: Panni\n"
        "00:00:12 - 00:00:31 | 1984_PICT0347.JPG | kijelölt személy: -\n"
        "00:00:31 - 00:01:08 | 1984_PICT0348.JPG | kijelölt személy: Kati\n"
    )


def test_consecutive_identical_states_collapse(tmp_path) -> None:
    log = RecordingTimelineLog(tmp_path / "timeline.txt")
    log.note_active("a.jpg", "Panni", 0)
    log.note_active("a.jpg", "Panni", 5)  # no change — ignored
    log.note_active("a.jpg", "Panni", 9)  # no change — ignored
    log.note_active("b.jpg", "Panni", 10)
    log.finalize(20)

    lines = (tmp_path / "timeline.txt").read_text(encoding="utf-8").splitlines()
    assert lines == [
        "00:00:00 - 00:00:10 | a.jpg | kijelölt személy: Panni",
        "00:00:10 - 00:00:20 | b.jpg | kijelölt személy: Panni",
    ]


def test_person_change_on_same_image_starts_new_entry(tmp_path) -> None:
    log = RecordingTimelineLog(tmp_path / "timeline.txt")
    log.note_active("a.jpg", None, 0)
    log.note_active("a.jpg", "Kati", 4)
    log.finalize(10)

    lines = (tmp_path / "timeline.txt").read_text(encoding="utf-8").splitlines()
    assert lines == [
        "00:00:00 - 00:00:04 | a.jpg | kijelölt személy: -",
        "00:00:04 - 00:00:10 | a.jpg | kijelölt személy: Kati",
    ]


def test_partial_written_crash_safe(tmp_path) -> None:
    log = RecordingTimelineLog(tmp_path / "timeline.txt")
    log.note_active("a.jpg", "Panni", 0)
    log.note_active("b.jpg", "Kati", 7)
    # Simulate a crash before finalize: the partial file must be readable.
    partial = tmp_path / "timeline.txt.partial"
    assert partial.exists()
    body = partial.read_text(encoding="utf-8")
    assert "a.jpg" in body and "Panni" in body


def test_finalize_removes_partial(tmp_path) -> None:
    log = RecordingTimelineLog(tmp_path / "timeline.txt")
    log.note_active("a.jpg", "Panni", 0)
    log.finalize(5)
    assert not (tmp_path / "timeline.txt.partial").exists()


def test_empty_timeline(tmp_path) -> None:
    log = RecordingTimelineLog(tmp_path / "timeline.txt")
    out = log.finalize(0)
    assert out.read_text(encoding="utf-8") == ""


def test_custom_prefix_and_placeholder(tmp_path) -> None:
    log = RecordingTimelineLog(
        tmp_path / "timeline.txt",
        person_prefix="selected person:",
        none_placeholder="(none)",
    )
    log.note_active(None, None, 0)
    log.finalize(3)
    line = (tmp_path / "timeline.txt").read_text(encoding="utf-8").strip()
    assert line == "00:00:00 - 00:00:03 | (none) | selected person: (none)"
