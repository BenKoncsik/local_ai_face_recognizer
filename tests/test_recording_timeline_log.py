"""Tests for the screen-recording SRT timeline log."""

from __future__ import annotations

from app.services.recording_timeline_log import (
    RecordingTimelineLog,
    format_timestamp,
)

# ---------------------------------------------------------------------------
# format_timestamp
# ---------------------------------------------------------------------------

def test_format_timestamp_zero() -> None:
    assert format_timestamp(0) == "00:00:00,000"


def test_format_timestamp_whole_seconds() -> None:
    assert format_timestamp(12) == "00:00:12,000"
    assert format_timestamp(68) == "00:01:08,000"
    assert format_timestamp(3661) == "01:01:01,000"


def test_format_timestamp_fractional_seconds() -> None:
    assert format_timestamp(12.5) == "00:00:12,500"
    assert format_timestamp(31.2) == "00:00:31,200"
    assert format_timestamp(68.0) == "00:01:08,000"


def test_format_timestamp_negative_clamped() -> None:
    assert format_timestamp(-5) == "00:00:00,000"


def test_format_timestamp_rounding() -> None:
    # 1.9995 rounds to 2.000 s
    assert format_timestamp(1.9995) == "00:00:02,000"
    assert format_timestamp(0.001) == "00:00:00,001"


# ---------------------------------------------------------------------------
# Basic timeline rendering
# ---------------------------------------------------------------------------

def test_basic_srt_output(tmp_path) -> None:
    log = RecordingTimelineLog(tmp_path / "timeline.srt")
    log.note_active("1984_PICT0346.JPG", "Panni", 0)
    log.note_active("1984_PICT0347.JPG", None, 12.5)
    log.note_active("1984_PICT0348.JPG", "Kati", 31.2)
    out = log.finalize(68.0)

    text = out.read_text(encoding="utf-8")
    assert text == (
        "1\n"
        "00:00:00,000 --> 00:00:12,500\n"
        "1984_PICT0346.JPG\n"
        "Kijelölt személy: Panni\n"
        "\n"
        "2\n"
        "00:00:12,500 --> 00:00:31,200\n"
        "1984_PICT0347.JPG\n"
        "\n"
        "3\n"
        "00:00:31,200 --> 00:01:08,000\n"
        "1984_PICT0348.JPG\n"
        "Kijelölt személy: Kati\n"
        "\n"
    )


def test_person_line_omitted_when_no_person(tmp_path) -> None:
    """When person is None the person line must be absent (not a '-' placeholder)."""
    log = RecordingTimelineLog(tmp_path / "timeline.srt")
    log.note_active("img.jpg", None, 0)
    log.finalize(5.0)
    text = (tmp_path / "timeline.srt").read_text(encoding="utf-8")
    # Must NOT contain any placeholder dash line
    assert "kijelölt" not in text.lower() or "személy" not in text.lower()
    lines = text.splitlines()
    # Block body: index, timecode, filename, blank — no 5th line
    assert lines == ["1", "00:00:00,000 --> 00:00:05,000", "img.jpg", ""]


# ---------------------------------------------------------------------------
# Deduplication / state tracking
# ---------------------------------------------------------------------------

def test_consecutive_identical_states_collapse(tmp_path) -> None:
    log = RecordingTimelineLog(tmp_path / "timeline.srt")
    log.note_active("a.jpg", "Panni", 0)
    log.note_active("a.jpg", "Panni", 5)   # no change — ignored
    log.note_active("a.jpg", "Panni", 9)   # no change — ignored
    log.note_active("b.jpg", "Panni", 10)
    log.finalize(20.0)

    text = (tmp_path / "timeline.srt").read_text(encoding="utf-8")
    blocks = [b.strip() for b in text.strip().split("\n\n")]
    assert len(blocks) == 2
    assert "a.jpg" in blocks[0]
    assert "00:00:00,000 --> 00:00:10,000" in blocks[0]
    assert "b.jpg" in blocks[1]
    assert "00:00:10,000 --> 00:00:20,000" in blocks[1]


def test_person_change_on_same_image_starts_new_block(tmp_path) -> None:
    log = RecordingTimelineLog(tmp_path / "timeline.srt")
    log.note_active("a.jpg", None, 0)
    log.note_active("a.jpg", "Kati", 4.0)
    log.finalize(10.0)

    text = (tmp_path / "timeline.srt").read_text(encoding="utf-8")
    blocks = [b.strip() for b in text.strip().split("\n\n")]
    assert len(blocks) == 2
    # First block: no person line
    assert "Kati" not in blocks[0]
    # Second block: person line present
    assert "Kati" in blocks[1]
    assert "00:00:04,000 --> 00:00:10,000" in blocks[1]


# ---------------------------------------------------------------------------
# Crash safety (partial file)
# ---------------------------------------------------------------------------

def test_partial_written_after_each_note(tmp_path) -> None:
    log = RecordingTimelineLog(tmp_path / "timeline.srt")
    log.note_active("a.jpg", "Panni", 0)
    log.note_active("b.jpg", "Kati", 7.0)

    partial = tmp_path / "timeline.srt.partial"
    assert partial.exists()
    body = partial.read_text(encoding="utf-8")
    assert "a.jpg" in body
    assert "b.jpg" in body


def test_finalize_removes_partial(tmp_path) -> None:
    log = RecordingTimelineLog(tmp_path / "timeline.srt")
    log.note_active("a.jpg", "Panni", 0)
    log.finalize(5.0)
    assert not (tmp_path / "timeline.srt.partial").exists()


def test_empty_timeline(tmp_path) -> None:
    log = RecordingTimelineLog(tmp_path / "timeline.srt")
    out = log.finalize(0.0)
    assert out.read_text(encoding="utf-8") == ""


# ---------------------------------------------------------------------------
# Custom prefix
# ---------------------------------------------------------------------------

def test_custom_person_prefix(tmp_path) -> None:
    log = RecordingTimelineLog(
        tmp_path / "timeline.srt",
        person_prefix="selected person:",
    )
    log.note_active("img.jpg", "Alice", 0)
    log.finalize(3.0)
    text = (tmp_path / "timeline.srt").read_text(encoding="utf-8")
    assert "selected person: Alice" in text
