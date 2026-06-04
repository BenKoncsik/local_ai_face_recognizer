"""Tests for video/subtitle filename pairing."""

from __future__ import annotations

from app.services.subtitle_service import (
    find_matching_subtitle,
    subtitle_path_for_video,
)


def test_subtitle_path_for_video_uses_same_basename() -> None:
    assert str(subtitle_path_for_video("/tmp/recording.mp4")).endswith(
        "/tmp/recording.srt"
    )


def test_exact_recording_srt_is_selected(tmp_path) -> None:
    video = tmp_path / "recording.mp4"
    subtitle = tmp_path / "recording.srt"
    video.touch()
    subtitle.touch()

    assert find_matching_subtitle(video) == subtitle


def test_language_suffix_is_not_selected(tmp_path) -> None:
    video = tmp_path / "recording.mp4"
    video.touch()
    (tmp_path / "recording_en.srt").touch()

    assert find_matching_subtitle(video) is None


def test_unrelated_subtitle_is_not_selected(tmp_path) -> None:
    video = tmp_path / "recording.mp4"
    video.touch()
    (tmp_path / "subtitle.srt").touch()

    assert find_matching_subtitle(video) is None


def test_dotted_language_suffix_is_not_selected(tmp_path) -> None:
    video = tmp_path / "recording.mp4"
    video.touch()
    (tmp_path / "recording.hu.srt").touch()

    assert find_matching_subtitle(video) is None


def test_case_difference_still_matches(tmp_path) -> None:
    video = tmp_path / "RECORDING.MP4"
    subtitle = tmp_path / "recording.srt"
    video.touch()
    subtitle.touch()

    assert find_matching_subtitle(video) == subtitle


def test_only_exact_match_is_selected_among_multiple_srt_files(tmp_path) -> None:
    video = tmp_path / "recording.mp4"
    exact = tmp_path / "recording.srt"
    video.touch()
    (tmp_path / "recording_en.srt").touch()
    (tmp_path / "subtitle.srt").touch()
    (tmp_path / "recording.hu.srt").touch()
    exact.touch()

    assert find_matching_subtitle(video) == exact


def test_missing_matching_subtitle_returns_none(tmp_path) -> None:
    video = tmp_path / "recording.mp4"
    video.touch()

    assert find_matching_subtitle(video) is None
