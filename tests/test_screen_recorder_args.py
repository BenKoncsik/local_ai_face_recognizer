"""Tests for the pure ffmpeg-argument helpers of the screen recorder."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services.screen_recorder_service import (
    CaptureDevices,
    RecordingOptions,
    build_concat_args,
    build_concat_list,
    build_ffmpeg_args,
    parse_avfoundation_devices,
    parse_dshow_audio_devices,
    pick_system_audio,
    quality_preset,
)


def test_quality_presets() -> None:
    assert quality_preset("low")["scale_height"] == 540
    assert quality_preset("normal")["scale_height"] == 720
    assert quality_preset("better")["scale_height"] == 1080
    # Unknown falls back to normal.
    assert quality_preset("bogus") == quality_preset("normal")


def test_macos_args_mic_only() -> None:
    args = build_ffmpeg_args(
        "darwin",
        CaptureDevices(screen="0", microphone="1", system_audio=None),
        RecordingOptions(fps=18, quality="normal", segment_seconds=8),
        "/out/seg_%05d.mp4",
    )
    assert "avfoundation" in args
    assert "-capture_cursor" in args
    # Single combined video:audio input.
    i = args.index("-i")
    assert args[i + 1] == "0:1"
    # Single audio source → mapped from input 0.
    assert "0:a" in args
    assert "-f" in args and "segment" in args
    assert args[-1] == "/out/seg_%05d.mp4"


def test_macos_args_with_system_audio_mixes() -> None:
    args = build_ffmpeg_args(
        "darwin",
        CaptureDevices(screen="0", microphone="1", system_audio="2"),
        RecordingOptions(),
        "/out/seg_%05d.mp4",
    )
    assert "-filter_complex" in args
    fc = args[args.index("-filter_complex") + 1]
    assert "amix=inputs=2" in fc
    assert "[aout]" in args


def test_windows_args_mic_only() -> None:
    args = build_ffmpeg_args(
        "win32",
        CaptureDevices(screen="desktop", microphone="Mic (USB)", system_audio=None),
        RecordingOptions(capture_cursor=True),
        r"C:\out\seg_%05d.mp4",
    )
    assert "gdigrab" in args
    assert "-draw_mouse" in args
    assert "audio=Mic (USB)" in args
    # Windows mic is a separate input → mapped from input 1.
    assert "1:a" in args


def test_windows_args_with_system_audio_mixes() -> None:
    args = build_ffmpeg_args(
        "win32",
        CaptureDevices(
            screen="desktop",
            microphone="Mic",
            system_audio="virtual-audio-capturer",
        ),
        RecordingOptions(),
        r"C:\out\seg_%05d.mp4",
    )
    fc = args[args.index("-filter_complex") + 1]
    assert fc.startswith("[1:a][2:a]amix")


def test_no_audio_uses_an() -> None:
    args = build_ffmpeg_args(
        "darwin",
        CaptureDevices(screen="0", microphone=None, system_audio=None),
        RecordingOptions(),
        "/out/seg_%05d.mp4",
    )
    assert "-an" in args


def test_unsupported_platform_raises() -> None:
    with pytest.raises(ValueError):
        build_ffmpeg_args(
            "linux",
            CaptureDevices(screen="0"),
            RecordingOptions(),
            "/out/seg_%05d.mp4",
        )


def test_forced_keyframes_at_segment_boundary() -> None:
    # The segment muxer only cuts on keyframes; without forced keyframes the
    # output would not actually split, breaking crash protection.
    args = build_ffmpeg_args(
        "darwin",
        CaptureDevices(screen="0", microphone="1"),
        RecordingOptions(fps=20, segment_seconds=5),
        "/out/seg_%05d.mp4",
    )
    assert "-force_key_frames" in args
    assert args[args.index("-force_key_frames") + 1] == "expr:gte(t,n_forced*5)"
    # GOP == fps * segment_seconds.
    assert args[args.index("-g") + 1] == "100"


def test_cursor_disabled() -> None:
    args = build_ffmpeg_args(
        "darwin",
        CaptureDevices(screen="0", microphone="1"),
        RecordingOptions(capture_cursor=False),
        "/out/seg_%05d.mp4",
    )
    assert args[args.index("-capture_cursor") + 1] == "0"


def test_concat_list_escapes_quotes() -> None:
    body = build_concat_list([Path("/a/seg_1.mp4"), Path("/a/o'clock.mp4")])
    assert "file '/a/seg_1.mp4'" in body
    assert r"o'\''clock" in body


def test_concat_args() -> None:
    args = build_concat_args(Path("/a/list.txt"), Path("/a/out.mp4"))
    assert "concat" in args
    assert "-c" in args and "copy" in args
    assert args[-1] == "/a/out.mp4"


def test_parse_avfoundation_devices() -> None:
    text = (
        "[AVFoundation indev @ 0x1] AVFoundation video devices:\n"
        "[AVFoundation indev @ 0x1] [0] FaceTime HD Camera\n"
        "[AVFoundation indev @ 0x1] [1] Capture screen 0\n"
        "[AVFoundation indev @ 0x1] AVFoundation audio devices:\n"
        "[AVFoundation indev @ 0x1] [0] MacBook Pro Microphone\n"
        "[AVFoundation indev @ 0x1] [1] BlackHole 2ch\n"
    )
    parsed = parse_avfoundation_devices(text)
    assert ("1", "Capture screen 0") in parsed["video"]
    assert ("1", "BlackHole 2ch") in parsed["audio"]


def test_parse_dshow_audio_devices() -> None:
    text = (
        'video devices\n'
        '"Integrated Camera"\n'
        'audio devices\n'
        '"Microphone (Realtek)"\n'
        '"virtual-audio-capturer"\n'
    )
    names = parse_dshow_audio_devices(text)
    assert "Microphone (Realtek)" in names
    assert "virtual-audio-capturer" in names
    assert "Integrated Camera" not in names


def test_pick_system_audio() -> None:
    assert pick_system_audio(["Mic", "BlackHole 2ch"]) == "BlackHole 2ch"
    assert pick_system_audio(["Mic", "virtual-audio-capturer"]) == (
        "virtual-audio-capturer"
    )
    assert pick_system_audio(["Mic", "Line In"]) is None
