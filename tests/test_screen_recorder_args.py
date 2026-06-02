"""Tests for the pure ffmpeg-argument helpers of the screen recorder."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services.screen_recorder_service import (
    CaptureDevices,
    CaptureRegion,
    RecordingDisplayInfo,
    RecordingDisplayMode,
    RecordingOptions,
    build_concat_args,
    build_concat_list,
    build_ffmpeg_args,
    displays_bounding_box,
    effective_fps,
    format_display_label,
    parse_avfoundation_devices,
    parse_dshow_audio_devices,
    pick_system_audio,
    quality_preset,
    resolve_capture_region,
    selected_displays,
)


def _displays():
    """Two monitors: primary 2560x1440 at origin, secondary 1920x1080 to the right."""
    return [
        RecordingDisplayInfo(
            id="A", name="Dell", width=2560, height=1440,
            is_primary=True, x=0, y=0, av_index=0,
        ),
        RecordingDisplayInfo(
            id="B", name="LG", width=1920, height=1080,
            is_primary=False, x=2560, y=0, av_index=1,
        ),
    ]


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


def test_pick_microphone_prefers_builtin_over_continuity() -> None:
    from app.services.screen_recorder_service import pick_microphone
    audio = [
        ("0", "Benedek iPhone-ja mikrofonja"),
        ("1", "AirPods Pro"),
        ("2", "MacBook Air mikrofon"),
    ]
    # Built-in mic wins even though the iPhone is index 0.
    assert pick_microphone(audio) == "2"
    # No built-in → first non-Continuity device (AirPods over iPhone).
    assert pick_microphone(audio[:2]) == "1"
    # Only a Continuity device → fall back to it rather than nothing.
    assert pick_microphone([("0", "Some iPhone mic")]) == "0"
    assert pick_microphone([]) is None


def test_pick_system_audio() -> None:
    assert pick_system_audio(["Mic", "BlackHole 2ch"]) == "BlackHole 2ch"
    assert pick_system_audio(["Mic", "virtual-audio-capturer"]) == (
        "virtual-audio-capturer"
    )
    assert pick_system_audio(["Mic", "Line In"]) is None


# ---------------------------------------------------------------------------
# Display selection / capture region / fps
# ---------------------------------------------------------------------------

def test_selected_displays_modes() -> None:
    displays = _displays()
    assert selected_displays(RecordingDisplayMode.ALL_DISPLAYS, displays, []) == displays
    one = selected_displays(RecordingDisplayMode.SELECTED_DISPLAYS, displays, ["B"])
    assert [d.id for d in one] == ["B"]
    # Empty selection falls back to the primary monitor.
    fallback = selected_displays(RecordingDisplayMode.SELECTED_DISPLAYS, displays, [])
    assert [d.id for d in fallback] == ["A"]
    # Active-window mode selects no full monitor.
    assert selected_displays(RecordingDisplayMode.ACTIVE_WINDOW, displays, []) == []


def test_displays_bounding_box() -> None:
    assert displays_bounding_box(_displays()) == (0, 0, 4480, 1440)
    assert displays_bounding_box([]) is None


def test_mode_from_value_defaults_to_all() -> None:
    assert RecordingDisplayMode.from_value("selected") is (
        RecordingDisplayMode.SELECTED_DISPLAYS
    )
    assert RecordingDisplayMode.from_value("bogus") is (
        RecordingDisplayMode.ALL_DISPLAYS
    )


def test_resolve_region_windows_all_is_bounding_box_crop() -> None:
    region = resolve_capture_region(
        RecordingDisplayMode.ALL_DISPLAYS, _displays(), [], None, platform="win32"
    )
    assert region.offset_x == 0 and region.offset_y == 0
    assert region.width == 4480 and region.height == 1440


def test_resolve_region_windows_selected_single_monitor() -> None:
    region = resolve_capture_region(
        RecordingDisplayMode.SELECTED_DISPLAYS, _displays(), ["B"], None,
        platform="win32",
    )
    assert (region.offset_x, region.width, region.height) == (2560, 1920, 1080)


def test_resolve_region_windows_active_window_uses_bounds() -> None:
    region = resolve_capture_region(
        RecordingDisplayMode.ACTIVE_WINDOW, _displays(), [], (100, 50, 801, 601),
        platform="win32",
    )
    # Odd dims are rounded down to even for libx264.
    assert (region.offset_x, region.offset_y) == (100, 50)
    assert (region.width, region.height) == (800, 600)


def test_capture_screen_indices_skips_cameras() -> None:
    from app.services.screen_recorder_service import capture_screen_indices
    video = [
        ("0", "FaceTime HD-kamera"),
        ("1", "iPhone desk view"),
        ("2", "iPhone camera"),
        ("3", "Capture screen 0"),
        ("4", "Capture screen 1"),
    ]
    # Screen devices sit after the cameras → indices 3, 4 (in screen-number order).
    assert capture_screen_indices(video) == ["3", "4"]
    assert capture_screen_indices(video[:3]) == []


def test_resolve_region_macos_all_uses_primary_av_index() -> None:
    # av_index here is the *real* avfoundation device index (post-camera).
    displays = [
        RecordingDisplayInfo("A", "Dell", 2560, 1440, True, 0, 0, av_index=3),
        RecordingDisplayInfo("B", "LG", 1920, 1080, False, 2560, 0, av_index=4),
    ]
    region = resolve_capture_region(
        RecordingDisplayMode.ALL_DISPLAYS, displays, [], None, platform="darwin"
    )
    assert region.screen_index == "3"  # primary, not the first camera index


def test_resolve_region_macos_falls_back_when_av_index_unknown() -> None:
    # No av_index → screen_index None so build_ffmpeg_args keeps the probed
    # default screen device (avoids the camera-index bug).
    displays = [RecordingDisplayInfo("A", "Dell", 2560, 1440, True, 0, 0)]
    region = resolve_capture_region(
        RecordingDisplayMode.ALL_DISPLAYS, displays, [], None, platform="darwin"
    )
    assert region.screen_index is None
    args = build_ffmpeg_args(
        "darwin",
        CaptureDevices(screen="3", microphone="2"),
        RecordingOptions(),
        "/out/seg_%05d.mp4",
        region,
    )
    assert args[args.index("-i") + 1] == "3:2"  # probed screen, not overridden


def test_resolve_region_macos_selected_uses_av_index() -> None:
    region = resolve_capture_region(
        RecordingDisplayMode.SELECTED_DISPLAYS, _displays(), ["B"], None,
        platform="darwin",
    )
    assert region.screen_index == "1"
    assert region.offset_x is None  # macOS never crops


def test_resolve_region_macos_active_window_falls_back_to_primary() -> None:
    region = resolve_capture_region(
        RecordingDisplayMode.ACTIVE_WINDOW, _displays(), [], (0, 0, 100, 100),
        platform="darwin",
    )
    # avfoundation can't capture a window → primary screen device index.
    assert region.screen_index == "0"


def test_effective_fps_reduces_for_multi_monitor() -> None:
    displays = _displays()
    assert effective_fps(
        30, RecordingDisplayMode.ALL_DISPLAYS, displays, [], multi_monitor_cap=15
    ) == 15
    # Single monitor is untouched.
    assert effective_fps(
        30, RecordingDisplayMode.SELECTED_DISPLAYS, displays, ["A"],
        multi_monitor_cap=15,
    ) == 30
    # Disabled → keep base fps even for multi-monitor.
    assert effective_fps(
        30, RecordingDisplayMode.ALL_DISPLAYS, displays, [], auto_reduce=False
    ) == 30


def test_build_args_windows_with_crop_region() -> None:
    region = CaptureRegion(offset_x=2560, offset_y=0, width=1920, height=1080)
    args = build_ffmpeg_args(
        "win32",
        CaptureDevices(screen="desktop", microphone="Mic"),
        RecordingOptions(),
        r"C:\out\seg_%05d.mp4",
        region,
    )
    assert "-offset_x" in args and args[args.index("-offset_x") + 1] == "2560"
    assert args[args.index("-video_size") + 1] == "1920x1080"
    # Crop options precede the desktop input.
    assert args.index("-video_size") < args.index("-i")


def test_build_args_windows_with_window_title() -> None:
    region = CaptureRegion(window_title="Local AI Face")
    args = build_ffmpeg_args(
        "win32",
        CaptureDevices(screen="desktop", microphone=None),
        RecordingOptions(),
        r"C:\out\seg_%05d.mp4",
        region,
    )
    assert "title=Local AI Face" in args
    assert "-video_size" not in args


def test_build_args_macos_region_overrides_screen_index() -> None:
    region = CaptureRegion(screen_index="2")
    args = build_ffmpeg_args(
        "darwin",
        CaptureDevices(screen="1", microphone="0"),
        RecordingOptions(),
        "/out/seg_%05d.mp4",
        region,
    )
    # Video:audio input uses the region's screen index, not the probed one.
    assert args[args.index("-i") + 1] == "2:0"


def test_format_display_label() -> None:
    displays = _displays()
    assert format_display_label(displays[0], 1, "monitor", "(primary)") == (
        "1. monitor (primary) Dell — 2560x1440"
    )
    assert format_display_label(displays[1], 2, "monitor", "(primary)") == (
        "2. monitor LG — 1920x1080"
    )
