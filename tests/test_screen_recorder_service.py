"""Extended tests for screen_recorder_service beyond pure ffmpeg args."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.services.screen_recorder_service import (
    AudioSource,
    AudioValidation,
    CaptureDevices,
    RecorderState,
    RecordingOptions,
    ScreenRecorderService,
    build_audio_filtergraph,
    is_macos,
    is_windows,
    list_audio_devices,
    probe_screen_indices,
    resolve_ffmpeg,
    resolve_ffprobe,
    validate_recording_audio,
    _is_loopback,
    _match_av_index,
    _match_name,
    _resolve_system_audio_macos,
)


# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------

def test_is_macos_and_windows():
    assert is_macos("darwin") is True
    assert is_macos("win32") is False
    assert is_windows("win32") is True
    assert is_windows("darwin") is False


# ---------------------------------------------------------------------------
# Audio filtergraph
# ---------------------------------------------------------------------------

def test_build_audio_filtergraph_single_source():
    graph, out = build_audio_filtergraph(
        [AudioSource("[0:a]", 1.0, "mic")],
        sample_rate=48000,
        channels=2,
    )
    assert graph is not None
    assert out == "[aout]"
    assert graph.startswith("[0:a]")
    assert "aresample=48000" in graph
    assert graph.endswith("[aout]")

def test_build_audio_filtergraph_mixes_two_sources():
    sources = [
        AudioSource("[0:a]", 0.5, "mic"),
        AudioSource("[1:a]", 1.0, "sys"),
    ]
    graph, out = build_audio_filtergraph(sources)
    assert "volume=0.5" in graph
    assert "amix=inputs=2" in graph
    assert "alimiter" in graph
    assert out == "[aout]"

def test_build_audio_filtergraph_meter_tap():
    graph, out = build_audio_filtergraph(
        [AudioSource("[0:a]", 1.0, "mic")],
        meter=True,
    )
    assert "asplit=2[aenc][amet]" in graph
    assert "ametadata=mode=print" in graph
    assert out == "[aenc]"

def test_build_audio_filtergraph_empty():
    graph, out = build_audio_filtergraph([])
    assert graph is None and out is None


# ---------------------------------------------------------------------------
# Device matching helpers
# ---------------------------------------------------------------------------

def test_match_name_exact_and_substring():
    names = ["Mic (USB)", "BlackHole 2ch"]
    assert _match_name(names, "Mic (USB)") == "Mic (USB)"
    assert _match_name(names, "blackhole") == "BlackHole 2ch"
    assert _match_name(names, "missing") is None

def test_match_av_index():
    audio = [("0", "iPhone mic"), ("2", "MacBook Air mikrofon")]
    assert _match_av_index(audio, "MacBook") == "2"
    assert _match_av_index(audio, "nope") is None

def test_resolve_system_audio_macos_disabled():
    idx, name, note = _resolve_system_audio_macos([], False, None)
    assert idx is None and note == "disabled in settings"

def test_resolve_system_audio_macos_explicit():
    audio = [("0", "Mic"), ("5", "BlackHole 2ch")]
    idx, name, note = _resolve_system_audio_macos(audio, True, "BlackHole")
    assert idx == "5"
    assert name == "BlackHole 2ch"
    assert note is None

def test_is_loopback():
    assert _is_loopback("CABLE Output (VB-Audio Virtual Cable)") is True
    assert _is_loopback("MacBook Pro Microphone") is False


# ---------------------------------------------------------------------------
# Binary resolution
# ---------------------------------------------------------------------------

def test_resolve_ffmpeg_explicit(tmp_path):
    exe = tmp_path / "ffmpeg"
    exe.write_text("stub")
    assert resolve_ffmpeg(str(exe)) == str(exe)
    assert resolve_ffmpeg(str(tmp_path / "missing")) is None or resolve_ffmpeg() is not None

def test_resolve_ffprobe_next_to_ffmpeg(tmp_path):
    ffmpeg = tmp_path / "ffmpeg"
    ffprobe = tmp_path / "ffprobe"
    ffmpeg.write_text("x")
    ffprobe.write_text("x")
    assert resolve_ffprobe(str(ffmpeg)) == str(ffprobe)


# ---------------------------------------------------------------------------
# list_audio_devices / probe_screen_indices
# ---------------------------------------------------------------------------

def test_list_audio_devices_empty_without_ffmpeg():
    assert list_audio_devices(None) == []

def test_list_audio_devices_macos(monkeypatch):
    fake = (
        "[AVFoundation] AVFoundation audio devices:\n"
        "[AVFoundation] [0] MacBook mic\n"
    )
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: type("R", (), {"stderr": fake, "stdout": ""})(),
    )
    names = list_audio_devices("ffmpeg", platform="darwin")
    assert "MacBook mic" in names

def test_probe_screen_indices_non_macos():
    assert probe_screen_indices("ffmpeg", platform="win32") == []


# ---------------------------------------------------------------------------
# validate_recording_audio
# ---------------------------------------------------------------------------

def test_validate_recording_audio_missing_ffprobe():
    result = validate_recording_audio(None, Path("/x.mp4"))
    assert result.has_audio is False
    assert result.error

def test_validate_recording_audio_missing_file(tmp_path):
    ffprobe = tmp_path / "ffprobe"
    ffprobe.write_text("stub")
    result = validate_recording_audio(str(ffprobe), tmp_path / "nope.mp4")
    assert "missing" in (result.error or "")

def test_validate_recording_audio_parses_probe_output(tmp_path, monkeypatch):
    mp4 = tmp_path / "clip.mp4"
    mp4.write_bytes(b"fake")
    ffprobe = tmp_path / "ffprobe"
    ffprobe.write_text("stub")

    class _Proc:
        returncode = 0
        stdout = (
            "[STREAM]\ncodec_type=audio\ncodec_name=aac\n"
            "duration=2.5\nbit_rate=128000\nchannels=2\n"
            "sample_rate=48000\n[/STREAM]\n"
        )
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Proc())
    result = validate_recording_audio(str(ffprobe), mp4)
    assert result.has_audio is True
    assert result.codec == "aac"


# ---------------------------------------------------------------------------
# ScreenRecorderService (Qt)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(ScreenRecorderService is None, reason="Qt unavailable")
class TestScreenRecorderService:
    def test_initial_state(self, qapp):
        svc = ScreenRecorderService(ffmpeg_path="/usr/bin/ffmpeg")
        assert svc.state is RecorderState.IDLE
        assert svc.elapsed_seconds == 0
        assert svc.output_dir is None

    def test_start_pause_resume_stop(self, qapp, qtbot, tmp_path, monkeypatch):
        svc = ScreenRecorderService(ffmpeg_path="/usr/bin/ffmpeg")
        devices = CaptureDevices(screen="0", microphone="1")
        options = RecordingOptions(fps=10, segment_seconds=2)

        recording = {"active": False}

        def _fake_spawn():
            recording["active"] = True
            svc._set_state(RecorderState.RECORDING)

        monkeypatch.setattr(svc, "_spawn_ffmpeg", _fake_spawn)
        monkeypatch.setattr(svc, "_terminate_proc", lambda: recording.update(active=False))
        monkeypatch.setattr(svc, "_validate_audio", lambda _final: None)

        states = []
        svc.state_changed.connect(lambda s: states.append(s))

        out_dir = tmp_path / "rec"
        svc.start(out_dir, devices, options, concat_on_stop=False)
        assert svc.state is RecorderState.RECORDING
        assert svc.output_dir == out_dir

        svc.pause()
        assert svc.state is RecorderState.PAUSED

        svc.resume()
        assert svc.state is RecorderState.RECORDING

        svc.stop()
        assert svc.state is RecorderState.IDLE
        assert RecorderState.RECORDING in states
        assert RecorderState.PAUSED in states

    def test_fail_when_ffmpeg_missing(self, qapp, qtbot, tmp_path):
        svc = ScreenRecorderService(ffmpeg_path=None)
        errors = []
        svc.error.connect(errors.append)
        svc.start(
            tmp_path / "out",
            CaptureDevices(screen="0"),
            RecordingOptions(),
        )
        assert svc.state is RecorderState.ERROR
        assert errors

    def test_segment_paths_sorted(self, qapp, tmp_path):
        svc = ScreenRecorderService(ffmpeg_path="/usr/bin/ffmpeg")
        out = tmp_path / "segments"
        out.mkdir()
        (out / "seg_00002_00001.mp4").write_bytes(b"a")
        (out / "seg_00001_00001.mp4").write_bytes(b"b")
        svc._output_dir = out
        paths = svc.segment_paths()
        assert [p.name for p in paths] == [
            "seg_00001_00001.mp4",
            "seg_00002_00001.mp4",
        ]

    def test_audio_validation_summary_error(self):
        v = AudioValidation(has_audio=False, error="probe failed")
        assert "failed" in v.summary()
