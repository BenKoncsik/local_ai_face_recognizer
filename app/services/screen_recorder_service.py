"""Screen recorder backed by the system ``ffmpeg`` binary.

The recorder captures the screen (with cursor), the microphone and — best
effort, where a virtual loopback device exists — the system/speaker audio,
writing the result as a sequence of short, independently playable MP4
**segments** (crash protection).  When recording stops the segments are
optionally concatenated into a single ``recording.mp4``.

Design notes
------------
* ffmpeg is driven through :class:`~PySide6.QtCore.QProcess` so it integrates
  with the Qt event loop (``finished`` / ``errorOccurred`` signals, no polling).
* Pause/resume is implemented by stopping the current ffmpeg process (which
  finalizes its open segment) and starting a fresh one that keeps numbering
  segments in the same directory.  The elapsed clock excludes paused time, so
  the video and the :class:`RecordingTimelineLog` stay in sync.
* The argument-building / device-parsing helpers below are plain module-level
  functions with no Qt dependency, so they can be unit tested without ffmpeg.
"""

from __future__ import annotations

import logging
import re
import shutil
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure helpers (no Qt / no subprocess) — unit tested
# ---------------------------------------------------------------------------

# crf: lower = better quality/bigger; scale_height: output height (px, -2 keeps
# aspect with even width); audio_bitrate: AAC bitrate.
_QUALITY_PRESETS: Dict[str, Dict[str, object]] = {
    "low":    {"crf": 30, "scale_height": 540,  "audio_bitrate": "64k"},
    "normal": {"crf": 26, "scale_height": 720,  "audio_bitrate": "96k"},
    "better": {"crf": 22, "scale_height": 1080, "audio_bitrate": "128k"},
}


def quality_preset(name: str) -> Dict[str, object]:
    """Return the encoder parameters for a named quality preset.

    Unknown names fall back to ``"normal"``.
    """
    return dict(_QUALITY_PRESETS.get(name, _QUALITY_PRESETS["normal"]))


@dataclass
class CaptureDevices:
    """Resolved capture device identifiers for the current platform.

    ``screen`` is the avfoundation video index (macOS) or ``"desktop"``
    (Windows gdigrab).  Audio fields are ``None`` when unavailable.
    """

    screen: str
    microphone: Optional[str] = None
    system_audio: Optional[str] = None


@dataclass
class RecordingOptions:
    """Encoding/capture options derived from :class:`RecordingConfig`."""

    fps: int = 18
    quality: str = "normal"
    segment_seconds: int = 8
    capture_cursor: bool = True


def is_macos(platform: str) -> bool:
    return platform.startswith("darwin")


def is_windows(platform: str) -> bool:
    return platform.startswith("win")


def build_ffmpeg_args(
    platform: str,
    devices: CaptureDevices,
    options: RecordingOptions,
    segment_pattern: str,
) -> List[str]:
    """Build the ffmpeg argument list (excluding the binary itself).

    Produces a segmented MP4 capture.  ``segment_pattern`` is an ffmpeg output
    pattern such as ``/out/seg_%05d.mp4``.
    """
    preset = quality_preset(options.quality)
    crf = preset["crf"]
    height = preset["scale_height"]
    audio_bitrate = preset["audio_bitrate"]

    args: List[str] = ["-hide_banner", "-loglevel", "warning", "-y"]

    # Count audio inputs so we know whether to mix.
    audio_inputs: List[str] = []

    if is_macos(platform):
        cursor = "1" if options.capture_cursor else "0"
        video_audio = devices.screen
        if devices.microphone is not None:
            video_audio = f"{devices.screen}:{devices.microphone}"
            audio_inputs.append("mic")
        args += [
            "-f", "avfoundation",
            "-capture_cursor", cursor,
            "-framerate", str(options.fps),
            "-i", video_audio,
        ]
        if devices.system_audio is not None:
            args += ["-f", "avfoundation", "-i", f":{devices.system_audio}"]
            audio_inputs.append("sys")
    elif is_windows(platform):
        draw_mouse = "1" if options.capture_cursor else "0"
        args += [
            "-f", "gdigrab",
            "-draw_mouse", draw_mouse,
            "-framerate", str(options.fps),
            "-i", "desktop",
        ]
        if devices.microphone is not None:
            args += ["-f", "dshow", "-i", f"audio={devices.microphone}"]
            audio_inputs.append("mic")
        if devices.system_audio is not None:
            args += ["-f", "dshow", "-i", f"audio={devices.system_audio}"]
            audio_inputs.append("sys")
    else:
        raise ValueError(f"unsupported platform for recording: {platform!r}")

    # Video encode (shared).  Force a keyframe at every segment boundary —
    # the segment muxer can only cut on keyframes, so without this the output
    # would not actually split into short files (breaking crash protection).
    args += [
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-vf", f"scale=-2:{height}",
        "-r", str(options.fps),
        "-g", str(max(1, options.fps * options.segment_seconds)),
        "-force_key_frames",
        f"expr:gte(t,n_forced*{options.segment_seconds})",
    ]

    # Audio mapping.  On macOS the mic shares input 0 with the video; the
    # system-audio device (when present) is a separate input.  On Windows each
    # audio device is its own input.  When two audio sources exist we mix them.
    if len(audio_inputs) >= 2:
        if is_macos(platform):
            mix = "[0:a][1:a]amix=inputs=2:duration=longest[aout]"
        else:  # windows: video=0, mic=1, sys=2
            mix = "[1:a][2:a]amix=inputs=2:duration=longest[aout]"
        args += [
            "-filter_complex", mix,
            "-map", "0:v",
            "-map", "[aout]",
            "-c:a", "aac",
            "-b:a", str(audio_bitrate),
        ]
    elif len(audio_inputs) == 1:
        # Single audio source: explicit mapping keeps Windows (separate input)
        # and macOS (audio on input 0) both correct.
        audio_index = 0 if is_macos(platform) else 1
        args += [
            "-map", "0:v",
            "-map", f"{audio_index}:a",
            "-c:a", "aac",
            "-b:a", str(audio_bitrate),
        ]
    else:
        args += ["-map", "0:v", "-an"]

    # Segment muxer — each closed segment is an independently playable file.
    args += [
        "-f", "segment",
        "-segment_time", str(options.segment_seconds),
        "-reset_timestamps", "1",
        "-segment_format", "mp4",
        segment_pattern,
    ]
    return args


def build_concat_list(segment_paths: List[Path]) -> str:
    """Build the body of an ffmpeg concat-demuxer list file.

    Each line is ``file '<absolute-path>'`` with single quotes escaped.
    """
    lines = []
    for p in segment_paths:
        escaped = str(p).replace("'", r"'\''")
        lines.append(f"file '{escaped}'")
    return "\n".join(lines) + ("\n" if lines else "")


def build_concat_args(list_file: Path, output_file: Path) -> List[str]:
    """Build ffmpeg args that stream-copy a concat list into one file."""
    return [
        "-hide_banner", "-loglevel", "warning", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(output_file),
    ]


def parse_avfoundation_devices(stderr_text: str) -> Dict[str, List[tuple]]:
    """Parse ``ffmpeg -f avfoundation -list_devices true`` stderr output.

    Returns ``{"video": [(index, name), ...], "audio": [(index, name), ...]}``.
    """
    video: List[tuple] = []
    audio: List[tuple] = []
    bucket: Optional[List[tuple]] = None
    line_re = re.compile(r"\[(\d+)\]\s+(.*)$")
    for raw in stderr_text.splitlines():
        low = raw.lower()
        if "avfoundation video devices" in low:
            bucket = video
            continue
        if "avfoundation audio devices" in low:
            bucket = audio
            continue
        if bucket is None:
            continue
        m = line_re.search(raw.strip())
        if m:
            bucket.append((m.group(1), m.group(2).strip()))
    return {"video": video, "audio": audio}


def parse_dshow_audio_devices(stderr_text: str) -> List[str]:
    """Parse ``ffmpeg -f dshow -list_devices true`` stderr for audio names."""
    names: List[str] = []
    in_audio = False
    name_re = re.compile(r'"([^"]+)"')
    for raw in stderr_text.splitlines():
        low = raw.lower()
        if "audio devices" in low:
            in_audio = True
            continue
        if "video devices" in low:
            in_audio = False
            continue
        if in_audio:
            m = name_re.search(raw)
            if m:
                names.append(m.group(1))
    return names


# Heuristic names that identify a virtual loopback (system-audio) device.
_LOOPBACK_HINTS = (
    "blackhole",
    "loopback",
    "soundflower",
    "virtual-audio-capturer",
    "stereo mix",
    "what u hear",
)


def pick_system_audio(candidates: List[str]) -> Optional[str]:
    """Return the first candidate device name that looks like a loopback."""
    for name in candidates:
        low = name.lower()
        if any(hint in low for hint in _LOOPBACK_HINTS):
            return name
    return None


# ---------------------------------------------------------------------------
# Qt-backed recorder service
# ---------------------------------------------------------------------------

class RecorderState(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"
    FINALIZING = "finalizing"
    ERROR = "error"


def resolve_ffmpeg(explicit_path: Optional[str] = None) -> Optional[str]:
    """Return a usable ffmpeg path, or ``None`` if it cannot be found."""
    if explicit_path:
        p = Path(explicit_path).expanduser()
        if p.exists():
            return str(p)
    found = shutil.which("ffmpeg")
    return found


def probe_devices(
    ffmpeg_path: str,
    platform: str = sys.platform,
    *,
    want_system_audio: bool = True,
) -> CaptureDevices:
    """Probe ffmpeg for the screen + microphone (+ best-effort loopback).

    Runs ``ffmpeg -list_devices true`` and parses stderr.  Falls back to
    sensible defaults when probing yields nothing.  Never raises — capture
    can still be attempted with the defaults.
    """
    import subprocess  # local import: only needed at probe time

    def _list(args: List[str]) -> str:
        try:
            proc = subprocess.run(
                [ffmpeg_path, *args],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return (proc.stderr or "") + (proc.stdout or "")
        except Exception as exc:  # noqa: BLE001
            log.warning("device probe failed: %s", exc)
            return ""

    if is_macos(platform):
        text = _list(["-f", "avfoundation", "-list_devices", "true", "-i", ""])
        parsed = parse_avfoundation_devices(text)
        screen = next(
            (idx for idx, name in parsed["video"] if "screen" in name.lower()),
            "1",  # avfoundation screen is commonly index 1
        )
        audio = parsed["audio"]
        mic = audio[0][0] if audio else "0"
        sys_audio = None
        if want_system_audio:
            loop_name = pick_system_audio([name for _, name in audio])
            if loop_name is not None:
                sys_audio = next(
                    (idx for idx, name in audio if name == loop_name), None
                )
        return CaptureDevices(screen=screen, microphone=mic, system_audio=sys_audio)

    if is_windows(platform):
        text = _list(["-f", "dshow", "-list_devices", "true", "-i", "dummy"])
        names = parse_dshow_audio_devices(text)
        mic = next(
            (n for n in names if not _is_loopback(n)),
            names[0] if names else None,
        )
        sys_audio = pick_system_audio(names) if want_system_audio else None
        return CaptureDevices(
            screen="desktop", microphone=mic, system_audio=sys_audio
        )

    return CaptureDevices(screen="0")


def _is_loopback(name: str) -> bool:
    low = name.lower()
    return any(hint in low for hint in _LOOPBACK_HINTS)


try:  # Qt is optional for importing the pure helpers (e.g. in unit tests).
    from PySide6.QtCore import QObject, QProcess, QTimer, Signal

    class ScreenRecorderService(QObject):
        """Lifecycle controller for a segmented ffmpeg screen recording."""

        state_changed = Signal(object)   # RecorderState
        elapsed_changed = Signal(int)    # whole seconds, excluding pauses
        error = Signal(str)

        def __init__(self, ffmpeg_path: Optional[str], parent=None) -> None:
            super().__init__(parent)
            self._ffmpeg = ffmpeg_path
            self._state = RecorderState.IDLE
            self._proc: Optional[QProcess] = None
            self._output_dir: Optional[Path] = None
            self._devices: Optional[CaptureDevices] = None
            self._options = RecordingOptions()
            self._concat_on_stop = True
            self._segment_index = 0
            self._elapsed_seconds = 0
            self._timer = QTimer(self)
            self._timer.setInterval(1000)
            self._timer.timeout.connect(self._on_tick)
            self._stopping = False

        # -- public API -------------------------------------------------

        @property
        def state(self) -> RecorderState:
            return self._state

        @property
        def elapsed_seconds(self) -> int:
            return self._elapsed_seconds

        @property
        def output_dir(self) -> Optional[Path]:
            return self._output_dir

        def start(
            self,
            output_dir: Path,
            devices: CaptureDevices,
            options: RecordingOptions,
            concat_on_stop: bool = True,
        ) -> None:
            if self._state in (RecorderState.RECORDING, RecorderState.PAUSED):
                return
            self._output_dir = Path(output_dir)
            self._output_dir.mkdir(parents=True, exist_ok=True)
            self._devices = devices
            self._options = options
            self._concat_on_stop = concat_on_stop
            self._segment_index = 0
            self._elapsed_seconds = 0
            self._stopping = False
            self._spawn_ffmpeg()
            if self._state is RecorderState.RECORDING:
                self._timer.start()

        def pause(self) -> None:
            if self._state is not RecorderState.RECORDING:
                return
            self._timer.stop()
            self._set_state(RecorderState.PAUSED)
            self._terminate_proc()

        def resume(self) -> None:
            if self._state is not RecorderState.PAUSED:
                return
            self._spawn_ffmpeg()
            if self._state is RecorderState.RECORDING:
                self._timer.start()

        def stop(self) -> Optional[Path]:
            """Stop recording, optionally concatenate, return final mp4 path."""
            if self._state in (RecorderState.IDLE, RecorderState.FINALIZING):
                return None
            self._timer.stop()
            self._stopping = True
            self._set_state(RecorderState.FINALIZING)
            self._terminate_proc()
            final = None
            if self._concat_on_stop:
                final = self._concatenate_segments()
            self._set_state(RecorderState.IDLE)
            self._stopping = False
            return final

        def segment_paths(self) -> List[Path]:
            if self._output_dir is None:
                return []
            return sorted(self._output_dir.glob("seg_*.mp4"))

        # -- internals --------------------------------------------------

        def _segment_pattern(self) -> str:
            assert self._output_dir is not None
            # Continue numbering across pause/resume so concat order is right.
            return str(self._output_dir / f"seg_{self._segment_index:05d}_%05d.mp4")

        def _spawn_ffmpeg(self) -> None:
            if not self._ffmpeg:
                self._fail("ffmpeg not available")
                return
            assert self._devices is not None and self._output_dir is not None
            try:
                args = build_ffmpeg_args(
                    sys.platform,
                    self._devices,
                    self._options,
                    self._segment_pattern(),
                )
            except ValueError as exc:
                self._fail(str(exc))
                return

            self._segment_index += 1
            proc = QProcess(self)
            proc.setProgram(self._ffmpeg)
            proc.setArguments(args)
            proc.errorOccurred.connect(self._on_proc_error)
            proc.finished.connect(self._on_proc_finished)
            proc.start()
            if not proc.waitForStarted(3000):
                self._fail("ffmpeg failed to start")
                return
            self._proc = proc
            self._set_state(RecorderState.RECORDING)
            log.info("recording started: %s %s", self._ffmpeg, " ".join(args))

        def _terminate_proc(self) -> None:
            proc = self._proc
            if proc is None:
                return
            self._proc = None
            try:
                # Ask ffmpeg to quit cleanly so it finalizes the open segment.
                proc.write(b"q")
                proc.closeWriteChannel()
                if not proc.waitForFinished(5000):
                    proc.terminate()
                    if not proc.waitForFinished(3000):
                        proc.kill()
                        proc.waitForFinished(2000)
            except Exception:  # noqa: BLE001
                proc.kill()

        def _concatenate_segments(self) -> Optional[Path]:
            segs = self.segment_paths()
            if not segs or self._output_dir is None or not self._ffmpeg:
                return None
            list_file = self._output_dir / "segments.txt"
            out_file = self._output_dir / "recording.mp4"
            try:
                list_file.write_text(build_concat_list(segs), encoding="utf-8")
            except OSError as exc:
                log.warning("could not write concat list: %s", exc)
                return None
            proc = QProcess(self)
            proc.setProgram(self._ffmpeg)
            proc.setArguments(build_concat_args(list_file, out_file))
            proc.start()
            if not proc.waitForFinished(60000):
                log.warning("concat timed out; segments are preserved")
                return None
            if proc.exitCode() != 0 or not out_file.exists():
                log.warning("concat failed (code %s); segments preserved", proc.exitCode())
                return None
            return out_file

        def _on_tick(self) -> None:
            self._elapsed_seconds += 1
            self.elapsed_changed.emit(self._elapsed_seconds)

        def _on_proc_error(self, _err) -> None:
            if self._stopping or self._state is RecorderState.PAUSED:
                return
            self._fail("ffmpeg process error")

        def _on_proc_finished(self, _code, _status) -> None:
            # Unexpected exit while we believed we were recording.
            if (
                self._state is RecorderState.RECORDING
                and not self._stopping
                and self._proc is not None
            ):
                self._fail("ffmpeg exited unexpectedly")

        def _fail(self, message: str) -> None:
            self._timer.stop()
            log.error("recorder error: %s", message)
            self._set_state(RecorderState.ERROR)
            self.error.emit(message)

        def _set_state(self, state: RecorderState) -> None:
            if state is self._state:
                return
            self._state = state
            self.state_changed.emit(state)

except ImportError:  # pragma: no cover - Qt missing (pure-helper unit tests)
    ScreenRecorderService = None  # type: ignore
