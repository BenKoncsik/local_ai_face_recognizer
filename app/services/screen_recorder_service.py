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
    (Windows gdigrab).  Audio fields are ``None`` when unavailable.  The
    ``*_name`` fields carry the human-readable device names purely for
    diagnostics/logging (the ``microphone``/``system_audio`` ids are what
    actually get passed to ffmpeg).
    """

    screen: str
    microphone: Optional[str] = None
    system_audio: Optional[str] = None
    microphone_name: Optional[str] = None
    system_audio_name: Optional[str] = None
    # Why a system-audio device could not be found (no loopback installed,
    # disabled by config, …) — surfaced in diagnostics so "no system sound"
    # is never silent.
    system_audio_note: Optional[str] = None


@dataclass
class RecordingOptions:
    """Encoding/capture options derived from :class:`RecordingConfig`."""

    fps: int = 18
    quality: str = "normal"
    segment_seconds: int = 8
    capture_cursor: bool = True
    # Audio mix controls.  Volumes are linear gain multipliers (1.0 = unity);
    # mutes drop the source from the mix entirely.  Output is always coerced to
    # ``audio_sample_rate`` / ``audio_channels`` (48 kHz stereo by default).
    mic_volume: float = 1.0
    system_volume: float = 1.0
    mute_microphone: bool = False
    mute_system_audio: bool = False
    audio_sample_rate: int = 48000
    audio_channels: int = 2
    # Emit live peak-level metering on ffmpeg's stdout (drives the VU meter).
    meter_audio: bool = False


class RecordingDisplayMode(Enum):
    """Which part of the screen to record."""

    ACTIVE_WINDOW = "active_window"   # only the application window
    ALL_DISPLAYS = "all"             # every monitor on one canvas
    SELECTED_DISPLAYS = "selected"   # a user-chosen subset of monitors

    @classmethod
    def from_value(cls, value: object) -> "RecordingDisplayMode":
        """Coerce a stored string into a mode, defaulting to ALL_DISPLAYS."""
        for mode in cls:
            if mode.value == value:
                return mode
        return cls.ALL_DISPLAYS


@dataclass
class RecordingDisplayInfo:
    """A monitor as reported by the windowing system.

    ``av_index`` is the macOS avfoundation "Capture screen N" index (best-effort
    ordinal mapping); it is ``None`` on other platforms / when unknown.
    """

    id: str
    name: str
    width: int
    height: int
    is_primary: bool = False
    x: int = 0
    y: int = 0
    av_index: Optional[int] = None


@dataclass
class CaptureRegion:
    """Resolved capture target for a given display mode + platform.

    Exactly one capture strategy is expressed:

    * ``screen_index`` — a macOS avfoundation video device index.
    * ``offset_x/offset_y/width/height`` — a Windows ``gdigrab`` crop rectangle.
    * ``window_title`` — a Windows ``gdigrab`` single-window capture.

    All ``None`` means "platform default" (macOS: probed screen device;
    Windows: the whole virtual desktop).
    """

    screen_index: Optional[str] = None
    offset_x: Optional[int] = None
    offset_y: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    window_title: Optional[str] = None


def is_macos(platform: str) -> bool:
    return platform.startswith("darwin")


def is_windows(platform: str) -> bool:
    return platform.startswith("win")


def format_display_label(
    info: RecordingDisplayInfo,
    ordinal: int,
    monitor_word: str = "monitor",
    primary_marker: str = "(primary)",
) -> str:
    """Human-readable monitor label, e.g. ``"1. monitor (primary) — 2560x1440"``."""
    primary = f" {primary_marker}" if info.is_primary else ""
    name = f" {info.name}" if info.name else ""
    return f"{ordinal}. {monitor_word}{primary}{name} — {info.width}x{info.height}"


def _even(value: int) -> int:
    """Round *value* down to the nearest even integer (libx264 needs even dims)."""
    v = max(2, int(value))
    return v - (v % 2)


def _primary_display(
    displays: List[RecordingDisplayInfo],
) -> Optional[RecordingDisplayInfo]:
    if not displays:
        return None
    return next((d for d in displays if d.is_primary), displays[0])


def selected_displays(
    mode: RecordingDisplayMode,
    displays: List[RecordingDisplayInfo],
    selected_ids: List[str],
) -> List[RecordingDisplayInfo]:
    """Return the monitors a non-window mode would capture.

    Falls back to the primary monitor when a SELECTED mode has no valid ids.
    """
    if mode is RecordingDisplayMode.ALL_DISPLAYS:
        return list(displays)
    if mode is RecordingDisplayMode.SELECTED_DISPLAYS:
        wanted = set(selected_ids or [])
        chosen = [d for d in displays if d.id in wanted]
        if chosen:
            return chosen
        primary = _primary_display(displays)
        return [primary] if primary else []
    # ACTIVE_WINDOW captures no full monitor.
    return []


def displays_bounding_box(
    displays: List[RecordingDisplayInfo],
) -> Optional[tuple]:
    """Return ``(x, y, width, height)`` spanning *displays*, or ``None``."""
    if not displays:
        return None
    x0 = min(d.x for d in displays)
    y0 = min(d.y for d in displays)
    x1 = max(d.x + d.width for d in displays)
    y1 = max(d.y + d.height for d in displays)
    return (x0, y0, x1 - x0, y1 - y0)


def resolve_capture_region(
    mode: RecordingDisplayMode,
    displays: List[RecordingDisplayInfo],
    selected_ids: List[str],
    active_window_bounds: Optional[tuple],
    platform: str = sys.platform,
) -> CaptureRegion:
    """Resolve the capture rectangle/device for *mode* on *platform*.

    ``active_window_bounds`` is ``(x, y, w, h)`` of the app window (used only by
    Windows ACTIVE_WINDOW; macOS avfoundation cannot crop to a window, so it
    falls back to the primary screen device — the caller logs that limitation).
    """
    if not isinstance(mode, RecordingDisplayMode):
        mode = RecordingDisplayMode.from_value(mode)
    mac = is_macos(platform)
    win = is_windows(platform)

    if mode is RecordingDisplayMode.ACTIVE_WINDOW:
        if win:
            if active_window_bounds:
                x, y, w, h = active_window_bounds
                return CaptureRegion(
                    offset_x=int(x), offset_y=int(y),
                    width=_even(w), height=_even(h),
                )
            return CaptureRegion()  # no bounds → whole desktop fallback
        if mac:
            primary = _primary_display(displays)
            idx = (
                str(primary.av_index)
                if primary is not None and primary.av_index is not None
                else None
            )
            return CaptureRegion(screen_index=idx)
        return CaptureRegion()

    chosen = selected_displays(mode, displays, selected_ids)
    if win:
        box = displays_bounding_box(chosen)
        if box is None:
            return CaptureRegion()  # whole desktop
        x, y, w, h = box
        return CaptureRegion(
            offset_x=int(x), offset_y=int(y), width=_even(w), height=_even(h)
        )
    if mac:
        # avfoundation records a single screen device.  For ALL_DISPLAYS there
        # is no merged canvas, so capture the primary monitor; for SELECTED take
        # the first chosen monitor.  ``screen_index`` stays None when the
        # avfoundation index is unknown — the caller then keeps the probed
        # default screen device (build_ffmpeg_args falls back to devices.screen).
        if mode is RecordingDisplayMode.ALL_DISPLAYS:
            target = _primary_display(chosen)
        else:
            target = next((d for d in chosen if d.av_index is not None), None)
        idx = (
            str(target.av_index)
            if target is not None and target.av_index is not None
            else None
        )
        return CaptureRegion(screen_index=idx)
    return CaptureRegion()


def effective_fps(
    base_fps: int,
    mode: RecordingDisplayMode,
    displays: List[RecordingDisplayInfo],
    selected_ids: List[str],
    *,
    auto_reduce: bool = True,
    multi_monitor_cap: int = 15,
) -> int:
    """Cap the frame rate when capturing more than one monitor.

    Multi-monitor captures produce a large canvas; dropping to a lower fps keeps
    the file size sane.  Single-monitor / active-window captures are unaffected.
    """
    if not auto_reduce or mode is RecordingDisplayMode.ACTIVE_WINDOW:
        return base_fps
    if len(selected_displays(mode, displays, selected_ids)) > 1:
        return max(1, min(base_fps, multi_monitor_cap))
    return base_fps


@dataclass
class AudioSource:
    """One audio leg feeding the mixer.

    ``label`` is the ffmpeg input pad (e.g. ``"[0:a]"``); ``volume`` is the
    linear gain; ``kind`` is ``"mic"`` or ``"sys"`` (diagnostics only).
    """

    label: str
    volume: float
    kind: str


# Peak-level metering tap appended to the mix bus.  ``astats`` recomputes the
# peak every ~0.1 s and ``ametadata`` prints it to ffmpeg's *stdout* (``file=-``)
# independently of ``-loglevel`` so the GUI can drive a live VU meter.  The
# parser looks for ``METER_KEY=<dBFS>`` lines.
METER_KEY = "lavfi.astats.Overall.Peak_level"
_METER_TAP = (
    "astats=metadata=1:reset=1:length=0.1,"
    f"ametadata=mode=print:key={METER_KEY}:file=-,"
    "anullsink"
)


def build_audio_filtergraph(
    sources: List[AudioSource],
    *,
    sample_rate: int = 48000,
    channels: int = 2,
    meter: bool = False,
) -> tuple:
    """Build the ``-filter_complex`` graph for the audio mix.

    Returns ``(graph, out_label)`` where *graph* is the filter_complex string
    (or ``None`` when there is no audio) and *out_label* is the pad to map into
    the encoder (``"[aout]"`` normally, ``"[aenc]"`` when *meter* splits off a
    metering tap).

    Each source is volume-adjusted, resampled to *sample_rate* and coerced to a
    *channels*-channel layout; two or more sources are mixed with ``amix`` and
    then run through ``alimiter`` so summed peaks cannot clip.
    """
    if not sources:
        return None, None

    layout = "stereo" if channels == 2 else "mono"
    legs: List[str] = []
    mixed_labels: List[str] = []
    for i, src in enumerate(sources):
        chain = [
            f"aresample={sample_rate}",
            f"aformat=sample_fmts=fltp:channel_layouts={layout}",
        ]
        if abs(src.volume - 1.0) > 1e-3:
            chain.insert(0, f"volume={src.volume:g}")
        out = f"[a{i}]"
        legs.append(f"{src.label}{','.join(chain)}{out}")
        mixed_labels.append(out)

    if len(sources) == 1:
        # Rename the single leg's output pad to the mix-bus label.
        legs[0] = legs[0][: -len(mixed_labels[0])] + "[aout]"
    else:
        # normalize=0 keeps user volumes intact; alimiter prevents the summed
        # signal from clipping past full scale.
        legs.append(
            f"{''.join(mixed_labels)}"
            f"amix=inputs={len(sources)}:duration=longest:normalize=0,"
            "alimiter=limit=0.95[aout]"
        )

    out_label = "[aout]"
    if meter:
        legs.append(f"[aout]asplit=2[aenc][amet];[amet]{_METER_TAP}")
        out_label = "[aenc]"
    return ";".join(legs), out_label


def parse_meter_peak_db(text: str) -> Optional[float]:
    """Extract the most recent peak level (dBFS) from metering stdout, if any."""
    last: Optional[float] = None
    for line in text.splitlines():
        if line.startswith(METER_KEY):
            _, _, value = line.partition("=")
            try:
                last = float(value.strip())
            except ValueError:
                continue
    return last


def build_ffmpeg_args(
    platform: str,
    devices: CaptureDevices,
    options: RecordingOptions,
    segment_pattern: str,
    region: Optional[CaptureRegion] = None,
) -> List[str]:
    """Build the ffmpeg argument list (excluding the binary itself).

    Produces a segmented MP4 capture.  ``segment_pattern`` is an ffmpeg output
    pattern such as ``/out/seg_%05d.mp4``.  ``region`` selects which monitor /
    window / desktop crop to capture; ``None`` keeps the platform default
    (macOS: the probed screen device; Windows: the whole virtual desktop).
    """
    preset = quality_preset(options.quality)
    crf = preset["crf"]
    height = preset["scale_height"]
    audio_bitrate = preset["audio_bitrate"]

    args: List[str] = ["-hide_banner", "-loglevel", "warning", "-y"]

    # Resolve which audio sources are actually mixed in.  A muted source is
    # dropped before opening it so we neither capture nor pay for it.
    use_mic = devices.microphone is not None and not options.mute_microphone
    use_sys = devices.system_audio is not None and not options.mute_system_audio

    # Audio legs feeding the mixer, with the ffmpeg input pad each maps to.
    audio_sources: List[AudioSource] = []
    input_index = 0  # advances for every ``-i`` we append

    if is_macos(platform):
        cursor = "1" if options.capture_cursor else "0"
        # A region may override which avfoundation screen device is captured.
        screen = devices.screen
        if region is not None and region.screen_index:
            screen = region.screen_index
        # On avfoundation the microphone shares input 0 with the screen video.
        video_audio = f"{screen}:{devices.microphone}" if use_mic else screen
        args += [
            "-f", "avfoundation",
            "-capture_cursor", cursor,
            "-framerate", str(options.fps),
            "-i", video_audio,
        ]
        if use_mic:
            audio_sources.append(
                AudioSource(f"[{input_index}:a]", options.mic_volume, "mic")
            )
        input_index += 1
        if use_sys:
            args += ["-f", "avfoundation", "-i", f":{devices.system_audio}"]
            audio_sources.append(
                AudioSource(f"[{input_index}:a]", options.system_volume, "sys")
            )
            input_index += 1
    elif is_windows(platform):
        draw_mouse = "1" if options.capture_cursor else "0"
        # gdigrab input options (crop offset / size) must precede ``-i``.
        grab: List[str] = [
            "-f", "gdigrab",
            "-draw_mouse", draw_mouse,
            "-framerate", str(options.fps),
        ]
        if region is not None and region.window_title:
            args += grab + ["-i", f"title={region.window_title}"]
        elif region is not None and region.width and region.height:
            args += grab + [
                "-offset_x", str(region.offset_x or 0),
                "-offset_y", str(region.offset_y or 0),
                "-video_size", f"{region.width}x{region.height}",
                "-i", "desktop",
            ]
        else:
            args += grab + ["-i", "desktop"]
        input_index += 1  # gdigrab desktop is input 0 (video only)
        if use_mic:
            args += ["-f", "dshow", "-i", f"audio={devices.microphone}"]
            audio_sources.append(
                AudioSource(f"[{input_index}:a]", options.mic_volume, "mic")
            )
            input_index += 1
        if use_sys:
            args += ["-f", "dshow", "-i", f"audio={devices.system_audio}"]
            audio_sources.append(
                AudioSource(f"[{input_index}:a]", options.system_volume, "sys")
            )
            input_index += 1
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

    # Audio mapping.  Every present source is volume-adjusted, resampled and
    # (when >1) mixed with clipping protection into a single ``[aout]`` bus that
    # is encoded as one AAC track at the configured sample-rate/channels.  When
    # there is no audio at all we mux video only (``-an``).
    graph, out_label = build_audio_filtergraph(
        audio_sources,
        sample_rate=options.audio_sample_rate,
        channels=options.audio_channels,
        meter=options.meter_audio,
    )
    if graph is not None:
        args += [
            "-filter_complex", graph,
            "-map", "0:v",
            "-map", out_label,
            "-c:a", "aac",
            "-b:a", str(audio_bitrate),
            "-ar", str(options.audio_sample_rate),
            "-ac", str(options.audio_channels),
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


def capture_screen_indices(video_devices: List[tuple]) -> List[str]:
    """avfoundation device indices of the "Capture screen N" entries, in order.

    ``video_devices`` is ``[(index, name), ...]`` as returned by
    :func:`parse_avfoundation_devices`.  Screen-capture devices are listed
    *after* cameras (e.g. ``[3] Capture screen 0``), so their device index is
    **not** the monitor ordinal — this maps monitor order → real device index.
    """
    screens = [
        (idx, name)
        for idx, name in video_devices
        if "capture screen" in name.lower()
    ]

    def _screen_num(name: str) -> int:
        m = re.search(r"(\d+)\s*$", name)
        return int(m.group(1)) if m else 0

    screens.sort(key=lambda p: _screen_num(p[1]))
    return [idx for idx, _ in screens]


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
        # New FFmpeg
        if "(audio)" in low:
            m = name_re.search(raw)
            if m:
                names.append(m.group(1))
            continue
        # Old FFmpeg
        if in_audio:
            m = name_re.search(raw)
            if m:
                names.append(m.group(1))
    return names


# Heuristic names that identify a virtual loopback (system-audio) device.
# Includes localized "Stereo Mix" variants — on a non-English Windows the
# device is renamed (e.g. Hungarian "Sztereó keverő"), so matching only the
# English string would silently skip a perfectly usable loopback.
_LOOPBACK_HINTS = (
    "blackhole",
    "loopback",
    "soundflower",
    "virtual-audio-capturer",
    "virtual audio",
    "vb-audio",
    "voicemeeter",
    "cable output",
    "what u hear",
    "what you hear",
    "wave out mix",
    "wave out",
    # "Stereo Mix" across locales (en / hu / de / fr / es / it / nl / pl / …).
    "stereo mix",
    "stereomix",
    "sztereó keverő",
    "szteró keverő",
    "sztereó keverés",
    "hangkeverő",
    "stereomischung",
    "mixage stéréo",
    "mezcla estéreo",
    "missaggio stereo",
    "stereo-mix",
)


def pick_system_audio(candidates: List[str]) -> Optional[str]:
    """Return the first candidate device name that looks like a loopback."""
    for name in candidates:
        low = name.lower()
        if any(hint in low for hint in _LOOPBACK_HINTS):
            return name
    return None


# Continuity / wireless devices that frequently fail to deliver samples when
# picked as the default capture mic (they may be asleep or off-network).
_FLAKY_MIC_HINTS = ("iphone", "ipad", "apple watch", "continuity")
# Names that indicate a reliable built-in / wired microphone.
_PREFERRED_MIC_HINTS = ("macbook", "built-in", "built in", "internal", "imac")


def pick_microphone(audio: List[tuple]) -> Optional[str]:
    """Choose the best microphone index from ``[(index, name), ...]``.

    Prefers a built-in mic, then any non-Continuity device, and only falls back
    to a flaky wireless/Continuity device (e.g. an iPhone mic) when nothing
    better exists — those often deliver no audio samples and can stall ffmpeg.
    """
    if not audio:
        return None
    for idx, name in audio:
        if any(h in name.lower() for h in _PREFERRED_MIC_HINTS):
            return idx
    for idx, name in audio:
        if not any(h in name.lower() for h in _FLAKY_MIC_HINTS):
            return idx
    return audio[0][0]


def audio_diagnostics(
    devices: CaptureDevices,
    options: RecordingOptions,
    platform: str = sys.platform,
) -> List[str]:
    """Build the ``[Audio] …`` diagnostic lines for *devices*/*options*.

    Returned as a list so callers can log each line; this makes a silent
    "no audio" outcome explainable (which devices were chosen, what was
    skipped and why, the mix format).
    """
    lines: List[str] = []
    plat = "macOS avfoundation" if is_macos(platform) else (
        "Windows dshow" if is_windows(platform) else platform
    )
    lines.append(f"[Audio] Platform: {plat}")

    if devices.microphone is not None:
        name = devices.microphone_name or devices.microphone
        if options.mute_microphone:
            lines.append(f"[Audio] Microphone MUTED (would use: {name})")
        else:
            lines.append(
                f"[Audio] Microphone capture: {name} "
                f"(volume {options.mic_volume:g})"
            )
    else:
        lines.append("[Audio] Microphone: none detected / disabled")

    if devices.system_audio is not None:
        name = devices.system_audio_name or devices.system_audio
        if options.mute_system_audio:
            lines.append(f"[Audio] System audio MUTED (would use: {name})")
        else:
            lines.append(
                f"[Audio] System audio capture: {name} "
                f"(volume {options.system_volume:g})"
            )
    else:
        note = devices.system_audio_note or "no loopback device available"
        lines.append(f"[Audio] System audio: NOT captured — {note}")

    n = sum(
        1
        for present, muted in (
            (devices.microphone is not None, options.mute_microphone),
            (devices.system_audio is not None, options.mute_system_audio),
        )
        if present and not muted
    )
    if n >= 2:
        lines.append("[Audio] Mixer initialized: 2 sources → 1 track")
    elif n == 1:
        lines.append("[Audio] Mixer initialized: 1 source → 1 track")
    else:
        lines.append("[Audio] Mixer initialized: NO audio sources → silent video")
    lines.append(
        f"[Audio] Output format: {options.audio_sample_rate} Hz, "
        f"{options.audio_channels} ch, AAC"
    )
    return lines


@dataclass
class AudioValidation:
    """Result of inspecting the final recording for a usable audio track."""

    has_audio: bool
    codec: Optional[str] = None
    duration: Optional[float] = None
    bit_rate: Optional[int] = None
    channels: Optional[int] = None
    sample_rate: Optional[int] = None
    error: Optional[str] = None

    def summary(self) -> str:
        if self.error:
            return f"[Audio] Final mux validation failed: {self.error}"
        if not self.has_audio:
            return "[Audio] Final mux contains audio=false — NO audio track"
        return (
            "[Audio] Final mux contains audio=true "
            f"(codec={self.codec}, {self.sample_rate} Hz, {self.channels} ch, "
            f"{self.duration:.1f}s, {self.bit_rate or 0} bit/s)"
        )


def resolve_ffprobe(ffmpeg_path: Optional[str]) -> Optional[str]:
    """Locate ``ffprobe`` — next to *ffmpeg_path* first, then on PATH."""
    if ffmpeg_path:
        cand = Path(ffmpeg_path).with_name(
            "ffprobe.exe" if ffmpeg_path.lower().endswith(".exe") else "ffprobe"
        )
        if cand.exists():
            return str(cand)
    return shutil.which("ffprobe")


def parse_ffprobe_audio(stdout_text: str) -> AudioValidation:
    """Parse ``ffprobe -show_streams`` flat key=value output for the audio track.

    Looks for an ``audio`` stream and reports its codec, duration, bitrate,
    channels and sample-rate.  ``has_audio`` is ``True`` only when an audio
    stream exists *and* its duration is greater than zero.
    """
    streams: List[Dict[str, str]] = []
    current: Optional[Dict[str, str]] = None
    for raw in stdout_text.splitlines():
        line = raw.strip()
        if line == "[STREAM]":
            current = {}
            continue
        if line == "[/STREAM]":
            if current is not None:
                streams.append(current)
            current = None
            continue
        if current is not None and "=" in line:
            key, _, value = line.partition("=")
            current[key.strip()] = value.strip()

    def _to_float(value: Optional[str]) -> Optional[float]:
        try:
            return float(value) if value not in (None, "", "N/A") else None
        except ValueError:
            return None

    def _to_int(value: Optional[str]) -> Optional[int]:
        f = _to_float(value)
        return int(f) if f is not None else None

    for st in streams:
        if st.get("codec_type") != "audio":
            continue
        duration = _to_float(st.get("duration"))
        return AudioValidation(
            has_audio=duration is not None and duration > 0,
            codec=st.get("codec_name"),
            duration=duration,
            bit_rate=_to_int(st.get("bit_rate")),
            channels=_to_int(st.get("channels")),
            sample_rate=_to_int(st.get("sample_rate")),
        )
    return AudioValidation(has_audio=False)


def validate_recording_audio(
    ffprobe_path: Optional[str], mp4_path: Path
) -> AudioValidation:
    """Probe *mp4_path* and report whether it carries a usable audio track.

    Never raises — a missing ffprobe / probe failure is returned as an
    ``AudioValidation`` with ``error`` set so the caller can log it.
    """
    if not ffprobe_path:
        return AudioValidation(has_audio=False, error="ffprobe not available")
    if not Path(mp4_path).exists():
        return AudioValidation(has_audio=False, error=f"file missing: {mp4_path}")
    import subprocess  # local import: only needed at validation time

    try:
        proc = subprocess.run(
            [
                ffprobe_path,
                "-hide_banner",
                "-loglevel", "error",
                "-show_streams",
                "-show_entries",
                "stream=codec_type,codec_name,duration,bit_rate,channels,sample_rate",
                str(mp4_path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
    except Exception as exc:  # noqa: BLE001
        return AudioValidation(has_audio=False, error=str(exc))
    if proc.returncode != 0:
        return AudioValidation(
            has_audio=False,
            error=(proc.stderr or "ffprobe failed").strip()[:200],
        )
    return parse_ffprobe_audio(proc.stdout or "")


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
    mic_name: Optional[str] = None,
    system_audio_name: Optional[str] = None,
) -> CaptureDevices:
    """Probe ffmpeg for the screen + microphone (+ best-effort loopback).

    Runs ``ffmpeg -list_devices true`` and parses stderr.  Falls back to
    sensible defaults when probing yields nothing.  Never raises — capture
    can still be attempted with the defaults.

    ``mic_name`` / ``system_audio_name`` request a specific device by name (or
    name substring); when given and matched they override the auto-pick.  The
    returned :class:`CaptureDevices` carries device names + a ``system_audio_note``
    for diagnostics.
    """
    import subprocess  # local import: only needed at probe time

    def _list(args: List[str]) -> str:
        try:
            proc = subprocess.run(
                [ffmpeg_path, *args],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
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
        # Explicit override by name → its index; else the auto-pick.
        mic = _match_av_index(audio, mic_name) or pick_microphone(audio) or "0"
        mic_label = next((n for i, n in audio if i == mic), mic_name)
        sys_audio, sys_label, note = _resolve_system_audio_macos(
            audio, want_system_audio, system_audio_name
        )
        return CaptureDevices(
            screen=screen,
            microphone=mic,
            system_audio=sys_audio,
            microphone_name=mic_label,
            system_audio_name=sys_label,
            system_audio_note=note,
        )

    if is_windows(platform):
        text = _list(["-f", "dshow", "-list_devices", "true", "-i", "dummy"])
        names = parse_dshow_audio_devices(text)
        mic = _match_name(names, mic_name) or next(
            (n for n in names if not _is_loopback(n)),
            names[0] if names else None,
        )
        sys_audio: Optional[str] = None
        note: Optional[str] = None
        if want_system_audio:
            sys_audio = _match_name(names, system_audio_name) or pick_system_audio(
                names
            )
            if sys_audio is None:
                note = (
                    "no WASAPI loopback / 'Stereo Mix' / 'virtual-audio-capturer' "
                    "device found"
                )
        else:
            note = "disabled in settings"
        return CaptureDevices(
            screen="desktop",
            microphone=mic,
            system_audio=sys_audio,
            microphone_name=mic,
            system_audio_name=sys_audio,
            system_audio_note=note,
        )

    return CaptureDevices(screen="0")


def _match_name(names: List[str], wanted: Optional[str]) -> Optional[str]:
    """Return the device name matching *wanted* (exact, then substring)."""
    if not wanted:
        return None
    if wanted in names:
        return wanted
    low = wanted.lower()
    return next((n for n in names if low in n.lower()), None)


def _match_av_index(
    audio: List[tuple], wanted: Optional[str]
) -> Optional[str]:
    """Return the avfoundation index whose name matches *wanted*."""
    if not wanted:
        return None
    low = wanted.lower()
    for idx, name in audio:
        if name == wanted or low in name.lower():
            return idx
    return None


def _resolve_system_audio_macos(
    audio: List[tuple],
    want_system_audio: bool,
    system_audio_name: Optional[str],
) -> tuple:
    """Resolve (index, name, note) for the macOS system-audio loopback."""
    if not want_system_audio:
        return None, None, "disabled in settings"
    if system_audio_name:
        idx = _match_av_index(audio, system_audio_name)
        if idx is not None:
            name = next((n for i, n in audio if i == idx), system_audio_name)
            return idx, name, None
    loop_name = pick_system_audio([name for _, name in audio])
    if loop_name is not None:
        idx = next((i for i, n in audio if n == loop_name), None)
        return idx, loop_name, None
    return None, None, (
        "no loopback device (install BlackHole or Loopback and route output "
        "through it)"
    )


def probe_screen_indices(
    ffmpeg_path: str, platform: str = sys.platform
) -> List[str]:
    """Return the avfoundation "Capture screen N" device indices, in order.

    Empty on non-macOS or when probing fails.  Used to map monitor ordinals to
    real avfoundation video device indices (which sit after the cameras).
    """
    if not is_macos(platform):
        return []
    import subprocess  # local import: only needed at probe time

    try:
        proc = subprocess.run(
            [ffmpeg_path, "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True,
            text=True,
            timeout=10,
        )
        text = (proc.stderr or "") + (proc.stdout or "")
    except Exception as exc:  # noqa: BLE001
        log.warning("screen-index probe failed: %s", exc)
        return []
    return capture_screen_indices(parse_avfoundation_devices(text)["video"])


def _is_loopback(name: str) -> bool:
    low = name.lower()
    return any(hint in low for hint in _LOOPBACK_HINTS)


def list_audio_devices(
    ffmpeg_path: Optional[str], platform: str = sys.platform
) -> List[str]:
    """Return the audio capture device names for the settings UI.

    Best-effort: returns ``[]`` when ffmpeg is unavailable or probing fails so
    the caller can fall back to the "Automatic" option.
    """
    if not ffmpeg_path:
        return []
    import subprocess  # local import: only needed when populating settings

    def _list(args: List[str]) -> str:
        try:
            proc = subprocess.run(
                [ffmpeg_path, *args],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )
            return (proc.stderr or "") + (proc.stdout or "")
        except Exception as exc:  # noqa: BLE001
            log.warning("audio device list failed: %s", exc)
            return ""

    if is_macos(platform):
        text = _list(["-f", "avfoundation", "-list_devices", "true", "-i", ""])
        return [name for _, name in parse_avfoundation_devices(text)["audio"]]
    if is_windows(platform):
        text = _list(["-f", "dshow", "-list_devices", "true", "-i", "dummy"])
        return parse_dshow_audio_devices(text)
    return []


try:  # Qt is optional for importing the pure helpers (e.g. in unit tests).
    from PySide6.QtCore import QObject, QProcess, QTimer, Signal

    class ScreenRecorderService(QObject):
        """Lifecycle controller for a segmented ffmpeg screen recording."""

        state_changed = Signal(object)   # RecorderState
        elapsed_changed = Signal(int)    # whole seconds, excluding pauses
        error = Signal(str)
        audio_level = Signal(float)      # live mix peak level in dBFS
        audio_validated = Signal(object) # AudioValidation for the final mp4

        def __init__(self, ffmpeg_path: Optional[str], parent=None) -> None:
            super().__init__(parent)
            self._ffmpeg = ffmpeg_path
            self._ffprobe = resolve_ffprobe(ffmpeg_path)
            self._meter_buf = ""             # partial metering stdout line
            self._last_validation: Optional[AudioValidation] = None
            self._state = RecorderState.IDLE
            self._proc: Optional[QProcess] = None
            self._output_dir: Optional[Path] = None
            self._devices: Optional[CaptureDevices] = None
            self._options = RecordingOptions()
            self._region: Optional[CaptureRegion] = None
            self._concat_on_stop = True
            self._segment_index = 0
            self._elapsed_seconds = 0
            self._timer = QTimer(self)
            self._timer.setInterval(1000)
            self._timer.timeout.connect(self._on_tick)
            self._stopping = False
            # Tail of the current ffmpeg's stderr, kept so a failure can report
            # the real reason instead of a generic "exited unexpectedly".
            self._stderr_tail = ""

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

        @property
        def last_validation(self) -> Optional["AudioValidation"]:
            return self._last_validation

        def start(
            self,
            output_dir: Path,
            devices: CaptureDevices,
            options: RecordingOptions,
            concat_on_stop: bool = True,
            region: Optional[CaptureRegion] = None,
        ) -> None:
            if self._state in (RecorderState.RECORDING, RecorderState.PAUSED):
                return
            self._output_dir = Path(output_dir)
            self._output_dir.mkdir(parents=True, exist_ok=True)
            self._devices = devices
            self._options = options
            self._region = region
            self._concat_on_stop = concat_on_stop
            self._segment_index = 0
            self._elapsed_seconds = 0
            self._stopping = False
            self._meter_buf = ""
            self._last_validation = None
            # Surface exactly which audio sources were resolved (and why any are
            # missing) so a silent recording is never a mystery.
            for line in audio_diagnostics(devices, options, sys.platform):
                log.info(line)
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
            # Validate that the produced file actually carries audio; if not,
            # log at ERROR so a silent recording is loud in the logs/UI.
            self._validate_audio(final)
            self._set_state(RecorderState.IDLE)
            self._stopping = False
            return final

        def _validate_audio(self, final: Optional[Path]) -> None:
            """Probe the final mp4 (or first segment) for a usable audio track."""
            target = final
            if target is None:
                segs = self.segment_paths()
                target = segs[0] if segs else None
            if target is None:
                return
            # No audio sources were requested → nothing to validate.
            opts = self._options
            dev = self._devices
            wanted_audio = dev is not None and (
                (dev.microphone is not None and not opts.mute_microphone)
                or (dev.system_audio is not None and not opts.mute_system_audio)
            )
            result = validate_recording_audio(self._ffprobe, target)
            self._last_validation = result
            if result.error:
                log.warning("recorder: %s", result.summary())
            elif not result.has_audio and wanted_audio:
                log.error("recorder: %s", result.summary())
            else:
                log.info("recorder: %s", result.summary())
            self.audio_validated.emit(result)

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
                    self._region,
                )
            except ValueError as exc:
                self._fail(str(exc))
                return

            self._segment_index += 1
            self._stderr_tail = ""
            proc = QProcess(self)
            proc.setProgram(self._ffmpeg)
            proc.setArguments(args)
            proc.setProcessChannelMode(QProcess.SeparateChannels)
            proc.readyReadStandardError.connect(self._on_proc_stderr)
            if self._options.meter_audio:
                proc.readyReadStandardOutput.connect(self._on_proc_stdout)
            proc.errorOccurred.connect(self._on_proc_error)
            proc.finished.connect(self._on_proc_finished)
            proc.start()
            if not proc.waitForStarted(3000):
                self._fail("ffmpeg failed to start")
                return
            self._proc = proc
            self._set_state(RecorderState.RECORDING)
            log.info("recording started: %s %s", self._ffmpeg, " ".join(args))

        def _on_proc_stderr(self) -> None:
            proc = self.sender()
            if proc is None:
                return
            try:
                chunk = bytes(proc.readAllStandardError()).decode(
                    "utf-8", errors="replace"
                )
            except Exception:  # noqa: BLE001
                return
            if not chunk:
                return
            # Keep only the last ~4 KB so a long-running capture stays bounded.
            self._stderr_tail = (self._stderr_tail + chunk)[-4096:]
            log.debug("ffmpeg: %s", chunk.rstrip())

        def _on_proc_stdout(self) -> None:
            """Parse the metering tap on ffmpeg's stdout → ``audio_level``."""
            proc = self.sender()
            if proc is None:
                return
            try:
                chunk = bytes(proc.readAllStandardOutput()).decode(
                    "utf-8", errors="replace"
                )
            except Exception:  # noqa: BLE001
                return
            if not chunk:
                return
            # Buffer until we have whole lines (the peak prints ~10×/s).
            buf = self._meter_buf + chunk
            buf, _, tail = buf.rpartition("\n")
            self._meter_buf = tail[-512:]
            if not buf:
                return
            peak = parse_meter_peak_db(buf)
            if peak is not None:
                self.audio_level.emit(peak)

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
            self._fail(self._with_stderr("ffmpeg process error"))

        def _on_proc_finished(self, code, _status) -> None:
            # Unexpected exit while we believed we were recording.
            if (
                self._state is RecorderState.RECORDING
                and not self._stopping
                and self._proc is not None
            ):
                self._fail(
                    self._with_stderr(f"ffmpeg exited unexpectedly (code {code})")
                )

        def _with_stderr(self, message: str) -> str:
            """Append the captured ffmpeg stderr tail to *message* if any.

            Without this the UI only ever sees a generic message; the appended
            tail reveals the real cause (e.g. a Screen-Recording permission
            denial or an unavailable capture device).
            """
            tail = self._stderr_tail.strip()
            if not tail:
                return message
            return f"{message}\n\n{tail}"

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
