"""Crash-safe ``recording_metadata.json`` writer for screen recordings.

The metadata file describes one recording session: when it started/ended, how
long it ran, the platform, codecs, frame rate and resolution, which monitors
were captured and in what layout, and — importantly — whether the session was
**closed cleanly** or left behind by a crash.

Crash safety
------------
The file is written **twice**:

* at :meth:`begin`, with ``clean_exit=False`` / ``crashed=True`` — so a process
  that dies mid-recording leaves a pessimistic, still-valid record on disk;
* at :meth:`finalize`, with ``clean_exit=True`` / ``crashed=False`` (unless an
  error was recorded), overwriting the pessimistic version.

The writer is pure Python (no Qt / ffmpeg dependency) so it is unit-testable in
isolation.  Every write goes through a temp file + atomic ``replace`` so a crash
mid-write never corrupts the JSON.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

METADATA_FILENAME = "recording_metadata.json"


@dataclass
class RecordingMetadata:
    """Serializable description of a recording session."""

    started_at: str
    ended_at: Optional[str] = None
    duration_seconds: float = 0.0
    platform: str = ""
    video_codec: str = "h264"
    audio_codec: str = "aac"
    fps: int = 0
    width: Optional[int] = None
    height: Optional[int] = None
    display_mode: str = ""
    monitors: List[dict] = field(default_factory=list)
    layout: Optional[dict] = None
    captured_microphone: bool = False
    captured_system_audio: bool = False
    clean_exit: bool = False
    crashed: bool = True
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": round(self.duration_seconds, 3),
            "platform": self.platform,
            "video_codec": self.video_codec,
            "audio_codec": self.audio_codec,
            "fps": self.fps,
            "resolution": (
                {"width": self.width, "height": self.height}
                if self.width is not None and self.height is not None
                else None
            ),
            "display_mode": self.display_mode,
            "monitors": self.monitors,
            "layout": self.layout,
            "captured_microphone": self.captured_microphone,
            "captured_system_audio": self.captured_system_audio,
            "clean_exit": self.clean_exit,
            "crashed": self.crashed,
            "errors": self.errors,
        }


class RecordingMetadataWriter:
    """Owns a :class:`RecordingMetadata` and persists it crash-safely."""

    def __init__(self, path: str | Path, metadata: RecordingMetadata) -> None:
        self._path = Path(path)
        self._meta = metadata

    @property
    def path(self) -> Path:
        return self._path

    @property
    def metadata(self) -> RecordingMetadata:
        return self._meta

    def begin(self) -> Path:
        """Write the initial pessimistic record (assume crash until finalized)."""
        self._meta.clean_exit = False
        self._meta.crashed = True
        self._write()
        return self._path

    def add_error(self, message: str) -> None:
        """Append an error message (and keep the record on disk current)."""
        if message:
            self._meta.errors.append(message)
            self._write()

    def finalize(
        self,
        ended_at: str,
        duration_seconds: float,
        *,
        crashed: bool = False,
    ) -> Path:
        """Mark the session finished and rewrite the file.

        ``crashed`` stays ``False`` for a normal stop; pass ``True`` when an
        error ended the recording (the partial video/segments are still valid,
        but the session did not complete normally).
        """
        self._meta.ended_at = ended_at
        self._meta.duration_seconds = max(0.0, duration_seconds)
        self._meta.crashed = crashed
        self._meta.clean_exit = not crashed
        self._write()
        return self._path

    # ------------------------------------------------------------------

    def _write(self) -> None:
        """Atomically write the metadata JSON; never raise into the caller."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_name(self._path.name + ".tmp")
            tmp.write_text(
                json.dumps(self._meta.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp, self._path)
        except OSError:  # noqa: BLE001 — metadata must never break a recording
            log.warning("could not write recording metadata %s", self._path)
