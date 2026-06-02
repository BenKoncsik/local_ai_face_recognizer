"""Image-file timeline log for screen recordings — SRT subtitle format.

Not speech recognition, not a transcript.  Records, with recording-relative
timestamps, which image file was active at each moment and which person was
selected::

    1
    00:00:00,000 --> 00:00:12,500
    1984_PICT0346.JPG
    Kijelölt személy: Panni

    2
    00:00:12,500 --> 00:00:31,200
    1984_PICT0347.JPG

    3
    00:00:31,200 --> 00:01:08,000
    1984_PICT0348.JPG
    Kijelölt személy: Kati

The writer is pure Python (no Qt / ffmpeg dependency) so it can be unit
tested in isolation.  It is crash-safe: every time a new entry opens, the
``.partial`` file is updated and flushed to disk immediately, so an
interrupted recording still leaves a readable subtitle file.

``elapsed`` is the recording-relative time in seconds **excluding pauses**
(the same clock the recorder uses for the video), so the subtitle timecodes
stay in sync with the final video.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

DEFAULT_PERSON_PREFIX = "Kijelölt személy:"


def format_timestamp(seconds: float) -> str:
    """Format *seconds* as SRT ``HH:MM:SS,mmm`` (clamped at zero)."""
    total_ms = int(max(0.0, seconds) * 1000 + 0.5)  # round to nearest ms
    ms = total_ms % 1000
    total_s = total_ms // 1000
    h, rem = divmod(total_s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


@dataclass
class TimelineEntry:
    """A single span during which one image/person was active."""

    start: float
    end: Optional[float]
    image_filename: Optional[str]
    person_name: Optional[str]


class RecordingTimelineLog:
    """Accumulates timeline entries and writes an SRT subtitle file.

    Parameters
    ----------
    path:
        Destination of the finalized log (``timeline.srt``).  A sibling
        ``<path>.partial`` file receives crash-safe incremental writes.
    person_prefix:
        Label prefix used in the person line (e.g. ``"Kijelölt személy:"``).
    """

    def __init__(
        self,
        path: str | Path,
        person_prefix: str = DEFAULT_PERSON_PREFIX,
        # none_placeholder kept for API compatibility but is no longer used in
        # the SRT output (person line is simply omitted when name is absent).
        none_placeholder: str = "-",
    ) -> None:
        self._path = Path(path)
        self._partial_path = self._path.with_name(self._path.name + ".partial")
        self._person_prefix = person_prefix
        self._entries: List[TimelineEntry] = []

    # ------------------------------------------------------------------
    # Recording-time API
    # ------------------------------------------------------------------

    def note_active(
        self,
        image_filename: Optional[str],
        person_name: Optional[str],
        elapsed: float,
    ) -> None:
        """Record that *image_filename* / *person_name* is active at *elapsed*.

        Opens a new entry only when the (image, person) pair actually changed,
        closing the previous entry at *elapsed*.  Consecutive identical states
        are collapsed into one span.
        """
        if self._entries:
            last = self._entries[-1]
            if (
                last.image_filename == image_filename
                and last.person_name == person_name
            ):
                return  # no change — keep the open span
            last.end = max(last.start, elapsed)

        self._entries.append(
            TimelineEntry(
                start=max(0.0, elapsed),
                end=None,
                image_filename=image_filename,
                person_name=person_name,
            )
        )
        self._flush_partial()

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def finalize(self, total_elapsed: float) -> Path:
        """Close the last entry at *total_elapsed* and write the SRT file.

        Returns the path of the written file.  Removes the ``.partial`` file
        on success.
        """
        if self._entries and self._entries[-1].end is None:
            self._entries[-1].end = max(self._entries[-1].start, total_elapsed)

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(self.render(total_elapsed), encoding="utf-8")
        try:
            if self._partial_path.exists():
                self._partial_path.unlink()
        except OSError:  # noqa: BLE001
            log.debug("could not remove partial timeline %s", self._partial_path)
        return self._path

    def render(self, fallback_end: Optional[float] = None) -> str:
        """Render all entries as an SRT document."""
        blocks = [
            self._format_block(idx + 1, e, fallback_end)
            for idx, e in enumerate(self._entries)
        ]
        # SRT: blocks separated by a blank line, trailing newline after last.
        return "\n".join(blocks) + ("\n" if blocks else "")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _format_block(
        self,
        index: int,
        entry: TimelineEntry,
        fallback_end: Optional[float],
    ) -> str:
        end = entry.end
        if end is None:
            end = fallback_end if fallback_end is not None else entry.start
        timecode = (
            f"{format_timestamp(entry.start)} --> {format_timestamp(end)}"
        )
        lines = [str(index), timecode]
        lines.append(entry.image_filename or "")
        if entry.person_name:
            lines.append(f"{self._person_prefix} {entry.person_name}")
        # Blank line terminates the SRT block.
        lines.append("")
        return "\n".join(lines)

    def _flush_partial(self) -> None:
        """Write a crash-safe snapshot of all entries to the partial file."""
        try:
            self._partial_path.parent.mkdir(parents=True, exist_ok=True)
            self._partial_path.write_text(self.render(), encoding="utf-8")
        except OSError:  # noqa: BLE001
            log.warning("could not flush partial timeline %s", self._partial_path)
