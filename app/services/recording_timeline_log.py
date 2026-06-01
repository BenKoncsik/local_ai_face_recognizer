"""Image-file timeline ("subtitle") log for screen recordings.

This is **not** speech recognition and not a real transcript.  It records,
with recording-relative timestamps, which image file was active at each
moment of the recording and which person was selected, e.g.::

    00:00:00 - 00:00:12 | 1984_PICT0346.JPG | kijelölt személy: Panni
    00:00:12 - 00:00:31 | 1984_PICT0347.JPG | kijelölt személy: -
    00:00:31 - 00:01:08 | 1984_PICT0348.JPG | kijelölt személy: Kati

The writer is pure Python (no Qt / ffmpeg dependency) so it can be unit
tested in isolation.  It is crash-safe: every time a new entry opens, an
``.partial`` line is appended and flushed to disk immediately, so an
interrupted recording still leaves a readable log.

``elapsed`` is the recording-relative time in seconds **excluding pauses**
(the same clock the recorder uses for the video), so the timeline lines stay
in sync with the final video.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

# Prefix and placeholder are configurable so the UI can localize them while
# the formatting logic stays testable with fixed strings.
DEFAULT_PERSON_PREFIX = "kijelölt személy:"
DEFAULT_NONE_PLACEHOLDER = "-"


def format_timestamp(seconds: float) -> str:
    """Format *seconds* as ``HH:MM:SS`` (clamped at zero)."""
    total = int(max(0.0, seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


@dataclass
class TimelineEntry:
    """A single span during which one image/person was active."""

    start: float
    end: Optional[float]
    image_filename: Optional[str]
    person_name: Optional[str]


class RecordingTimelineLog:
    """Accumulates timeline entries and writes the ``timeline.txt`` log.

    Parameters
    ----------
    path:
        Destination of the finalized log (``timeline.txt``).  A sibling
        ``<path>.partial`` file receives crash-safe incremental writes.
    person_prefix / none_placeholder:
        Localizable strings used when formatting each line.
    """

    def __init__(
        self,
        path: str | Path,
        person_prefix: str = DEFAULT_PERSON_PREFIX,
        none_placeholder: str = DEFAULT_NONE_PLACEHOLDER,
    ) -> None:
        self._path = Path(path)
        self._partial_path = self._path.with_name(self._path.name + ".partial")
        self._person_prefix = person_prefix
        self._none_placeholder = none_placeholder
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
        """Close the last entry at *total_elapsed* and write ``timeline.txt``.

        Returns the path of the written log.  Removes the ``.partial`` file on
        success.
        """
        if self._entries and self._entries[-1].end is None:
            self._entries[-1].end = max(self._entries[-1].start, total_elapsed)

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(self.render(total_elapsed), encoding="utf-8")
        try:
            if self._partial_path.exists():
                self._partial_path.unlink()
        except OSError:  # noqa: BLE001 — partial cleanup is best-effort
            log.debug("could not remove partial timeline %s", self._partial_path)
        return self._path

    def render(self, fallback_end: Optional[float] = None) -> str:
        """Render all entries as the final multi-line log text."""
        lines = [self._format_entry(e, fallback_end) for e in self._entries]
        return "\n".join(lines) + ("\n" if lines else "")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _format_entry(
        self, entry: TimelineEntry, fallback_end: Optional[float]
    ) -> str:
        end = entry.end
        if end is None:
            end = fallback_end if fallback_end is not None else entry.start
        image = entry.image_filename or self._none_placeholder
        person = entry.person_name or self._none_placeholder
        return (
            f"{format_timestamp(entry.start)} - {format_timestamp(end)} "
            f"| {image} | {self._person_prefix} {person}"
        )

    def _flush_partial(self) -> None:
        """Write a crash-safe snapshot of all entries to the partial file."""
        try:
            self._partial_path.parent.mkdir(parents=True, exist_ok=True)
            self._partial_path.write_text(self.render(), encoding="utf-8")
        except OSError:  # noqa: BLE001 — never let logging break a recording
            log.warning("could not flush partial timeline %s", self._partial_path)
