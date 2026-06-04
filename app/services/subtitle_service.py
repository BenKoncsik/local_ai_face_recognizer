"""Helpers for pairing video files with external SRT subtitle files."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def subtitle_path_for_video(video_path: str | Path) -> Path:
    """Return the expected same-basename ``.srt`` path for *video_path*."""
    return Path(video_path).with_suffix(".srt")


def find_matching_subtitle(video_path: str | Path) -> Optional[Path]:
    """Find an SRT file whose basename exactly matches *video_path*.

    Matching is case-insensitive for both the basename and the ``.srt`` suffix,
    but no extra language/code suffixes are accepted.
    """
    video = Path(video_path)
    wanted_stem = video.stem.casefold()
    try:
        candidates = sorted(
            video.parent.iterdir(),
            key=lambda p: (p.name.casefold(), p.name),
        )
    except OSError:
        return None

    for candidate in candidates:
        if not candidate.is_file():
            continue
        if candidate.suffix.casefold() != ".srt":
            continue
        if candidate.stem.casefold() == wanted_stem:
            return candidate
    return None
