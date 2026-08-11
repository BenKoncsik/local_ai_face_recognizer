"""Find images the database points at but that are not on disk, and re-match
them by scanning a folder.

After a database-only import (:mod:`app.services.database_package_service`) the
paths are re-based onto the local library root.  That is enough when the folder
layout below the root is identical on both machines.  When it is not — files
were re-organised, a sub-folder renamed, a phone dump merged — the record still
resolves to a path with no file behind it.

This module handles exactly that case:

1. :meth:`ImagePathMatcher.find_missing` lists the image rows whose resolved
   path does not exist.
2. :meth:`ImagePathMatcher.build_index` walks the chosen search folders once and
   indexes every image file by (lower-cased) file name.
3. :meth:`ImagePathMatcher.match` proposes candidates per missing record and
   scores them:

   * the file name matches (entry condition),
   * how many *trailing path components* also match (``1998/nyar/img.jpg``
     beats a lone ``img.jpg`` somewhere else),
   * the SHA-256 content hash recorded at index time — an exact hash match is
     proof, not a guess, and marks the candidate as *confident*.

4. :meth:`ImagePathMatcher.apply` writes the accepted choices back, updating both
   ``file_path`` and the portable ``relative_path``.

Every long step takes ``progress_cb`` / ``checkpoint`` callbacks so it can run
as a pausable, cancellable background task.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

log = logging.getLogger(__name__)

#: Fallback extensions when the caller does not pass the configured set.
DEFAULT_EXTENSIONS: tuple[str, ...] = (
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic",
)

#: Structural score of a bare file-name match, before any bonus.
_NAME_SCORE = 0.40
#: Bonus per matching parent folder (the file name itself is not counted).
_DEPTH_BONUS = 0.15
#: Maximum number of parent folders that can contribute a bonus.
_MAX_DEPTH_BONUS = 3
#: Bonus weight for parent folders that appear *anywhere* in the candidate path.
#: Small on purpose — it only breaks ties between equally deep matches, which is
#: what happens when one folder along the way was renamed.
_OVERLAP_BONUS = 0.10
#: A content-hash match is proof — such a candidate always scores 1.0.
_HASH_MATCH_SCORE = 1.0
#: Same name, different content: keep it visible but clearly demote it.
_HASH_MISMATCH_FACTOR = 0.25

ProgressCb = Callable[[int, str], None]
Checkpoint = Callable[[], None]


# ---------------------------------------------------------------------------
# Data objects
# ---------------------------------------------------------------------------

@dataclass
class MissingImage:
    """An image row whose file is not where the database says it is."""

    image_id: int
    file_path: str
    relative_path: Optional[str]
    file_hash: Optional[str]
    resolved_path: str

    @property
    def name(self) -> str:
        return _basename_any(self.relative_path or self.file_path)


@dataclass
class Candidate:
    """One possible on-disk file for a :class:`MissingImage`."""

    path: str
    score: float
    #: Matching trailing path components, including the file name (≥ 1).
    depth: int
    #: ``True``/``False`` after a content-hash comparison, ``None`` if not checked.
    hash_match: Optional[bool] = None

    @property
    def is_proof(self) -> bool:
        return self.hash_match is True


@dataclass
class MatchProposal:
    """A missing image together with its ranked candidates."""

    missing: MissingImage
    candidates: List[Candidate] = field(default_factory=list)

    @property
    def best(self) -> Optional[Candidate]:
        return self.candidates[0] if self.candidates else None

    @property
    def is_confident(self) -> bool:
        """True when the top candidate can be accepted without human review.

        Either the content hash proves it, or — when no hash is on record — a
        single candidate whose whole parent folder chain matches as well.
        """
        best = self.best
        if best is None:
            return False
        if best.is_proof:
            return True
        if self.missing.file_hash:
            return False  # a hash was available and did not prove this file
        return len(self.candidates) == 1 and best.depth >= 2


@dataclass
class MatchReport:
    """Outcome of a full find → scan → match run."""

    proposals: List[MatchProposal] = field(default_factory=list)
    missing_total: int = 0
    scanned_files: int = 0
    search_roots: List[str] = field(default_factory=list)
    cancelled: bool = False

    @property
    def matched_count(self) -> int:
        return sum(1 for p in self.proposals if p.candidates)

    @property
    def confident_count(self) -> int:
        return sum(1 for p in self.proposals if p.is_confident)

    @property
    def unmatched_count(self) -> int:
        return sum(1 for p in self.proposals if not p.candidates)


@dataclass
class ApplyResult:
    """Outcome of writing accepted matches back to the database."""

    updated: int = 0
    skipped: int = 0
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class ImagePathMatcher:
    """Re-attach database records to image files found by scanning a folder.

    Args:
        search_roots:  Folders to scan recursively for candidate files.
        extensions:    Image extensions to index (defaults to
                       :data:`DEFAULT_EXTENSIONS`).
        verify_hash:   Compare the SHA-256 recorded at index time with the
                       candidate's content.  Slower, but turns a guess into
                       proof; strongly recommended.
        max_candidates: How many candidates to keep (and hash) per record.
    """

    def __init__(
        self,
        search_roots: Sequence[Path | str],
        *,
        extensions: Optional[Sequence[str]] = None,
        verify_hash: bool = True,
        max_candidates: int = 5,
    ) -> None:
        self._roots = [Path(r) for r in search_roots]
        exts = extensions if extensions is not None else DEFAULT_EXTENSIONS
        self._extensions = {e.lower() if e.startswith(".") else f".{e.lower()}"
                            for e in exts}
        self._verify_hash = verify_hash
        self._max_candidates = max(1, max_candidates)
        # lower-cased file name -> on-disk paths
        self._index: Dict[str, List[str]] = {}
        self._scanned_files = 0
        self._hash_cache: Dict[str, Optional[str]] = {}

    # ------------------------------------------------------------------
    # Step 1 — what is missing
    # ------------------------------------------------------------------

    @staticmethod
    def find_missing(
        session,
        *,
        progress_cb: Optional[ProgressCb] = None,
        checkpoint: Optional[Checkpoint] = None,
    ) -> List[MissingImage]:
        """Return every image row whose resolved file is absent on disk.

        Resolution goes through the image-library service, so a record with a
        portable ``relative_path`` is looked up under the *current* library
        root rather than at its original absolute path.
        """
        from app.db.models import Image
        from app.services.image_library_service import resolve_image_path

        rows = (
            session.query(
                Image.id, Image.file_path, Image.relative_path, Image.file_hash
            )
            .order_by(Image.id)
            .all()
        )
        total = len(rows)
        missing: List[MissingImage] = []
        for idx, (image_id, file_path, relative_path, file_hash) in enumerate(rows, 1):
            if checkpoint is not None and idx % 200 == 0:
                checkpoint()
            stub = _ImageStub(file_path, relative_path)
            resolved = resolve_image_path(stub)
            resolved_str = str(resolved) if resolved else (file_path or "")
            if resolved_str and _exists(resolved_str):
                continue
            missing.append(
                MissingImage(
                    image_id=image_id,
                    file_path=file_path or "",
                    relative_path=relative_path,
                    file_hash=file_hash,
                    resolved_path=resolved_str,
                )
            )
            if progress_cb is not None and idx % 200 == 0:
                progress_cb(
                    int(idx / total * 100) if total else 100,
                    f"{idx}/{total}",
                )
        log.info("Path matcher: %d of %d image(s) missing", len(missing), total)
        return missing

    # ------------------------------------------------------------------
    # Step 2 — index the search folders
    # ------------------------------------------------------------------

    def build_index(
        self,
        *,
        progress_cb: Optional[ProgressCb] = None,
        checkpoint: Optional[Checkpoint] = None,
    ) -> int:
        """Walk every search root once, indexing image files by name.

        Returns the number of indexed files.
        """
        self._index.clear()
        self._scanned_files = 0
        for root in self._roots:
            if not root.is_dir():
                log.warning("Path matcher: search folder missing: %s", root)
                continue
            for dirpath, _dirnames, filenames in os.walk(root):
                if checkpoint is not None:
                    checkpoint()
                for filename in filenames:
                    if Path(filename).suffix.lower() not in self._extensions:
                        continue
                    self._index.setdefault(filename.lower(), []).append(
                        os.path.join(dirpath, filename)
                    )
                    self._scanned_files += 1
                if progress_cb is not None:
                    progress_cb(-1, f"{self._scanned_files} — {dirpath}")
        log.info(
            "Path matcher: indexed %d file(s) in %d folder(s)",
            self._scanned_files, len(self._roots),
        )
        return self._scanned_files

    @property
    def scanned_files(self) -> int:
        return self._scanned_files

    # ------------------------------------------------------------------
    # Step 3 — propose matches
    # ------------------------------------------------------------------

    def match(
        self,
        missing: Sequence[MissingImage],
        *,
        progress_cb: Optional[ProgressCb] = None,
        checkpoint: Optional[Checkpoint] = None,
    ) -> List[MatchProposal]:
        """Rank the indexed files against each missing record."""
        proposals: List[MatchProposal] = []
        total = len(missing)
        for idx, item in enumerate(missing, 1):
            if checkpoint is not None:
                checkpoint()
            proposals.append(self._propose(item))
            if progress_cb is not None:
                progress_cb(
                    int(idx / total * 100) if total else 100,
                    f"{idx}/{total} — {item.name}",
                )
        return proposals

    def _propose(self, item: MissingImage) -> MatchProposal:
        paths = self._index.get(item.name.lower(), [])
        if not paths:
            return MatchProposal(missing=item)

        reference = item.relative_path or item.file_path
        # Rank structurally first so the (expensive) hashing only touches the
        # few candidates that stand a chance.
        ranked = sorted(
            paths,
            key=lambda p: -_structural_score(reference, p)[0],
        )[: self._max_candidates]

        candidates: List[Candidate] = []
        for path in ranked:
            score, depth = _structural_score(reference, path)
            hash_match: Optional[bool] = None
            if self._verify_hash and item.file_hash:
                digest = self._file_hash(path)
                if digest is not None:
                    hash_match = digest == item.file_hash
                    score = (
                        _HASH_MATCH_SCORE
                        if hash_match
                        else score * _HASH_MISMATCH_FACTOR
                    )
            candidates.append(
                Candidate(
                    path=path,
                    score=round(min(score, 1.0), 3),
                    depth=depth,
                    hash_match=hash_match,
                )
            )

        candidates.sort(key=lambda c: (-c.score, -c.depth, c.path))
        return MatchProposal(missing=item, candidates=candidates)

    def _file_hash(self, path: str) -> Optional[str]:
        """SHA-256 of *path*, cached; ``None`` when the file cannot be read."""
        if path in self._hash_cache:
            return self._hash_cache[path]
        try:
            from app.services.scan_service import hash_file

            digest: Optional[str] = hash_file(Path(path))
        except OSError as exc:
            log.warning("Path matcher: could not hash %s: %s", path, exc)
            digest = None
        self._hash_cache[path] = digest
        return digest

    # ------------------------------------------------------------------
    # Convenience — the whole pass in one call
    # ------------------------------------------------------------------

    def run(
        self,
        session,
        *,
        progress_cb: Optional[ProgressCb] = None,
        checkpoint: Optional[Checkpoint] = None,
    ) -> MatchReport:
        """Find missing records, scan the folders, and rank the candidates."""
        def phase(base: int, span: int) -> ProgressCb:
            def _cb(pct: int, msg: str) -> None:
                if progress_cb is None:
                    return
                # -1 means "indeterminate" — keep the bar where the phase began.
                value = base if pct < 0 else base + int(pct / 100 * span)
                progress_cb(min(value, 100), msg)
            return _cb

        missing = self.find_missing(
            session, progress_cb=phase(0, 20), checkpoint=checkpoint
        )
        report = MatchReport(
            missing_total=len(missing),
            search_roots=[str(r) for r in self._roots],
        )
        if not missing:
            if progress_cb is not None:
                progress_cb(100, "")
            return report

        self.build_index(progress_cb=phase(20, 40), checkpoint=checkpoint)
        report.scanned_files = self._scanned_files
        report.proposals = self.match(
            missing, progress_cb=phase(60, 40), checkpoint=checkpoint
        )
        return report

    # ------------------------------------------------------------------
    # Step 4 — write the accepted matches back
    # ------------------------------------------------------------------

    @staticmethod
    def apply(
        session,
        decisions: Dict[int, str],
        *,
        library_root: Optional[Path | str] = None,
        progress_cb: Optional[ProgressCb] = None,
        checkpoint: Optional[Checkpoint] = None,
    ) -> ApplyResult:
        """Point the given image rows at their accepted files.

        Both ``file_path`` (absolute, this machine) and ``relative_path``
        (portable) are updated, so the record keeps working after the library
        root moves again.  Place/alias thumbnails that copied the old path are
        rewritten too.

        Args:
            decisions:    ``{image_id: chosen absolute path}``.
            library_root: Root used to derive the portable relative path.
                          Falls back to the configured library root.

        Returns:
            An :class:`ApplyResult`.
        """
        from app.db.models import Image, Place, PlaceAlias
        from app.services.image_library_service import get_image_library_optional

        result = ApplyResult()
        if not decisions:
            return result

        if library_root is None:
            lib = get_image_library_optional()
            library_root = lib.library_root if lib else None
        root = Path(library_root) if library_root else None

        # Two records must never end up on the same file.
        seen: Dict[str, int] = {}
        mapping: Dict[str, str] = {}
        total = len(decisions)

        with session.no_autoflush:
            for idx, (image_id, new_path) in enumerate(decisions.items(), 1):
                if checkpoint is not None:
                    checkpoint()
                if progress_cb is not None:
                    progress_cb(int(idx / total * 100), _basename_any(new_path))

                normalized = str(Path(new_path))
                owner = seen.get(normalized.lower())
                if owner is not None:
                    result.skipped += 1
                    result.errors.append(
                        f"{_basename_any(normalized)}: already assigned to image {owner}"
                    )
                    continue

                image = session.get(Image, image_id)
                if image is None:
                    result.skipped += 1
                    continue

                clash = (
                    session.query(Image.id)
                    .filter(Image.file_path == normalized, Image.id != image_id)
                    .first()
                )
                if clash is not None:
                    result.skipped += 1
                    result.errors.append(
                        f"{_basename_any(normalized)}: already used by image {clash[0]}"
                    )
                    continue

                seen[normalized.lower()] = image_id
                old_path = image.file_path
                if old_path and old_path != normalized:
                    mapping[old_path] = normalized
                image.file_path = normalized
                rel = _relative_to(normalized, root)
                if rel:
                    image.relative_path = rel
                result.updated += 1

            if mapping:
                for place in (
                    session.query(Place)
                    .filter(Place.thumbnail_path.isnot(None))
                    .all()
                ):
                    new = mapping.get(place.thumbnail_path)
                    if new:
                        place.thumbnail_path = new
                for alias in (
                    session.query(PlaceAlias)
                    .filter(PlaceAlias.thumbnail_path.isnot(None))
                    .all()
                ):
                    new = mapping.get(alias.thumbnail_path)
                    if new:
                        alias.thumbnail_path = new

        session.flush()
        log.info(
            "Path matcher: re-attached %d image(s), skipped %d",
            result.updated, result.skipped,
        )
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _ImageStub:
    """Minimal duck-type for :func:`resolve_image_path` (avoids loading rows)."""

    __slots__ = ("file_path", "relative_path", "id")

    def __init__(self, file_path: Optional[str], relative_path: Optional[str]) -> None:
        self.file_path = file_path
        self.relative_path = relative_path
        self.id = None


def _exists(path: str) -> bool:
    """Existence check that never raises (long/foreign paths, dead mounts)."""
    try:
        return os.path.exists(path)
    except OSError:
        return False


def _basename_any(path_str: str) -> str:
    """File name of *path_str*, honouring both ``/`` and ``\\`` separators."""
    return path_str.replace("\\", "/").rsplit("/", 1)[-1].strip() or "file"


def _components(path_str: str) -> List[str]:
    """Split a path written by *any* OS into its components."""
    return [p for p in path_str.replace("\\", "/").split("/") if p and p != "."]


def _suffix_match_depth(reference: str, candidate: str) -> int:
    """Count matching trailing components of two paths (case-insensitive).

    ``1998/nyar/img.jpg`` vs ``D:/kepek/1998/nyar/img.jpg`` → 3.  The file name
    itself counts as 1, so a name-only match returns 1.
    """
    ref = [c.lower() for c in _components(reference)]
    cand = [c.lower() for c in _components(candidate)]
    depth = 0
    for a, b in zip(reversed(ref), reversed(cand)):
        if a != b:
            break
        depth += 1
    return depth


def _structural_score(reference: str, candidate: str) -> tuple[float, int]:
    """Score *candidate* against *reference* on path shape alone.

    Returns ``(score, depth)``.  The matching trailing components dominate; a
    small bonus counts parent folders that appear *anywhere* in the candidate
    path, which is what distinguishes a file whose folder was merely renamed
    (``2001/winter`` → ``2001/tel``) from an unrelated copy of the same name.
    """
    depth = _suffix_match_depth(reference, candidate)
    score = _NAME_SCORE + _DEPTH_BONUS * min(depth - 1, _MAX_DEPTH_BONUS)

    ref_parents = {c.lower() for c in _components(reference)[:-1]}
    if ref_parents:
        cand_parents = {c.lower() for c in _components(candidate)[:-1]}
        overlap = len(ref_parents & cand_parents) / len(ref_parents)
        score += _OVERLAP_BONUS * overlap
    return min(score, 1.0), depth


def _relative_to(path_str: str, root: Optional[Path]) -> Optional[str]:
    """POSIX-style path of *path_str* under *root*, or ``None`` if outside."""
    if root is None:
        return None
    try:
        rel = Path(path_str).resolve().relative_to(Path(root).resolve())
    except (ValueError, OSError):
        return None
    return rel.as_posix() or None
