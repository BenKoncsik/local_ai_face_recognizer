"""Tests for scanning a folder and re-matching missing image records."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.db.database import init_db, session_scope
from app.db.models import Image
from app.services.image_library_service import get_image_library
from app.services.image_path_matcher import ImagePathMatcher
from app.services.scan_service import hash_file
from app.utils.image_utils import save_image_bgr


def _write_image(path: Path, shade: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((24, 24, 3), shade, dtype=np.uint8)
    assert save_image_bgr(path, arr)


@pytest.fixture()
def library(tmp_path):
    """A library root whose photos moved into a differently named sub-folder."""
    init_db(tmp_path / "faces.db")
    root = tmp_path / "library"
    root.mkdir()
    get_image_library().set_library_root(root)

    # Where the files really are now.
    actual = root / "2001" / "tel" / "photo.jpg"
    _write_image(actual, 40)

    with session_scope() as session:
        session.add(
            Image(
                # Where the database thinks it is (folder was renamed).
                file_path=str(root / "2001" / "winter" / "photo.jpg"),
                relative_path="2001/winter/photo.jpg",
                file_hash=hash_file(actual),
                file_mtime=0.0,
            )
        )
    return {"root": root, "actual": actual, "tmp_path": tmp_path}


# ---------------------------------------------------------------------------
# find_missing
# ---------------------------------------------------------------------------

def test_find_missing_lists_only_records_without_a_file(library):
    present = library["root"] / "here.jpg"
    _write_image(present, 10)
    with session_scope() as session:
        session.add(
            Image(
                file_path=str(present),
                relative_path="here.jpg",
                file_hash="h",
                file_mtime=0.0,
            )
        )

    with session_scope() as session:
        missing = ImagePathMatcher.find_missing(session)

    assert [m.name for m in missing] == ["photo.jpg"]


# ---------------------------------------------------------------------------
# match
# ---------------------------------------------------------------------------

def test_hash_match_is_confident(library):
    matcher = ImagePathMatcher([library["root"]])
    with session_scope() as session:
        report = matcher.run(session)

    assert report.missing_total == 1
    proposal = report.proposals[0]
    assert proposal.best.path == str(library["actual"])
    assert proposal.best.is_proof
    assert proposal.is_confident
    assert report.confident_count == 1


def test_same_name_different_content_is_demoted_and_not_confident(library):
    # A decoy with the same file name but different pixels, in a folder that
    # structurally looks like a better match than the real file's.
    decoy = library["root"] / "2001" / "winter_backup" / "photo.jpg"
    _write_image(decoy, 200)

    matcher = ImagePathMatcher([library["root"]])
    with session_scope() as session:
        report = matcher.run(session)

    proposal = report.proposals[0]
    assert proposal.best.path == str(library["actual"])  # hash wins over depth
    assert proposal.is_confident
    demoted = [c for c in proposal.candidates if c.path == str(decoy)]
    assert demoted and demoted[0].hash_match is False
    assert demoted[0].score < proposal.best.score


def test_without_hash_verification_deeper_path_ranks_first(library, tmp_path):
    other = tmp_path / "elsewhere" / "photo.jpg"
    _write_image(other, 90)

    matcher = ImagePathMatcher(
        [library["root"], tmp_path / "elsewhere"], verify_hash=False
    )
    with session_scope() as session:
        report = matcher.run(session)

    proposal = report.proposals[0]
    # Both are name-only matches (the "winter" folder was renamed to "tel"), but
    # the one that still shares the "2001" component outranks the stray copy.
    assert proposal.best.path == str(library["actual"])
    assert proposal.best.score > proposal.candidates[1].score
    # No proof and more than one candidate — the user must decide.
    assert not proposal.is_confident


def test_no_candidate_when_the_name_is_nowhere(library, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    matcher = ImagePathMatcher([empty])
    with session_scope() as session:
        report = matcher.run(session)

    assert report.proposals[0].candidates == []
    assert report.unmatched_count == 1


def test_progress_and_checkpoint_are_called(library):
    seen: list[int] = []
    checkpoints: list[int] = []

    matcher = ImagePathMatcher([library["root"]])
    with session_scope() as session:
        matcher.run(
            session,
            progress_cb=lambda pct, _msg: seen.append(pct),
            checkpoint=lambda: checkpoints.append(1),
        )

    assert seen and max(seen) <= 100
    assert checkpoints


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

def test_apply_updates_absolute_and_relative_paths(library):
    matcher = ImagePathMatcher([library["root"]])
    with session_scope() as session:
        report = matcher.run(session)
        proposal = report.proposals[0]
        result = ImagePathMatcher.apply(
            session,
            {proposal.missing.image_id: proposal.best.path},
            library_root=library["root"],
        )
        assert result.updated == 1

    with session_scope() as session:
        image = session.query(Image).one()
        assert image.file_path == str(library["actual"])
        assert image.relative_path == "2001/tel/photo.jpg"


def test_apply_refuses_to_point_two_records_at_one_file(library):
    second = library["root"] / "2001" / "winter" / "other.jpg"
    with session_scope() as session:
        session.add(
            Image(
                file_path=str(second),
                relative_path="2001/winter/other.jpg",
                file_hash="h2",
                file_mtime=0.0,
            )
        )

    with session_scope() as session:
        ids = [i.id for i in session.query(Image).order_by(Image.id).all()]
        result = ImagePathMatcher.apply(
            session,
            {ids[0]: str(library["actual"]), ids[1]: str(library["actual"])},
            library_root=library["root"],
        )

    assert result.updated == 1
    assert result.skipped == 1
    assert result.errors
