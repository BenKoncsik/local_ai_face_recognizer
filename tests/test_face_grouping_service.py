"""Tests for app.services.face_grouping_service.group_faces."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.services.face_grouping_service import group_faces


def _face(fid, vec, quality=None, confidence=0.9):
    arr = None if vec is None else np.asarray(vec, dtype=np.float32)
    return SimpleNamespace(
        id=fid,
        quality_score=quality,
        confidence=confidence,
        get_embedding=lambda a=arr: a,
    )


def test_similar_faces_group_together():
    a = [1.0, 0.0, 0.0]
    a2 = [0.99, 0.05, 0.0]      # cos(a, a2) ≈ 0.998 → same group
    b = [0.0, 1.0, 0.0]         # orthogonal → its own group
    groups = group_faces([_face(1, a), _face(2, a2), _face(3, b)])
    sizes = sorted(g.count for g in groups)
    assert sizes == [1, 2]
    # The two similar ones land together.
    pair = next(g for g in groups if g.count == 2)
    assert set(pair.member_ids) == {1, 2}


def test_faces_without_embedding_are_singletons():
    groups = group_faces([_face(1, None), _face(2, None)])
    assert len(groups) == 2
    assert all(g.count == 1 for g in groups)


def test_group_order_is_stable_and_members_keep_order():
    a = [1.0, 0.0]
    groups = group_faces([_face(5, a), _face(3, a), _face(9, a)])
    assert len(groups) == 1
    assert groups[0].member_ids == [5, 3, 9]


def test_representative_prefers_higher_quality():
    a = [1.0, 0.0, 0.0]
    groups = group_faces([
        _face(1, a, quality=0.4),
        _face(2, a, quality=0.95),
        _face(3, a, quality=0.6),
    ])
    assert len(groups) == 1
    assert groups[0].representative.id == 2


def test_threshold_controls_granularity():
    a = [1.0, 0.0, 0.0]
    c = [0.8, 0.6, 0.0]         # cos(a, c) = 0.8
    # Loose threshold merges them; strict keeps them apart.
    assert len(group_faces([_face(1, a), _face(2, c)], threshold=0.7)) == 1
    assert len(group_faces([_face(1, a), _face(2, c)], threshold=0.92)) == 2


def _pair_face(fid, vec, path):
    from types import SimpleNamespace
    f = _face(fid, vec)
    f.image = SimpleNamespace(id=fid, file_path=path)
    return f


def test_deoldified_variants_stack_despite_embedding_drift():
    from app.services.face_grouping_service import deoldified_pair_key
    # B&W original + a colorized variant of the SAME photo, embeddings drifted
    # below the normal merge threshold (cos ≈ 0.8).
    bw = _pair_face(1, [1.0, 0.0, 0.0], "/p/album/IMG_2.jpg")
    color = _pair_face(2, [0.8, 0.6, 0.0], "/p/colorized/IMG_2-deoldified.jpg")

    def pk(face):
        return deoldified_pair_key(face.image.file_path)

    # Without pairing the drift keeps them apart...
    assert len(group_faces([bw, color], threshold=0.92)) == 2
    # ...with pairing they collapse into one variant stack.
    groups = group_faces([bw, color], threshold=0.92, pair_key_fn=pk)
    assert len(groups) == 1
    assert set(groups[0].member_ids) == {1, 2}


def test_same_photo_different_people_not_merged_by_pairing():
    from app.services.face_grouping_service import deoldified_pair_key
    # Two DIFFERENT instances in the same photo (orthogonal embeddings) share a
    # pair key but must not be force-merged — the relaxed bar is still a bar.
    p1 = _pair_face(1, [1.0, 0.0, 0.0], "/p/IMG_9.jpg")
    p2 = _pair_face(2, [0.0, 1.0, 0.0], "/p/IMG_9.jpg")

    def pk(face):
        return deoldified_pair_key(face.image.file_path)

    groups = group_faces([p1, p2], threshold=0.92, pair_key_fn=pk)
    assert len(groups) == 2


def test_pair_key_matches_browser_pairing():
    from app.services.face_grouping_service import deoldified_pair_key
    # A deoldified variant resolves to the original's basename, folder-independent.
    assert deoldified_pair_key("/a/b/IMG_5.jpg") == "img_5.jpg"
    assert (
        deoldified_pair_key("/x/IMG_5-deoldified.jpg")
        == deoldified_pair_key("/y/IMG_5.jpg")
    )
