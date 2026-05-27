"""Greedy collision-free label placement for face name overlays.

Inputs : image size, face bboxes with label dimensions and selection state.
Outputs: final (x, y) for each label, plus optional leader-line endpoints
         when the label had to move far from its anchor face.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

_GAP = 6  # px gap between bbox edge and label


@dataclass
class FaceLabel:
    face_id: int
    x: int
    y: int
    w: int
    h: int
    selected: bool
    label_w: int
    label_h: int


@dataclass
class LabelLayout:
    label_x: int
    label_y: int
    leader_start: Optional[Tuple[int, int]] = None  # point on face bbox
    leader_end: Optional[Tuple[int, int]] = None    # point on label rect


def _overlaps(ax: int, ay: int, aw: int, ah: int,
              bx: int, by: int, bw: int, bh: int) -> bool:
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def _overlap_area(ax: int, ay: int, aw: int, ah: int,
                  bx: int, by: int, bw: int, bh: int) -> int:
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    return ix * iy


def _candidates(fx: int, fy: int, fw: int, fh: int,
                lw: int, lh: int) -> List[Tuple[int, int, int]]:
    """Return (label_x, label_y, base_penalty) candidate positions."""
    fc_x = fx + fw // 2
    fc_y = fy + fh // 2
    return [
        (fc_x - lw // 2, fy - lh - _GAP,      0),   # above center   (preferred)
        (fx,              fy - lh - _GAP,      10),   # above left-aligned
        (fx + fw - lw,    fy - lh - _GAP,      10),   # above right-aligned
        (fc_x - lw // 2,  fy + fh + _GAP,     50),   # below center
        (fx,              fy + fh + _GAP,      60),   # below left
        (fx + fw - lw,    fy + fh + _GAP,      60),   # below right
        (fx - lw - _GAP,  fc_y - lh // 2,    100),   # left
        (fx + fw + _GAP,  fc_y - lh // 2,    100),   # right
    ]


def place_labels(
    image_w: int,
    image_h: int,
    labels: List[FaceLabel],
) -> List[LabelLayout]:
    """Assign non-overlapping positions for face name labels (greedy).

    Priority order: selected face first, then largest face, then top-most.
    """
    if not labels:
        return []

    order = sorted(
        range(len(labels)),
        key=lambda i: (
            0 if labels[i].selected else 1,
            -(labels[i].w * labels[i].h),
            labels[i].y,
        ),
    )

    placed: List[Tuple[int, int, int, int]] = []  # (x, y, w, h) of committed labels
    results: List[Optional[LabelLayout]] = [None] * len(labels)

    for idx in order:
        lb = labels[idx]
        lw, lh = lb.label_w, lb.label_h

        if lw <= 0 or lh <= 0:
            results[idx] = LabelLayout(label_x=lb.x, label_y=lb.y)
            continue

        best: Optional[Tuple[int, int]] = None
        best_score = float("inf")

        for cx, cy, base_penalty in _candidates(lb.x, lb.y, lb.w, lb.h, lw, lh):
            cx = max(0, min(cx, image_w - lw))
            cy = max(0, min(cy, image_h - lh))

            score = float(base_penalty)

            # Heavy penalty for overlapping already-placed labels
            for px, py, pw, ph in placed:
                if _overlaps(cx, cy, lw, lh, px, py, pw, ph):
                    score += 5000 + _overlap_area(cx, cy, lw, lh, px, py, pw, ph)

            # Lighter penalty for covering any face bbox
            for fl in labels:
                if _overlaps(cx, cy, lw, lh, fl.x, fl.y, fl.w, fl.h):
                    score += 200

            if score < best_score:
                best_score = score
                best = (cx, cy)

        if best is None:
            best = (
                max(0, min(lb.x + lb.w // 2 - lw // 2, image_w - lw)),
                max(0, lb.y - lh - _GAP),
            )

        lx, ly = best
        placed.append((lx, ly, lw, lh))

        # Draw a leader line only when the label had to move far from the face
        leader_start = leader_end = None
        adj = _GAP + 4
        is_adjacent = (
            abs(ly + lh - lb.y) <= adj
            or abs(ly - (lb.y + lb.h)) <= adj
            or abs(lx + lw - lb.x) <= adj
            or abs(lx - (lb.x + lb.w)) <= adj
        )
        if not is_adjacent:
            leader_start = (lb.x + lb.w // 2, lb.y + lb.h // 2)
            leader_end = (lx + lw // 2, ly + lh // 2)

        results[idx] = LabelLayout(
            label_x=lx,
            label_y=ly,
            leader_start=leader_start,
            leader_end=leader_end,
        )

    return results  # type: ignore[return-value]
