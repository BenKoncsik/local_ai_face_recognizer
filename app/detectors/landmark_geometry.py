"""Landmark-geometry sanity check — a cheap, high-precision false-positive gate.

YuNet returns 5 facial landmarks per detection (right-eye, left-eye, nose,
right-mouth-corner, left-mouth-corner — the canonical ArcFace order), but
nothing in the pipeline ever validated that those 5 points form a plausible
face.  A real face has a rigid arrangement: the two eyes sit roughly side by
side, the nose is below the eye line and between the eyes, and the mouth is
below the nose.  Pareidolia sources — knots in tree bark, hands, feet, the
texture on a playing card — that happen to fall inside YuNet's size + aspect
window produce *scrambled* landmark constellations that violate this geometry.

:func:`validate_face_landmarks` rejects those constellations.  Everything is
measured **relative to the face's own axes** (the eye line and the
eye→mouth direction) and normalized by the inter-ocular distance, so the check
is invariant to scale, image resolution and in-plane head tilt; tolerances are
deliberately loose so small, turned or mild-profile *real* faces still pass.

Design contract — **fail open**: when the landmarks are missing, malformed,
non-finite or degenerate (eyes coincide), the function returns ``True`` so a
landmark-less detector (Coral / Caffe SSD / Haar) is never penalised and a
freak numeric input never silently deletes a real face.  The two heaviest gates
(the crop-level verifier, the confidence-exempt shortcut) still apply on top.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from app.detectors.base import Detection

# Canonical YuNet / ArcFace landmark indices.
_RIGHT_EYE = 0
_LEFT_EYE = 1
_NOSE = 2
_RIGHT_MOUTH = 3
_LEFT_MOUTH = 4


@dataclass(frozen=True)
class LandmarkGeometryConfig:
    """Tolerances for :func:`validate_face_landmarks`.

    Loose by design: real faces (including small, turned and mildly tilted
    ones) must pass; only constellations that are geometrically impossible for
    a face should fail.  Tuned for the canonical YuNet 5-point order.
    """

    # Eyes must be horizontally separated (right eye left of left eye in image
    # space).  Required minimum x-separation as a fraction of inter-ocular dist.
    eye_x_min_frac: float = 0.30
    # Max in-plane roll of the eye line, in degrees.
    eye_line_max_deg: float = 45.0
    # Inter-ocular distance as a fraction of the bounding-box width.
    iod_box_min: float = 0.15
    iod_box_max: float = 0.95
    # Nose must project below the eye line and not far past the mouth, along
    # the face's own vertical (eye-midpoint → mouth-midpoint) axis, in iod.
    nose_below_eyes_min_frac: float = 0.05
    nose_below_mouth_slack_frac: float = 0.35
    # Eye-midpoint → mouth-midpoint distance, as a fraction of iod.
    mouth_below_eyes_min_frac: float = 0.40
    mouth_below_eyes_max_frac: float = 2.30
    # Nose horizontal position along the eye axis (0 = right eye, 1 = left eye).
    nose_axis_min_frac: float = -0.25
    nose_axis_max_frac: float = 1.25
    # Mouth corner separation as a fraction of iod.
    mouth_width_min_frac: float = 0.15
    mouth_width_max_frac: float = 1.80
    # Every landmark must sit within the bbox padded by this fraction per side.
    box_pad_frac: float = 0.25


DEFAULT = LandmarkGeometryConfig()


def _finite_point(p: Sequence[float]) -> bool:
    return (
        len(p) >= 2
        and math.isfinite(float(p[0]))
        and math.isfinite(float(p[1]))
    )


def geometry_reject_reason(
    det: Detection, cfg: LandmarkGeometryConfig = DEFAULT
) -> Optional[str]:
    """Return the name of the first failing geometry check, or ``None``.

    ``None`` means *accept* — either the geometry is plausible or the input is
    degenerate/landmark-less and the gate fails open.
    """
    lms = det.landmarks
    # --- Fail-open guards -------------------------------------------------
    if lms is None or len(lms) != 5:
        return None
    if not all(_finite_point(p) for p in lms):
        return None
    if det.w <= 0 or det.h <= 0:
        return None

    rex, rey = float(lms[_RIGHT_EYE][0]), float(lms[_RIGHT_EYE][1])
    lex, ley = float(lms[_LEFT_EYE][0]), float(lms[_LEFT_EYE][1])
    nx, ny = float(lms[_NOSE][0]), float(lms[_NOSE][1])
    rmx, rmy = float(lms[_RIGHT_MOUTH][0]), float(lms[_RIGHT_MOUTH][1])
    lmx, lmy = float(lms[_LEFT_MOUTH][0]), float(lms[_LEFT_MOUTH][1])

    iod = math.hypot(lex - rex, ley - rey)
    if iod < 1.0:  # eyes coincide → degenerate, fail open
        return None

    # Eye axis (right → left eye) unit vector.
    eux, euy = (lex - rex) / iod, (ley - rey) / iod

    # --- Check 1: eyes ordered & roughly horizontal -----------------------
    if (lex - rex) < cfg.eye_x_min_frac * iod:
        return "eyes_not_separated"
    eye_angle = abs(math.degrees(math.atan2(ley - rey, lex - rex)))
    if eye_angle > cfg.eye_line_max_deg:
        return "eye_line_tilt"

    # --- Check 2: inter-ocular distance sane vs box width -----------------
    if not (cfg.iod_box_min * det.w <= iod <= cfg.iod_box_max * det.w):
        return "iod_box_ratio"

    # --- Face vertical axis (eye midpoint → mouth midpoint) ---------------
    eye_mx, eye_my = (rex + lex) / 2.0, (rey + ley) / 2.0
    mouth_mx, mouth_my = (rmx + lmx) / 2.0, (rmy + lmy) / 2.0
    dvx, dvy = mouth_mx - eye_mx, mouth_my - eye_my
    dlen = math.hypot(dvx, dvy)
    if dlen < 1e-3:
        return "mouth_on_eyes"
    dux, duy = dvx / dlen, dvy / dlen

    # --- Check 3: mouth below eyes by a sane amount -----------------------
    if not (
        cfg.mouth_below_eyes_min_frac * iod
        <= dlen
        <= cfg.mouth_below_eyes_max_frac * iod
    ):
        return "mouth_eye_distance"

    # --- Check 4: nose below the eye line, above (near) the mouth ----------
    nose_t = (nx - eye_mx) * dux + (ny - eye_my) * duy
    if nose_t < cfg.nose_below_eyes_min_frac * iod:
        return "nose_above_eyes"
    if nose_t > dlen + cfg.nose_below_mouth_slack_frac * iod:
        return "nose_below_mouth"

    # --- Check 5: nose horizontally between the eyes ----------------------
    nose_s = ((nx - rex) * eux + (ny - rey) * euy) / iod
    if not (cfg.nose_axis_min_frac <= nose_s <= cfg.nose_axis_max_frac):
        return "nose_off_center"

    # --- Check 6: mouth corners ordered & sanely wide ---------------------
    mouth_width = math.hypot(lmx - rmx, lmy - rmy)
    if not (
        cfg.mouth_width_min_frac * iod
        <= mouth_width
        <= cfg.mouth_width_max_frac * iod
    ):
        return "mouth_width"
    rm_s = (rmx - rex) * eux + (rmy - rey) * euy
    lm_s = (lmx - rex) * eux + (lmy - rey) * euy
    if rm_s >= lm_s:  # right mouth corner must precede left along the eye axis
        return "mouth_corners_swapped"

    # --- Check 7: all landmarks inside the padded bbox --------------------
    pad_x = cfg.box_pad_frac * det.w
    pad_y = cfg.box_pad_frac * det.h
    x_lo, x_hi = det.x - pad_x, det.x + det.w + pad_x
    y_lo, y_hi = det.y - pad_y, det.y + det.h + pad_y
    for px, py in ((rex, rey), (lex, ley), (nx, ny), (rmx, rmy), (lmx, lmy)):
        if not (x_lo <= px <= x_hi and y_lo <= py <= y_hi):
            return "landmark_outside_box"

    return None


def validate_face_landmarks(
    det: Detection, cfg: LandmarkGeometryConfig = DEFAULT
) -> bool:
    """Return ``True`` when *det*'s 5 landmarks form a plausible face geometry.

    Fails open (returns ``True``) for landmark-less or degenerate detections;
    see the module docstring for the contract.
    """
    return geometry_reject_reason(det, cfg) is None
