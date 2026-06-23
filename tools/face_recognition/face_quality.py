"""Face quality filtering — blur, size, pose, and liveness heuristics.

Reject faces that are:
  - Too blurry  (Laplacian variance < BLUR_THRESHOLD)
  - Too small   (shorter side < MIN_FACE_SIZE_PX)
  - Bad pose    (|yaw| or |pitch| or |roll| > MAX_POSE_DEGREES)
  - Suspected spoof (low colour variance + no specular highlights + bad EAR)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from face_detector import DetectedFace

log = logging.getLogger(__name__)

# ── Tuneable thresholds ──────────────────────────────────────────────────────
BLUR_THRESHOLD: float = 80.0       # Laplacian variance — higher = sharper
MIN_FACE_SIZE_PX: int = 120        # shorter bbox side in pixels
MAX_POSE_DEGREES: float = 20.0     # yaw / pitch / roll each

# Liveness heuristics (used when InsightFace's is_real is unavailable)
EAR_THRESHOLD: float = 0.20        # eye aspect ratio — flat images have low EAR
COLOR_VAR_THRESHOLD: float = 200.0 # colour variance — printouts/screens are flat
SPECULAR_RATIO_THRESHOLD: float = 0.002  # fraction of very bright pixels


@dataclass
class QualityResult:
    """Outcome of quality evaluation for one face."""

    accepted: bool
    reasons: List[str]   # why the face was rejected (empty when accepted)
    blur_score: Optional[float] = None
    pose: Optional[Tuple[float, float, float]] = None  # (yaw, pitch, roll)


class FaceQualityFilter:
    """Evaluate and filter face detections for downstream embedding quality."""

    def check(
        self,
        face: DetectedFace,
        img_rgb: np.ndarray,
        run_liveness: bool = True,
    ) -> QualityResult:
        """Return a :class:`QualityResult` for *face* in *img_rgb*.

        Args:
            face:          Detection from :class:`FaceDetector`.
            img_rgb:       Full image (RGB uint8) the face was detected in.
            run_liveness:  Whether to run the liveness heuristic check.
        """
        reasons: List[str] = []

        # ── Size ────────────────────────────────────────────────────────────
        min_side = min(face.bbox[2], face.bbox[3])
        if min_side < MIN_FACE_SIZE_PX:
            reasons.append(f"too_small (min_side={min_side}px < {MIN_FACE_SIZE_PX}px)")

        # ── Crop and blur ────────────────────────────────────────────────────
        crop = self._crop(img_rgb, face)
        blur = None
        if crop is not None and crop.size > 0:
            blur = self._laplacian_variance(crop)
            if blur < BLUR_THRESHOLD:
                reasons.append(f"blurry (score={blur:.1f} < {BLUR_THRESHOLD})")
        else:
            reasons.append("invalid_crop")

        # ── Pose ─────────────────────────────────────────────────────────────
        pose = face.pose  # (yaw, pitch, roll) or None
        if pose is not None:
            yaw, pitch, roll = pose
            for name, val in [("yaw", yaw), ("pitch", pitch), ("roll", roll)]:
                if abs(val) > MAX_POSE_DEGREES:
                    reasons.append(f"bad_pose_{name} ({val:.1f}°)")

        # ── Liveness ─────────────────────────────────────────────────────────
        if run_liveness and crop is not None:
            liveness_reason = self._liveness_check(crop, face)
            if liveness_reason:
                reasons.append(liveness_reason)

        return QualityResult(
            accepted=len(reasons) == 0,
            reasons=reasons,
            blur_score=blur,
            pose=pose,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _crop(img_rgb: np.ndarray, face: DetectedFace) -> Optional[np.ndarray]:
        x, y, w, h = face.bbox
        ih, iw = img_rgb.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(iw, x + w)
        y2 = min(ih, y + h)
        if x2 <= x1 or y2 <= y1:
            return None
        return img_rgb[y1:y2, x1:x2]

    @staticmethod
    def _laplacian_variance(crop_rgb: np.ndarray) -> float:
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _liveness_check(self, crop_rgb: np.ndarray, face: DetectedFace) -> Optional[str]:
        """Heuristic liveness gate.

        A flat photo / screen replay typically has:
          - Very low colour variance (uniform lighting across the face)
          - No specular highlights (bright reflective spots on skin/glasses)
          - Low eye aspect ratio (eyes are open in a real face, often closed/flat in prints)

        This is NOT a robust anti-spoofing model — it is a lightweight sanity
        check that catches the most obvious paper/screen attacks.
        """
        # Colour variance
        color_var = float(np.var(crop_rgb.astype(np.float32)))
        if color_var < COLOR_VAR_THRESHOLD:
            return f"liveness_flat_color (var={color_var:.1f})"

        # Specular highlights (very bright pixels)
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        bright_px = float(np.sum(gray > 240))
        total_px = gray.size
        if total_px > 0 and (bright_px / total_px) < SPECULAR_RATIO_THRESHOLD:
            # Real skin under normal light almost always has some bright spots.
            # A completely flat image (printout, screen with diffuse light) has none.
            # Only trigger when colour is also suspiciously uniform.
            if color_var < COLOR_VAR_THRESHOLD * 2:
                return f"liveness_no_specular (bright_ratio={bright_px/total_px:.4f})"

        # Eye Aspect Ratio via landmark geometry (if 5-point landmarks available)
        if face.landmarks is not None and len(face.landmarks) >= 5:
            ear = self._eye_aspect_ratio(face.landmarks)
            if ear is not None and ear < EAR_THRESHOLD:
                return f"liveness_low_ear ({ear:.3f})"

        return None

    @staticmethod
    def _eye_aspect_ratio(landmarks: np.ndarray) -> Optional[float]:
        """Approximate EAR from 5-point InsightFace landmarks.

        InsightFace 5-point layout: [left-eye, right-eye, nose, left-mouth, right-mouth]
        We estimate EAR from the inter-eye distance vs face height.
        """
        try:
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            nose = landmarks[2]
            eye_dist = float(np.linalg.norm(right_eye - left_eye))
            mid_eye_y = (left_eye[1] + right_eye[1]) / 2.0
            face_height = abs(float(nose[1]) - float(mid_eye_y)) * 3.0  # rough estimate
            if face_height < 1.0:
                return None
            return eye_dist / face_height
        except Exception:  # noqa: BLE001
            return None
