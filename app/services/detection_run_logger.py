"""Per-run detection debug logger.

Creates one timestamped plain-text log file in the detection_logs directory
for each scan run when the detection debug log setting is enabled.  Each file
captures raw detector output, verifier decisions, and any errors so that
cross-platform false-positive differences can be diagnosed offline.
"""

from __future__ import annotations

import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)


class DetectionRunLogger:
    """Writes a timestamped log file for one detection run.

    Typical usage inside DetectionService::

        logger = DetectionRunLogger(detection_logs_dir())
        logger.start(total_images=len(ids), mode="fast", backend="yunet")
        # per image:
        logger.begin_image(path, w, h, mode_label, threshold)
        logger.log_verification(raw_dets, kept_dets)  # may be called >1x (adaptive)
        logger.end_image(final_face_count)
        # on error:
        logger.log_error(path, exc)
        # at the end:
        logger.finish(total_faces, total_images)
    """

    def __init__(self, logs_dir: Path) -> None:
        logs_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._path = logs_dir / f"detection_{ts}.log"
        self._fh = None
        try:
            self._fh = self._path.open("w", encoding="utf-8")
        except OSError as exc:
            log.warning("DetectionRunLogger: cannot open %s: %s", self._path, exc)
        # Accumulator for the current image being processed.
        self._cur_image: Optional[str] = None
        self._cur_lines: List[str] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write(self, text: str) -> None:
        if self._fh is None:
            return
        try:
            self._fh.write(text)
            self._fh.flush()
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Public API — called by DetectionService
    # ------------------------------------------------------------------

    def start(self, total_images: int, mode: str, backend: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write(
            f"=== Detection Run Started: {ts} ===\n"
            f"  mode={mode}  backend={backend}  images={total_images}\n\n"
        )

    def begin_image(
        self, path: str, width: int, height: int, mode: str, threshold: float
    ) -> None:
        self._cur_image = path
        self._cur_lines = [
            f"--- IMAGE: {path}\n",
            f"    size={width}x{height}  mode={mode}  threshold={threshold:.3f}\n",
        ]

    def log_verification(
        self,
        raw_dets: list,
        kept_dets: list,
        label: str = "main",
        face_details: Optional[list] = None,
    ) -> None:
        """Record the result of one verification pass (main or one adaptive rung).

        When *face_details* is provided it is a list of ``(det, detail_dict)``
        pairs collected by the caller during the per-face verify loop.  Each
        *detail_dict* is the payload delivered by the verifier's ``detail_cb``
        callback (see ``FaceVerifier`` and ``MultiStageFaceValidator``).
        When absent, only ``[KEPT]``/``[DROP]`` labels are written.
        """
        dropped_count = len(raw_dets) - len(kept_dets)
        self._cur_lines.append(
            f"    [{label}] raw={len(raw_dets)}  kept={len(kept_dets)}"
            f"  dropped={dropped_count}\n"
        )
        if face_details is not None:
            for det, detail in face_details:
                self._cur_lines.extend(self._format_face_detail(det, detail))
        else:
            kept_set = {(d.x, d.y, d.w, d.h) for d in kept_dets}
            for det in raw_dets:
                key = (det.x, det.y, det.w, det.h)
                status = "KEPT" if key in kept_set else "DROP"
                self._cur_lines.append(
                    f"      [{status}] bbox=({det.x},{det.y},{det.w},{det.h})"
                    f" conf={det.confidence:.3f}\n"
                )

    @staticmethod
    def _format_face_detail(det, detail: dict) -> list:
        """Return a list of log lines for one face's verification detail."""
        result = detail.get("result", "?")
        lines = [
            f"      bbox=({det.x},{det.y},{det.w},{det.h})"
            f" conf={det.confidence:.3f} → {result}\n"
        ]
        mode = detail.get("mode", detail.get("verifier", ""))

        if mode == "exempt":
            lines.append(f"        (confidence exempt: {detail.get('reason', '')})\n")

        elif mode == "yunet":
            reason = detail.get("reason", "")
            crop_conf = detail.get("crop_conf")
            line = f"        verifier=yunet  reason={reason}"
            if crop_conf is not None:
                line += f"  crop_conf={crop_conf:.3f}"
            lines.append(line + "\n")

        elif mode == "yunet_only":
            fallback = detail.get("fallback_reason", "")
            yunet_vote = detail.get("yunet", "?")
            lines.append(
                f"        verifier=yunet_only (fallback: {fallback})"
                f"  vote={yunet_vote}\n"
            )

        elif mode == "multistage":
            votes: dict = detail.get("votes", {})
            errors: dict = detail.get("errors", {})
            weight = detail.get("confirm_weight", 0)
            min_conf = detail.get("min_confirmations", "?")
            for tech, vote in votes.items():
                err = f"  ERROR: {errors[tech]}" if tech in errors else ""
                lines.append(f"        {tech:<14} {vote}{err}\n")
            lines.append(f"        confirm_weight={weight}/{min_conf}\n")

        return lines

    def log_adaptive_rung(
        self, conf: float, min_size: int, count_before: int, count_after: int
    ) -> None:
        self._cur_lines.append(
            f"    [adaptive] conf={conf:.3f}  min_size={min_size}"
            f"  raw={count_before}  after_verify={count_after}\n"
        )

    def log_fallback(self, from_backend: str, to_backend: str, reason: str) -> None:
        """Log a detector backend fallback (e.g. Coral TPU → CPU)."""
        self._cur_lines.append(
            f"    [FALLBACK] {from_backend} -> {to_backend}  reason={reason}\n"
        )
        self._write("".join(self._cur_lines))
        self._cur_lines = []

    def end_image(self, final_face_count: int) -> None:
        self._cur_lines.append(f"    => final faces: {final_face_count}\n\n")
        self._write("".join(self._cur_lines))
        self._cur_image = None
        self._cur_lines = []

    def log_error(self, path: str, exc: Exception) -> None:
        tb = traceback.format_exc()
        self._write(f"ERROR: {path}\n  {exc}\n{tb}\n\n")

    def finish(self, total_faces: int, total_images: int) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write(
            f"=== Run Complete: {ts} ===\n"
            f"  total_images={total_images}  total_faces={total_faces}\n"
        )
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None

    @property
    def log_path(self) -> Path:
        return self._path
