"""Google Coral Edge TPU face detector.

Uses ai-edge-litert (Google's replacement for tflite-runtime) with the
libedgetpu delegate.  Falls back to pycoral if available for backward compat.

Model must be an EdgeTPU-compiled TFLite file (*_edgetpu.tflite).
"""

from __future__ import annotations

import logging
import platform
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.detectors.base import Detection, FaceDetector, greedy_nms

log = logging.getLogger(__name__)

def _find_edgetpu_lib() -> str:
    """Return the first existing libedgetpu path, or the default name."""
    import ctypes.util
    if platform.system() == "Darwin":
        candidates = [
            "/usr/local/lib/libedgetpu.1.dylib",
            "/opt/homebrew/lib/libedgetpu.1.dylib",
            "libedgetpu.1.dylib",
        ]
    else:
        candidates = [
            "/usr/lib/libedgetpu.so.1",
            "/usr/local/lib/libedgetpu.so.1",
            "libedgetpu.so.1",
        ]
    for path in candidates:
        if path.startswith("/"):
            import os
            if os.path.exists(path):
                return path
        else:
            return path  # relative — let dlopen handle it
    return candidates[-1]


_EDGETPU_LIB = _find_edgetpu_lib()


def _make_interpreter(model_path: str):
    """Return (interpreter, backend_name).

    Priority:
    1. ai-edge-litert + EdgeTPU delegate  (Python 3.11+)
    2. pycoral make_interpreter           (Python ≤3.9, legacy)
    """
    # ── ai-edge-litert ────────────────────────────────────────────────────
    try:
        from ai_edge_litert.interpreter import Interpreter, load_delegate  # type: ignore[import]

        delegate = load_delegate(_EDGETPU_LIB)
        interp = Interpreter(
            model_path=model_path,
            experimental_delegates=[delegate],
        )
        interp.allocate_tensors()
        log.info("CoralDetector: using ai_edge_litert + EdgeTPU delegate")
        return interp, "ai_edge_litert"
    except ImportError:
        pass
    except Exception as exc:
        raise ImportError(
            f"ai-edge-litert found but EdgeTPU delegate failed to load: {exc}\n"
            f"Make sure libedgetpu is installed and the device is connected."
        ) from exc

    # ── pycoral (legacy) ──────────────────────────────────────────────────
    try:
        from pycoral.utils.edgetpu import make_interpreter  # type: ignore[import]

        interp = make_interpreter(model_path)
        interp.allocate_tensors()
        log.info("CoralDetector: using pycoral make_interpreter")
        return interp, "pycoral"
    except ImportError:
        pass

    raise ImportError(
        "No EdgeTPU backend available.\n"
        "Install ai-edge-litert:  pip install ai-edge-litert\n"
        f"Install libedgetpu:       see https://coral.ai/software/"
    )


def _set_input(interpreter, rgb_hwc: np.ndarray) -> None:
    inp = interpreter.get_input_details()[0]
    interpreter.set_tensor(inp["index"], np.expand_dims(rgb_hwc, axis=0))


def _get_detections(
    interpreter, img_w: int, img_h: int, score_threshold: float
) -> List[Tuple[int, int, int, int, float]]:
    """Parse SSD output tensors → list of (x, y, w, h, score)."""
    out = interpreter.get_output_details()
    # SSD postprocess output order: boxes, classes, scores, count
    boxes  = interpreter.get_tensor(out[0]["index"])[0]   # [N, 4] ymin xmin ymax xmax [0,1]
    scores = interpreter.get_tensor(out[2]["index"])[0]   # [N]
    count  = int(interpreter.get_tensor(out[3]["index"])[0])

    results = []
    for i in range(min(count, len(scores))):
        if scores[i] < score_threshold:
            continue
        ymin, xmin, ymax, xmax = boxes[i]
        x = int(xmin * img_w)
        y = int(ymin * img_h)
        w = int((xmax - xmin) * img_w)
        h = int((ymax - ymin) * img_h)
        results.append((x, y, w, h, float(scores[i])))
    return results


class CoralDetector(FaceDetector):
    """Face detector backed by a Google Coral Edge TPU."""

    def __init__(self, model_path: str) -> None:
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"Coral model file not found: {model_file}\n"
                "Download a suitable *_edgetpu.tflite model and set "
                "detection.coral_model_path in config.yaml."
            )

        self._interpreter, self._backend = _make_interpreter(str(model_file))
        inp = self._interpreter.get_input_details()[0]
        _, h, w, _ = inp["shape"]
        self._input_h = int(h)
        self._input_w = int(w)
        log.info("Coral detector ready — input %dx%d via %s", w, h, self._backend)

    @property
    def backend_name(self) -> str:
        return f"coral_{self._backend}"

    def detect(
        self,
        image_bgr: np.ndarray,
        confidence_threshold: float = 0.65,
        min_face_size: int = 50,
    ) -> List[Detection]:
        img_h, img_w = image_bgr.shape[:2]

        resized = cv2.resize(
            image_bgr, (self._input_w, self._input_h), interpolation=cv2.INTER_LINEAR
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        _set_input(self._interpreter, rgb)
        try:
            self._interpreter.invoke()
        except RuntimeError as exc:
            msg = str(exc)
            if "EdgeTpuDelegateForCustomOp" in msg or "unresolved custom op" in msg.lower():
                raise RuntimeError(
                    "EdgeTPU inference failed — device may be disconnected or not recognized. "
                    "Falling back to CPU detector."
                ) from exc
            raise

        raw = _get_detections(self._interpreter, img_w, img_h, confidence_threshold)

        results: List[Detection] = []
        for x, y, w, h, conf in raw:
            det = Detection(x=x, y=y, w=w, h=h, confidence=conf).clamp(img_w, img_h)
            if det.w < min_face_size or det.h < min_face_size:
                continue
            # The quantized SSD model regularly emits implausibly elongated
            # boxes over background structures — same plausibility window as
            # the CPU detectors.
            aspect = det.w / det.h if det.h > 0 else 0.0
            if not 0.4 <= aspect <= 2.5:
                continue
            results.append(det)

        # The SSD postprocess op leaves overlapping duplicates in; collapse
        # them so the same physical face never yields nested boxes.
        results = greedy_nms(results, iou_threshold=0.4)

        log.debug("Coral detected %d face(s)", len(results))
        return results
