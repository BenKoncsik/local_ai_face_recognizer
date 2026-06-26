"""Core ML face embedder (Apple Neural Engine / GPU).

Runs a Core ML MobileFaceNet model through Apple's Core ML runtime with the
compute units set to ``ALL``, so the Neural Engine (ANE) is used for every op
it supports and the rest fall back to GPU/CPU automatically.

This backend is **macOS-only** and entirely optional: it is selected only when
all of the following hold, otherwise the project's CPU TFLite path is used,
unchanged:

* the platform is macOS,
* ``coremltools`` is importable,
* a converted model file (``.mlpackage`` / ``.mlmodel``) exists on disk.

So Windows and Linux never touch this module — :func:`try_build_coreml_embedder`
returns ``None`` immediately off macOS.

Producing the model
-------------------
The Core ML model is generated once from the MobileFaceNet source with
``scripts/convert_mobilefacenet_to_coreml.py`` and written to
``models/mobilefacenet.mlpackage``.  It must produce the *same* 192-dim vectors
as the TFLite model (same weights), so existing embeddings stay comparable and
no re-embed is required when switching between the CPU and Core ML backends.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from app.embeddings.base import FaceEmbedder

log = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = "models/mobilefacenet.mlpackage"


class CoreMLEmbedder(FaceEmbedder):
    """Face embedder backed by a Core ML model running on the ANE/GPU.

    Args:
        model_path:  Path to a ``.mlpackage`` / ``.mlmodel`` file.
        input_size:  (width, height) the model expects (default 112×112).
    """

    def __init__(
        self,
        model_path: str,
        input_size: tuple[int, int] = (112, 112),
    ) -> None:
        import coremltools as ct

        self._backend = "coreml"
        self._input_w, self._input_h = input_size

        # ComputeUnit.ALL lets Core ML place ops on the Neural Engine first,
        # then GPU, then CPU — there is no API to pin work to the ANE only.
        self._model = ct.models.MLModel(
            str(model_path), compute_units=ct.ComputeUnit.ALL
        )

        spec = self._model.get_spec()
        self._input_name = spec.description.input[0].name
        self._output_name = spec.description.output[0].name
        self._layout, self._embedding_dim = self._probe_io(spec)

        from app import accel

        accel.report("embedding", accel.ACC_COREML, "MobileFaceNet Core ML")
        log.info(
            "Core ML embedder loaded: %s (input=%s '%s', out='%s', dim=%d, layout=%s)",
            Path(model_path).name, input_size, self._input_name,
            self._output_name, self._embedding_dim, self._layout,
        )

    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        arr = self._preprocess(face_bgr)
        out = self._model.predict({self._input_name: arr})
        vec = np.asarray(out[self._output_name], dtype=np.float32).ravel()
        return self._l2_normalise(vec)

    def embed_batch(self, faces_bgr: List[np.ndarray]) -> List[np.ndarray]:
        """Core ML's Python API runs one sample per ``predict`` call.

        There is no real batch speed-up here, so fall back to the per-crop
        default; the caller's parallel crop loading still hides the disk I/O.
        """
        return super().embed_batch(faces_bgr)

    # ------------------------------------------------------------------

    @staticmethod
    def _probe_io(spec) -> tuple[str, int]:
        """Infer the input tensor layout (NCHW/NHWC) and the output dimension."""
        out = spec.description.output[0]
        dims = list(getattr(out.type.multiArrayType, "shape", []) or [])
        embedding_dim = int(dims[-1]) if dims else 192

        layout = "NCHW"
        try:
            in_dims = list(spec.description.input[0].type.multiArrayType.shape)
            # NHWC if the last dim is the channel count (3); else NCHW.
            if in_dims and int(in_dims[-1]) == 3:
                layout = "NHWC"
        except Exception:  # noqa: BLE001 — image-typed inputs have no shape here
            layout = "NCHW"
        return layout, embedding_dim

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        """Resize + normalise a crop to the model's expected float32 tensor."""
        from app.embeddings.sface_embedder import _enhance_grayscale, _is_grayscale

        if _is_grayscale(face_bgr):
            face_bgr = _enhance_grayscale(face_bgr)
        resized = cv2.resize(
            face_bgr, (self._input_w, self._input_h), interpolation=cv2.INTER_LINEAR
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalised = (rgb.astype(np.float32) - 127.5) / 128.0  # (H, W, 3)
        if self._layout == "NCHW":
            normalised = np.transpose(normalised, (2, 0, 1))    # (3, H, W)
        return normalised[np.newaxis, ...]                      # add batch dim

    @staticmethod
    def _l2_normalise(vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        return vec if norm < 1e-8 else vec / norm


def try_build_coreml_embedder(config) -> Optional[CoreMLEmbedder]:
    """Build a :class:`CoreMLEmbedder` when (and only when) it is usable.

    Returns ``None`` — never raises — on any non-macOS platform, when
    ``coremltools`` is unavailable, when the user disabled it
    (``prefer_coreml=False``), or when no converted model file exists.  Callers
    then fall back to the CPU TFLite embedder exactly as before.
    """
    from app import accel

    emb_cfg = config.embedding
    if not accel.is_macos() or not getattr(emb_cfg, "prefer_coreml", True):
        return None

    from app.paths import resource_path

    raw_path = getattr(emb_cfg, "coreml_model_path", None)
    model_path = Path(raw_path) if raw_path else resource_path(_DEFAULT_MODEL_PATH)
    if not Path(model_path).exists():
        return None

    try:
        import coremltools  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        log.info("Core ML embedder unavailable (coremltools not importable): %s", exc)
        return None

    try:
        return CoreMLEmbedder(
            str(model_path), input_size=tuple(emb_cfg.input_size)
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "Core ML embedder failed to load (%s) — falling back to TFLite.", exc
        )
        return None
