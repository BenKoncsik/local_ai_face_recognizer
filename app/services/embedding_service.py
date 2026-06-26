"""Face embedding service.

Generates embedding vectors for all faces that do not yet have one stored.
Resumable: only processes faces with ``_embedding IS NULL``.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.config import AppConfig
from app.db.models import Face
from app.embeddings.base import FaceEmbedder
from app.utils.image_utils import load_image_bgr

log = logging.getLogger(__name__)


class EmbeddingService:
    """Generates and persists face embeddings.

    Args:
        session:     SQLAlchemy session.
        embedder:    A :class:`~app.embeddings.base.FaceEmbedder` instance.
        config:      Full application configuration.
        progress_cb: Optional ``(current, total, face_id)`` callback.
    """

    def __init__(
        self,
        session: Session,
        embedder: FaceEmbedder,
        config: AppConfig,
        progress_cb: Optional[Callable[[int, Optional[int], int], None]] = None,
    ) -> None:
        self._session = session
        self._embedder = embedder
        self._config = config
        self._progress_cb = progress_cb or (lambda *_: None)

    def process_pending(
        self,
        exclude_low_quality: bool = False,
        expected_dim: Optional[int] = None,
    ) -> int:
        """Generate embeddings for all faces that don't have one yet.

        If *expected_dim* is given, faces whose stored embedding has a
        different length (e.g. after switching the embedding backend) are
        treated as missing and are re-embedded.  This prevents the
        dimension-mismatch crash that occurs when ``train_matrix`` (built from
        old-dim embeddings) is multiplied against a new-dim query vector.

        Args:
            exclude_low_quality: When True, faces flagged as low quality are
                skipped.  NULL quality (not yet evaluated) is treated as
                usable for backward compatibility.
            expected_dim: Expected embedding vector length.  When set, any
                face with a stored embedding of a different length has its
                embedding cleared before processing so it will be re-embedded.

        Returns:
            Number of faces embedded.
        """
        if expected_dim is not None:
            self._clear_wrong_dim_embeddings(expected_dim)

        query = (
            self._session.query(Face)
            .filter(~Face.embedding_exists())
            .filter(Face.is_excluded == False)  # noqa: E712
        )
        if exclude_low_quality:
            query = query.filter(
                or_(Face.is_low_quality.is_(None), Face.is_low_quality == False)  # noqa: E712
            )
        pending: List[Face] = query.all()

        total = len(pending)
        log.info("Embedding %d face(s) without vectors", total)
        if total == 0:
            return 0

        batch_size = max(1, int(getattr(self._config.embedding, "batch_size", 32)))
        workers = max(
            1,
            min(
                int(getattr(self._config.embedding, "loader_workers", 8)),
                os.cpu_count() or 1,
            ),
        )

        embedded = 0
        processed = 0
        # Crop loading is I/O- and decode-bound (the GIL is released during
        # cv2 decode), so it overlaps the previous batch's model inference on a
        # small thread pool.  All ORM access stays on this thread — only plain
        # crop-path strings are handed to the workers.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, total, batch_size):
                chunk = pending[start : start + batch_size]
                crop_paths = [face.crop_path for face in chunk]
                crops = list(pool.map(self._load_crop_from_path, crop_paths))

                loaded = [
                    (face, img)
                    for face, img in zip(chunk, crops)
                    if img is not None
                ]
                if loaded:
                    vectors = self._embed_loaded(loaded)
                    for (face, _img), vec in zip(loaded, vectors):
                        if vec is None:
                            continue
                        face.set_embedding(vec)
                        embedded += 1

                for face in chunk:
                    processed += 1
                    self._progress_cb(processed, total, face.id)

                self._session.commit()

        log.info("Embedding complete: %d / %d face(s) embedded", embedded, total)
        return embedded

    def _embed_loaded(self, loaded) -> List[Optional[np.ndarray]]:
        """Embed a batch of already-loaded crops, degrading gracefully.

        Tries the embedder's batched path first; if it fails (e.g. one bad
        crop poisons the whole batch) it falls back to per-crop embedding so a
        single failure never loses the rest of the batch.  Returns one vector
        per input (``None`` for a crop that could not be embedded).
        """
        imgs = [img for _face, img in loaded]
        try:
            return list(self._embedder.embed_batch(imgs))
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Batched embedding failed (%s) — falling back to per-crop", exc
            )

        vectors: List[Optional[np.ndarray]] = []
        for (face, img) in loaded:
            try:
                vectors.append(self._embedder.embed(img))
            except Exception as exc:  # noqa: BLE001
                log.warning("Embedding failed for face id=%d: %s", face.id, exc)
                vectors.append(None)
        return vectors

    @staticmethod
    def _load_crop_from_path(crop_path: Optional[str]) -> "Optional[np.ndarray]":
        """Read + decode a face crop from disk; thread-safe (no ORM access).

        Returns the BGR image, or None when the path is missing/unreadable.
        """
        if not crop_path:
            return None
        crop_file = Path(crop_path)
        if not crop_file.exists():
            return None
        return load_image_bgr(str(crop_file))

    def _clear_wrong_dim_embeddings(self, expected_dim: int) -> None:
        """Null-out stored embeddings whose byte length doesn't match *expected_dim*.

        Called before the pending-face query so that dimension-mismatched faces
        are re-embedded in the same pass rather than silently kept.
        """
        from app.db.models import FaceBlob

        expected_bytes = expected_dim * 4  # float32 = 4 bytes per element
        # SQLite / SQLAlchemy: length() on a BLOB column returns byte count.
        from sqlalchemy import func

        mismatched: List = (
            self._session.query(FaceBlob)
            .filter(FaceBlob.embedding.isnot(None))
            .filter(func.length(FaceBlob.embedding) != expected_bytes)
            .all()
        )
        if mismatched:
            log.warning(
                "Embedding dimension mismatch: found %d face(s) with %d-byte "
                "embeddings (expected %d bytes for %d-dim). "
                "These will be re-embedded with the current model.",
                len(mismatched),
                len(mismatched[0].embedding) if mismatched[0].embedding else 0,
                expected_bytes,
                expected_dim,
            )
            for blob in mismatched:
                blob.embedding = None
            self._session.commit()
            self._session.expire_all()

    def reembed_all(self, exclude_low_quality: bool = False) -> int:
        """Clear every stored embedding and recompute it from the saved crops.

        Use this after changing ``embedding.crop_mode`` (or the embedding model):
        existing embeddings are crop-geometry-specific and are not comparable to
        ones produced under a different mode.

        NOTE: This only re-runs the *embedding* step against the crops already on
        disk.  Changing ``crop_mode`` also changes crop geometry, so the crops
        themselves must first be regenerated by a full **re-detect** (which
        rewrites each ``crop_path`` file using the new mode).  Re-detect, then
        call this, to make a ``crop_mode`` switch take full effect.

        Returns:
            Number of faces re-embedded.
        """
        from app.db.models import FaceBlob

        cleared = (
            self._session.query(FaceBlob)
            .filter(FaceBlob.embedding.isnot(None))
            .update({FaceBlob.embedding: None}, synchronize_session=False)
        )
        self._session.commit()
        self._session.expire_all()  # drop stale blob state cached on Face rows
        log.info("Re-embed: cleared %d existing embedding(s)", cleared)
        return self.process_pending(exclude_low_quality=exclude_low_quality)

    def _embed_face(self, face: Face) -> bool:
        """Load the crop and compute embedding for a single face.

        Returns:
            ``True`` on success.
        """
        img_bgr = self._load_crop_from_path(face.crop_path)
        if img_bgr is None:
            log.debug("Cannot read crop for face id=%d: %s", face.id, face.crop_path)
            return False

        embedding = self._embedder.embed(img_bgr)
        face.set_embedding(embedding)

        # Marking the parent image as embedding-done is handled lazily by
        # later pipeline stages.
        return True

    def embed_face(self, face: Face) -> bool:
        """Compute and persist the embedding for a single already-saved face.

        Public wrapper around the per-face embedding step.  Used for faces that
        are created outside the batch pipeline (e.g. manually marked faces) so
        they get a full embedding immediately and become comparable in the
        person-assign UI.  Returns ``True`` on success.
        """
        return self._embed_face(face)


def build_embedder(config: AppConfig) -> FaceEmbedder:
    """Construct the configured face embedder from the embedding configuration.

    Dispatches on ``config.embedding.backend``:
      * ``"arcface"``       → :class:`InsightFaceEmbedder` (ArcFace ResNet-50,
                              512-dim, reuses the InsightFace ``buffalo_l`` pack).
      * everything else     → :class:`TFLiteEmbedder` (MobileFaceNet, 192-dim).

    All embedding paths (batch pipeline, manual faces) go through here so a
    person's faces are always embedded with the exact same model/geometry.  If
    the ArcFace backend cannot be loaded (missing package/model) it falls back
    to the TFLite embedder so embedding never hard-fails.
    """
    if config.embedding.backend == "arcface":
        try:
            from app.embeddings.insightface_embedder import InsightFaceEmbedder

            return InsightFaceEmbedder()
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "ArcFace embedder unavailable (%s) — falling back to MobileFaceNet.",
                exc,
            )

    # macOS only: prefer a Core ML MobileFaceNet (Apple Neural Engine / GPU)
    # when a converted model is present.  Returns None off macOS / when absent,
    # so Windows and Linux fall straight through to the unchanged TFLite path.
    from app.embeddings.coreml_embedder import try_build_coreml_embedder

    coreml = try_build_coreml_embedder(config)
    if coreml is not None:
        return coreml

    from app.embeddings.tflite_embedder import TFLiteEmbedder

    return TFLiteEmbedder(
        model_path=config.embedding.model_path,
        embedding_dim=config.embedding.embedding_dim,
        input_size=config.embedding.input_size,
    )


def embed_manual_face(session: Session, face: Face, config: AppConfig) -> bool:
    """Best-effort embedding for a manually-marked face.

    Builds the default embedder and computes the embedding for *face* (which
    must already have a saved crop).  Never raises — embedding is best-effort,
    so a missing model or read error simply leaves the face without a vector
    and returns ``False``.
    """
    try:
        embedder = build_embedder(config)
        return EmbeddingService(session, embedder, config).embed_face(face)
    except Exception as exc:  # noqa: BLE001 — embedding must never break marking
        log.warning("Manual-face embedding failed for face id=%s: %s", face.id, exc)
        return False
