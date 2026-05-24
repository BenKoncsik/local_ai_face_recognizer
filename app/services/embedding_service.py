"""Face embedding service.

Generates embedding vectors for all faces that do not yet have one stored.
Resumable: only processes faces with ``_embedding IS NULL``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional

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

    def process_pending(self) -> int:
        """Generate embeddings for all faces that don't have one yet.

        Returns:
            Number of faces embedded.
        """
        pending: List[Face] = (
            self._session.query(Face)
            .filter(Face._embedding.is_(None))
            .filter(Face.is_excluded == False)  # noqa: E712
            .all()
        )

        total = len(pending)
        log.info("Embedding %d face(s) without vectors", total)

        embedded = 0
        for idx, face in enumerate(pending, start=1):
            self._progress_cb(idx, total, face.id)
            try:
                if self._embed_face(face):
                    embedded += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("Embedding failed for face id=%d: %s", face.id, exc)

            # Commit in batches to avoid large transactions
            if idx % 50 == 0:
                self._session.commit()

        self._session.commit()
        log.info("Embedding complete: %d / %d face(s) embedded", embedded, total)
        return embedded

    def _embed_face(self, face: Face) -> bool:
        """Load the crop and compute embedding for a single face.

        Returns:
            ``True`` on success.
        """
        if not face.crop_path:
            log.debug("Face id=%d has no crop path — skipping", face.id)
            return False

        crop_file = Path(face.crop_path)
        if not crop_file.exists():
            log.debug("Crop file missing for face id=%d: %s", face.id, crop_file)
            return False

        img_bgr = load_image_bgr(str(crop_file))
        if img_bgr is None:
            log.debug("Cannot read crop: %s", crop_file)
            return False

        embedding = self._embedder.embed(img_bgr)
        face.set_embedding(embedding)

        # Marking the parent image as embedding-done is handled lazily by
        # later pipeline stages.
        return True
