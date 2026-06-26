"""Abstract face embedder interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class FaceEmbedder(ABC):
    """Converts a face crop (BGR image array) to a fixed-length embedding vector."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Length of the output embedding vector."""

    @abstractmethod
    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        """Generate an L2-normalised embedding for *face_bgr*.

        Args:
            face_bgr: BGR uint8 numpy array of a face crop (any size —
                      implementations must resize internally).

        Returns:
            1-D float32 numpy array of length :attr:`embedding_dim`.
        """

    def embed_batch(self, faces_bgr: List[np.ndarray]) -> List[np.ndarray]:
        """Embed several crops at once; returns one vector per input crop.

        Default implementation simply calls :meth:`embed` per crop — backends
        that support true batched inference (e.g. the TFLite embedder) override
        this to run the whole batch through the model in one invocation, which
        is substantially faster.  The returned list has the same length and
        order as *faces_bgr*.
        """
        return [self.embed(face_bgr) for face_bgr in faces_bgr]

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} dim={self.embedding_dim}>"
