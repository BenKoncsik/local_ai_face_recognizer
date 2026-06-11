"""Debug/diagnostics data structures for the deep recognition engine.

When the debug mode is active, :meth:`DeepFaceClassifier.predict_debug` returns
one :class:`DeepDebugInfo` alongside the regular :class:`DeepPrediction`.  The
info object is self-contained: it carries everything the visualisation window
and the debug log need without touching the database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from app.deep.classifier import DeepPrediction


@dataclass
class GateResult:
    """Result of one open-set rejection gate."""

    name: str           # "outlier" | "prob" | "margin" | "sim_floor" | "sim_margin"
    passed: bool        # True → gate let the face through
    value: float        # measured value (similarity / probability / margin)
    threshold: float    # gate threshold


@dataclass
class DeepDebugInfo:
    """All diagnostic data produced for one face prediction."""

    face_id: int
    crop_path: Optional[str]

    # Raw embedding statistics (not the full 192-dim vector to keep logs compact)
    embedding_norm: float
    embedding_top_dims: List[Tuple[int, float]]   # top-10 (dim_index, value)

    # Per-person cosine similarities (name → max similarity to any training face)
    all_similarities: Dict[str, float]

    # Hidden-layer activations from the first ensemble member (or empty in
    # prototype mode).  Each array has shape (n_neurons,).
    layer_activations: List[np.ndarray]

    # Output probability distribution (person_name → probability) from the
    # ensemble average, sorted descending.
    output_probs: Dict[str, float]

    # Gate results in evaluation order
    gates: List[GateResult]

    # Final decision (same object as returned by predict/predict_debug)
    prediction: "DeepPrediction"

    # Which mode was used
    mode: str   # "ensemble" | "prototype" | "empty"


def compute_layer_activations(mlp, vec: np.ndarray) -> List[np.ndarray]:
    """Run a forward pass through *mlp* and return hidden-layer activations.

    Uses the ReLU formula layer-by-layer, matching sklearn's implementation.
    The output layer is excluded (use ``mlp.predict_proba`` for that).

    Returns an empty list when the MLP has not been fitted yet.
    """
    try:
        coefs = mlp.coefs_
        intercepts = mlp.intercepts_
    except AttributeError:
        return []

    activations: List[np.ndarray] = []
    x = vec.reshape(1, -1).astype(np.float64)
    # All layers except the last (output) use ReLU in our config
    for coef, intercept in zip(coefs[:-1], intercepts[:-1]):
        x = x @ coef + intercept
        x = np.maximum(0.0, x)   # ReLU
        activations.append(x[0].copy())
    return activations
