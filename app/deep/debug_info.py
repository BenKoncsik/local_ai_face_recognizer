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

    # Strongest contribution path through the network towards the winning
    # output (see :func:`compute_decision_path`); None in prototype mode.
    decision_path: Optional[Dict] = None


# Schema version of the dict produced by :func:`decision_to_dict`.
DECISION_VERSION = 1


def decision_to_dict(info: "DeepDebugInfo") -> Dict:
    """Serialize a :class:`DeepDebugInfo` into a compact, JSON-safe dict.

    Keeps only what the stored decision graph needs (the gate flow + winner
    summary + the top ranked persons) — NOT the heavy per-neuron layer
    activations or the full 192-dim embedding.  The result is small enough to
    persist on every auto-assignment row and round-trips back into the same
    flow-chart widget as the live AI debug view.
    """
    pred = info.prediction
    top_persons = sorted(
        (info.output_probs or {}).items(), key=lambda kv: -kv[1]
    )[:8]
    top_sims = sorted(
        (info.all_similarities or {}).items(), key=lambda kv: -kv[1]
    )[:8]
    return {
        "version": DECISION_VERSION,
        "engine": "deep",
        "mode": info.mode,
        "reason": pred.reason,
        "person_name": pred.person_name,
        "score": float(pred.score),
        "similarity": float(pred.similarity),
        "probability": float(pred.probability),
        "margin": float(pred.margin),
        "embedding_norm": float(info.embedding_norm),
        "gates": [
            {
                "name": g.name,
                "passed": bool(g.passed),
                "value": float(g.value),
                "threshold": float(g.threshold),
            }
            for g in info.gates
        ],
        "top_persons": [[str(n), float(p)] for n, p in top_persons],
        "top_similarities": [[str(n), float(s)] for n, s in top_sims],
    }


def compute_decision_path(
    mlp,
    vec: np.ndarray,
    layer_activations: List[np.ndarray],
    embedding_top_dims: List[Tuple[int, float]],
    output_probs: Dict[str, float],
    class_names: List[str],
    top_n: int = 10,
    fan_k: int = 3,
) -> Optional[Dict]:
    """Trace the strongest contribution path to the winning output neuron.

    Backtracks from the winner through every hidden layer down to the input:
    at each layer boundary the *fan_k* displayed source neurons with the
    largest positive contribution (activation × connection weight) to each
    active destination neuron are kept.  Node selection mirrors the top-N
    display of :class:`NeuralNetworkGraphWidget` (labels ``d<i>`` for input
    dims, ``n<i>`` for hidden neurons, person name for the output), so the
    result can be overlaid 1:1 on the rendered graph.

    Returns a JSON-safe dict or None when the MLP is unfitted / there is
    nothing to trace (e.g. prototype mode)::

        {
            "winner": str,
            # active node labels per graph column: input, hidden 1..H, output
            "nodes":  [[labels...], ...],
            # edges per column boundary (input→h1, h1→h2, ..., hH→output)
            "edges":  [[[src_label, dst_label, strength 0..1], ...], ...],
        }
    """
    try:
        coefs = mlp.coefs_
    except AttributeError:
        return None
    if not layer_activations or not output_probs or not embedding_top_dims:
        return None

    # Displayed node indices per column — must match the widget's selection.
    in_dims = [
        int(i)
        for i, _ in sorted(embedding_top_dims, key=lambda x: -abs(x[1]))[:top_n]
    ]
    hidden_idx = [
        np.argsort(np.abs(np.asarray(a, dtype=np.float64)))[::-1][:top_n]
        for a in layer_activations
    ]
    winner = max(output_probs.items(), key=lambda kv: kv[1])[0]
    col_of_name = {n: j for j, n in enumerate(class_names)}
    n_hidden = len(layer_activations)
    if winner not in col_of_name or len(coefs) != n_hidden + 1:
        return None
    winner_col = col_of_name[winner]

    # nodes[c] / edges[g]: column c = input, hidden 1..H, output;
    # boundary g = c → c+1.
    nodes: List[set] = [set() for _ in range(n_hidden + 2)]
    edges: List[List] = [[] for _ in range(n_hidden + 1)]
    nodes[-1].add(winner)

    def _keep_top(gap: int, contribs: List[Tuple[str, str, float]]) -> List[str]:
        """Record the strongest positive contributions of one boundary."""
        positive = [c for c in contribs if c[2] > 0.0]
        positive.sort(key=lambda c: -c[2])
        kept = positive[:fan_k]
        if not kept:
            return []
        peak = kept[0][2]
        for src, dst, strength in kept:
            edges[gap].append([src, dst, float(strength / peak)])
            nodes[gap].add(src)
        return [src for src, _, _ in kept]

    # Output ← last hidden layer.
    act_last = np.asarray(layer_activations[-1], dtype=np.float64)
    active = _keep_top(n_hidden, [
        (f"n{int(j)}", winner, float(act_last[j] * coefs[-1][j, winner_col]))
        for j in hidden_idx[-1]
    ])

    # Hidden layer l ← hidden layer l-1 (1-based layers; gap index = l-1... +1
    # for the input column → boundary h_{l-1}→h_l is edges[l - 1 + 1] = edges[l]).
    for layer in range(n_hidden - 1, 0, -1):
        act_prev = np.asarray(layer_activations[layer - 1], dtype=np.float64)
        next_active: List[str] = []
        for dst_label in active:
            dst = int(dst_label[1:])
            next_active.extend(_keep_top(layer, [
                (
                    f"n{int(j)}",
                    dst_label,
                    float(act_prev[j] * coefs[layer][j, dst]),
                )
                for j in hidden_idx[layer - 1]
            ]))
        active = list(dict.fromkeys(next_active))

    # First hidden layer ← input dims.
    x = np.asarray(vec, dtype=np.float64).ravel()
    for dst_label in active:
        dst = int(dst_label[1:])
        _keep_top(0, [
            (f"d{d}", dst_label, float(x[d] * coefs[0][d, dst]))
            for d in in_dims
        ])

    if not any(edges):
        return None
    return {
        "winner": winner,
        "nodes": [sorted(col) for col in nodes],
        "edges": edges,
    }


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
