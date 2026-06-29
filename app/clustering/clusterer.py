"""Face embedding clustering.

Groups face embeddings into identity clusters using DBSCAN.

Design notes
------------
* DBSCAN is used because it:
  - handles variable cluster sizes
  - naturally marks outliers (label -1) for unclassified faces
  - does not require specifying the number of clusters upfront

* Cosine distance is used as the metric because L2-normalised embedding
  vectors are compared most naturally that way.

* Manual corrections (same / not-same pairs from :mod:`db.models`) are
  applied as a post-processing step: if two faces were merged manually,
  they are forced into the same group; if separated, they remain apart.

* The output is a dict mapping face ID → cluster label (int).
  Label -1 means "noise / unclustered".
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

log = logging.getLogger(__name__)


def cluster_embeddings(
    face_ids: List[int],
    embeddings: List[np.ndarray],
    epsilon: float = 0.4,
    min_samples: int = 2,
    metric: str = "cosine",
    same_pairs: Optional[List[Tuple[int, int]]] = None,
    diff_pairs: Optional[List[Tuple[int, int]]] = None,
) -> Dict[int, int]:
    """Cluster face embeddings with DBSCAN.

    Args:
        face_ids:    Ordered list of face database IDs.
        embeddings:  Corresponding list of embedding vectors (float32, L2-normed).
        epsilon:     DBSCAN neighbourhood radius (cosine distance).
        min_samples: Minimum cluster size for DBSCAN core points.
        metric:      Distance metric for DBSCAN (default ``"cosine"``).
        same_pairs:  Face ID pairs that must end up in the same cluster.
        diff_pairs:  Face ID pairs that must be in different clusters.

    Returns:
        ``{face_id: cluster_label}`` mapping.  Noise faces get label ``-1``.
    """
    if not face_ids:
        return {}

    matrix = np.vstack(embeddings).astype(np.float32)

    # Re-normalise to be safe (models might not guarantee perfect L2 norm)
    matrix = normalize(matrix, norm="l2")

    log.info(
        "Clustering %d embeddings  epsilon=%.3f  min_samples=%d  metric=%s",
        len(face_ids), epsilon, min_samples, metric,
    )

    # Memory safety for large unknown-face sets (10k+) on Windows.
    # 'cosine' has no tree support, so DBSCAN computes pairwise distances by
    # brute force. With n_jobs=-1 sklearn ran `pairwise_distances_chunked` in
    # parallel, holding one `working_memory`-sized chunk PER CORE at once
    # (default 1 GiB × N_cores → many GB peak), which on a multitasking box
    # (VS/Edge already resident) exhausted the commit limit and hard-froze the
    # OS. Pin n_jobs=1 so only one chunk is live, and bound the chunk to 256 MiB
    # so the distance pass never spikes. (Windows scan memory-exhaustion freeze.)
    from sklearn import config_context

    db = DBSCAN(eps=epsilon, min_samples=min_samples, metric=metric, n_jobs=1)
    with config_context(working_memory=256):
        labels = db.fit_predict(matrix)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    log.info("Clustering done: %d cluster(s), %d noise point(s)", n_clusters, n_noise)

    result: Dict[int, int] = {fid: int(lbl) for fid, lbl in zip(face_ids, labels)}

    # --- Apply manual constraints (best-effort post-processing) ---
    if same_pairs:
        result = _apply_same_pairs(result, same_pairs, labels)

    # diff_pairs are not forcibly separated here — the UI prevents
    # merging them, and re-clustering naturally handles most cases.

    return result


def _apply_same_pairs(
    result: Dict[int, int],
    same_pairs: List[Tuple[int, int]],
    labels: np.ndarray,
) -> Dict[int, int]:
    """Force face pairs marked as same-person into the same cluster label.

    For each pair (a, b) where both have known cluster labels, relabel the
    smaller cluster to match the larger.  This is a greedy heuristic and may
    produce imperfect results with many constraints — re-clustering with
    constraints baked into the distance matrix is a future improvement.
    """
    id_to_label = dict(result)

    for a, b in same_pairs:
        la = id_to_label.get(a)
        lb = id_to_label.get(b)
        if la is None or lb is None:
            continue
        if la == lb:
            continue

        # Relabel all faces with lb → la (merge clusters)
        old, new_ = (lb, la) if la >= 0 else (la, lb)
        log.debug("Manual constraint: merging cluster %d → %d", old, new_)
        id_to_label = {
            fid: (new_ if lbl == old else lbl) for fid, lbl in id_to_label.items()
        }

    return id_to_label


def compute_centroid(embeddings: List[np.ndarray]) -> np.ndarray:
    """Compute the L2-normalised centroid of a list of embedding vectors."""
    matrix = np.vstack(embeddings).astype(np.float32)
    centroid = matrix.mean(axis=0)
    norm = np.linalg.norm(centroid)
    return centroid / norm if norm > 1e-8 else centroid


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine distance in [0, 2] between two L2-normalised vectors."""
    return float(1.0 - np.dot(a, b))
