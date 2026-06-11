"""Deep (AI ensemble) probability overlay in the shared match-scoring helper.

These exercise the pure scoring path — loading a saved deep model and overlaying
its calibrated probabilities — without touching the database, so they stay fast
and focused on the new behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pytest

from app.deep.classifier import DeepFaceClassifier
from app.deep.dataset import TrainingDataset
from app.services import match_scoring

DIM = 128


# ── Minimal stubs mirroring tests/test_deep_classifier.py ────────────────────
@dataclass
class _DeepCfg:
    # Small thresholds so a tiny synthetic cohort still trains an ensemble.
    model_dir: str = "."
    enabled: bool = True
    ensemble_size: int = 2
    hidden_layers: Tuple[int, ...] = (32,)
    max_iter: int = 300
    min_persons_for_ensemble: int = 2
    min_examples_for_ensemble: int = 8
    min_class_size: int = 4
    calibration_folds: int = 2


@dataclass
class _AppCfg:
    deep_recognition: _DeepCfg = field(default_factory=_DeepCfg)


def _person_vec(axis: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = np.zeros(DIM, dtype=np.float32)
    v[axis] = 1.0
    v += rng.normal(0, 0.05, DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _dataset(persons):
    vectors, labels, face_ids, fid = [], [], [], 1
    for pid, vecs in persons.items():
        for v in vecs:
            vectors.append(v)
            labels.append(pid)
            face_ids.append(fid)
            fid += 1
    return TrainingDataset(
        embeddings=np.vstack(vectors).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        face_ids=face_ids,
        person_names={pid: f"Person {pid}" for pid in persons},
    )


@pytest.fixture()
def trained_config(tmp_path):
    """Train an ensemble model, persist it, and return a config pointing at it."""
    persons = {
        1: [_person_vec(0, seed=i) for i in range(6)],
        2: [_person_vec(20, seed=10 + i) for i in range(6)],
        3: [_person_vec(40, seed=20 + i) for i in range(6)],
    }
    clf = DeepFaceClassifier(_DeepCfg())
    clf.train(_dataset(persons))
    assert clf.is_trained and clf._state.mode == "ensemble"
    clf.save(tmp_path / "deep_face_model.pkl")

    # Fresh cache per test so the keyed-by-mtime loader reloads our model.
    match_scoring._deep_cache["key"] = None
    match_scoring._deep_cache["clf"] = None
    return _AppCfg(deep_recognition=_DeepCfg(model_dir=str(tmp_path)))


class TestDeepProbabilities:
    def test_returns_distribution_for_known_query(self, trained_config):
        probs = match_scoring._deep_probabilities(_person_vec(0, seed=99), trained_config)
        assert set(probs) == {1, 2, 3}
        assert max(probs, key=probs.get) == 1

    def test_no_config_uses_default_dir_and_misses(self):
        # No model at the default model_dir → empty, never raises.
        match_scoring._deep_cache["key"] = None
        assert match_scoring._deep_probabilities(_person_vec(0), None) == {}

    def test_disabled_engine_returns_empty(self, trained_config):
        trained_config.deep_recognition.enabled = False
        assert match_scoring._deep_probabilities(_person_vec(0), trained_config) == {}


class TestOverlay:
    def test_overrides_known_keeps_unknown(self, trained_config):
        # 1..3 are known to the deep model; 99 is not (e.g. an unknown person
        # only the cosine fallback ranked).
        base = {1: 0.30, 2: 0.31, 3: 0.29, 99: 0.40}
        merged = match_scoring._overlay_deep(base, _person_vec(0, seed=7), trained_config)

        assert merged[99] == 0.40                       # untouched cosine score
        assert max((1, 2, 3), key=merged.get) == 1      # AI winner re-ranks to top
        assert merged[1] > 0.5                           # real probability, not cosine

    def test_no_model_returns_scores_unchanged(self):
        match_scoring._deep_cache["key"] = None
        base = {1: 0.3, 2: 0.7}
        assert match_scoring._overlay_deep(base, _person_vec(0), None) == base
