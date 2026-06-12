"""Unit tests for the deep recognition classifier (app/deep)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pytest

from app.deep.classifier import DeepFaceClassifier
from app.deep.dataset import TrainingDataset

DIM = 64


@dataclass
class StubConfig:
    """Small/fast hyper-parameters for tests."""

    ensemble_size: int = 2
    hidden_layers: Tuple[int, ...] = (32, 16)
    max_iter: int = 300
    min_class_size: int = 4
    augment_noise_sigma: float = 0.03
    calibration_folds: int = 2
    min_persons_for_ensemble: int = 2
    min_examples_for_ensemble: int = 8
    base_prob_threshold: float = 0.60
    min_prob_threshold: float = 0.35
    min_margin: float = 0.10
    min_prototype_similarity: float = 0.55
    min_sim_margin: float = 0.05
    outlier_similarity: float = 0.42


def _person_vec(axis: int, noise: float = 0.05, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = np.zeros(DIM, dtype=np.float32)
    v[axis] = 1.0
    v += rng.normal(0, noise, DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _dataset(persons: dict[int, list[np.ndarray]]) -> TrainingDataset:
    vectors, labels, face_ids = [], [], []
    fid = 1
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
def trained() -> DeepFaceClassifier:
    persons = {
        1: [_person_vec(0, seed=i) for i in range(6)],
        2: [_person_vec(20, seed=10 + i) for i in range(6)],
        3: [_person_vec(40, seed=20 + i) for i in range(6)],
    }
    clf = DeepFaceClassifier(StubConfig())
    clf.train(_dataset(persons))
    return clf


class TestTraining:
    def test_train_produces_ensemble_mode(self, trained):
        assert trained.is_trained
        assert trained.report is not None
        assert trained.report.mode == "ensemble"
        assert trained.report.n_persons == 3
        assert trained.report.validation_accuracy is not None
        # Clearly separated synthetic people must cross-validate near-perfectly.
        assert trained.report.validation_accuracy > 0.9

    def test_empty_dataset_stays_untrained(self):
        clf = DeepFaceClassifier(StubConfig())
        report = clf.train(TrainingDataset())
        assert report.mode == "empty"
        assert not clf.is_trained
        assert clf.predict(_person_vec(0)).person_id is None

    def test_single_person_uses_prototype_mode(self):
        persons = {7: [_person_vec(5, seed=i) for i in range(4)]}
        clf = DeepFaceClassifier(StubConfig())
        report = clf.train(_dataset(persons))
        assert report.mode == "prototype"
        prediction = clf.predict(_person_vec(5, seed=99))
        assert prediction.person_id == 7
        assert prediction.reason == "assigned"

    def test_progress_callback_fires(self):
        persons = {
            1: [_person_vec(0, seed=i) for i in range(4)],
            2: [_person_vec(30, seed=10 + i) for i in range(4)],
        }
        calls = []
        clf = DeepFaceClassifier(StubConfig())
        clf.train(_dataset(persons), progress_cb=lambda c, t, d: calls.append((c, t)))
        assert calls


class TestPrediction:
    def test_known_person_is_recognized(self, trained):
        prediction = trained.predict(_person_vec(0, seed=123))
        assert prediction.person_id == 1
        assert prediction.reason == "assigned"
        assert prediction.score > 0.5

    def test_each_person_maps_to_itself(self, trained):
        for axis, expected in ((0, 1), (20, 2), (40, 3)):
            prediction = trained.predict(_person_vec(axis, seed=axis + 77))
            assert prediction.person_id == expected

    def test_stranger_is_rejected_as_outlier(self, trained):
        # A vector orthogonal to every training axis — an unknown person or a
        # non-face crop.  Must never be force-assigned.
        stranger = np.zeros(DIM, dtype=np.float32)
        stranger[55] = 1.0
        prediction = trained.predict(stranger)
        assert prediction.person_id is None
        assert prediction.reason == "outlier"

    def test_missing_embedding_is_rejected(self, trained):
        prediction = trained.predict(None)
        assert prediction.person_id is None
        assert prediction.reason == "no_embedding"

    def test_zero_vector_is_rejected(self, trained):
        prediction = trained.predict(np.zeros(DIM, dtype=np.float32))
        assert prediction.person_id is None


class TestClassProbabilities:
    def test_distribution_covers_all_people_and_sums_to_one(self, trained):
        probs = trained.class_probabilities(_person_vec(0, seed=321))
        assert set(probs) == {1, 2, 3}
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_winner_matches_prediction(self, trained):
        for axis, expected in ((0, 1), (20, 2), (40, 3)):
            probs = trained.class_probabilities(_person_vec(axis, seed=axis + 5))
            assert max(probs, key=probs.get) == expected
            assert probs[expected] > 0.5

    def test_untrained_returns_empty(self):
        clf = DeepFaceClassifier(StubConfig())
        assert clf.class_probabilities(_person_vec(0)) == {}

    def test_missing_embedding_returns_empty(self, trained):
        assert trained.class_probabilities(None) == {}

    def test_prototype_mode_returns_empty(self):
        # A single-person cohort stays in prototype mode — no calibrated softmax.
        persons = {7: [_person_vec(5, seed=i) for i in range(4)]}
        clf = DeepFaceClassifier(StubConfig())
        clf.train(_dataset(persons))
        assert clf.class_probabilities(_person_vec(5)) == {}


class TestDecisionPath:
    def test_debug_info_carries_winner_path(self, trained):
        pred, info = trained.predict_debug(_person_vec(0, seed=777), face_id=1)
        assert pred.person_name == "Person 1"
        path = info.decision_path
        assert path is not None
        assert path["winner"] == "Person 1"
        # Columns: input + one per hidden layer + output; gaps between them.
        n_cols = 1 + len(info.layer_activations) + 1
        assert len(path["nodes"]) == n_cols
        assert len(path["edges"]) == n_cols - 1
        assert path["nodes"][-1] == ["Person 1"]
        for gap in path["edges"]:
            for src, dst, strength in gap:
                assert 0.0 <= strength <= 1.0
        # JSON-safe (the live visualizer caches it to disk).
        import json
        json.dumps(path)

    def test_prototype_mode_has_no_path(self):
        persons = {7: [_person_vec(5, seed=i) for i in range(4)]}
        clf = DeepFaceClassifier(StubConfig())
        clf.train(_dataset(persons))
        _pred, info = clf.predict_debug(_person_vec(5, seed=9))
        assert info.decision_path is None


class TestPersistence:
    def test_save_and_load_roundtrip(self, trained, tmp_path):
        path = tmp_path / "model.pkl"
        trained.save(path)

        loaded = DeepFaceClassifier(StubConfig())
        assert loaded.load(path)
        assert loaded.is_trained
        assert loaded.fingerprint == trained.fingerprint
        assert loaded.predict(_person_vec(20, seed=5)).person_id == 2

    def test_load_missing_file_returns_false(self, tmp_path):
        clf = DeepFaceClassifier(StubConfig())
        assert not clf.load(tmp_path / "nope.pkl")

    def test_load_corrupt_file_returns_false(self, tmp_path):
        path = tmp_path / "model.pkl"
        path.write_bytes(b"not a pickle")
        clf = DeepFaceClassifier(StubConfig())
        assert not clf.load(path)


class TestOpenSetSafety:
    """Regression tests for the one-person-avalanche failure mode."""

    def test_sim_floor_never_below_configured_min(self):
        """A person with noisy/diverse labeled faces must not drag the
        prototype floor below min_prototype_similarity — that floor is hard."""
        diverse = [_person_vec(axis, noise=0.3, seed=axis) for axis in (0, 3, 7, 11)]
        persons = {
            1: diverse,
            2: [_person_vec(30, seed=50 + i) for i in range(4)],
        }
        clf = DeepFaceClassifier(StubConfig())
        clf.train(_dataset(persons))
        assert clf._state.sim_floors
        assert all(f >= 0.55 for f in clf._state.sim_floors.values())

    def test_small_cohort_stays_in_prototype_mode(self):
        """Below the person/example minimums the MLP is overconfident —
        the engine must fall back to pure prototype matching."""
        persons = {
            1: [_person_vec(0, seed=i) for i in range(6)],
            2: [_person_vec(20, seed=10 + i) for i in range(6)],
            3: [_person_vec(40, seed=20 + i) for i in range(6)],
        }
        cfg = StubConfig(min_persons_for_ensemble=4)
        clf = DeepFaceClassifier(cfg)
        report = clf.train(_dataset(persons))
        assert report.mode == "prototype"

    def test_ambiguous_face_between_two_lookalikes_is_rejected(self):
        """A face equally similar to two (correlated) people must stay unknown."""

        def unit(*pairs: Tuple[int, float]) -> np.ndarray:
            v = np.zeros(DIM, dtype=np.float32)
            for axis, w in pairs:
                v[axis] = w
            return v / np.linalg.norm(v)

        # Two lookalike persons whose prototypes are 0.8-correlated; the
        # probe sits exactly between them: high similarity to both, ~no margin.
        t1, t2 = unit((0, 1.0)), unit((0, 0.8), (1, 0.6))
        persons = {
            1: [t1, unit((0, 0.9), (2, 0.436))],
            2: [t2, unit((0, 0.72), (1, 0.54), (3, 0.436))],
        }
        cfg = StubConfig(min_persons_for_ensemble=99)  # force prototype mode
        clf = DeepFaceClassifier(cfg)
        clf.train(_dataset(persons))
        ambiguous = (t1 + t2) / np.linalg.norm(t1 + t2)
        prediction = clf.predict(ambiguous)
        assert prediction.person_id is None
        assert prediction.reason == "margin"

    def test_imbalanced_tiny_cohort_does_not_avalanche(self):
        """Mirror of a real incident: 17/5/1 labeled faces with sloppy crops
        assigned nearly every unknown face to the dominant person. A face that
        only vaguely resembles them (cos ≈ 0.45) must be rejected."""
        from app.config import DeepRecognitionConfig

        rng = np.random.default_rng(0)
        dominant = [_person_vec(0, noise=0.35, seed=i) for i in range(17)]
        persons = {
            1: dominant,
            2: [_person_vec(20, noise=0.1, seed=30 + i) for i in range(5)],
            3: [_person_vec(40, seed=60)],
        }
        clf = DeepFaceClassifier(DeepRecognitionConfig(ensemble_size=2, max_iter=200))
        report = clf.train(_dataset(persons))
        # 3 persons / 23 examples is below the ensemble minimums.
        assert report.mode == "prototype"

        vague = _person_vec(0, noise=1.4, seed=999)  # far-off lookalike
        sims = np.vstack(dominant) @ vague
        assert sims.max() < 0.55, "probe must sit below the prototype floor"
        prediction = clf.predict(vague)
        assert prediction.person_id is None


class TestContinualLearning:
    def test_more_examples_increase_confidence(self):
        """The engine must get better at a person as more faces are confirmed."""
        few = {
            1: [_person_vec(0, seed=i) for i in range(2)],
            2: [_person_vec(30, seed=50 + i) for i in range(6)],
        }
        many = {
            1: [_person_vec(0, seed=i) for i in range(14)],
            2: [_person_vec(30, seed=50 + i) for i in range(6)],
        }
        probe = _person_vec(0, noise=0.06, seed=999)

        clf_few = DeepFaceClassifier(StubConfig())
        clf_few.train(_dataset(few))
        clf_many = DeepFaceClassifier(StubConfig())
        clf_many.train(_dataset(many))

        p_few = clf_few.predict(probe)
        p_many = clf_many.predict(probe)
        assert p_many.person_id == 1
        assert p_many.score >= p_few.score - 1e-6

    def test_fingerprint_changes_with_data(self):
        ds1 = _dataset({1: [_person_vec(0, seed=1)], 2: [_person_vec(9, seed=2)]})
        ds2 = _dataset(
            {1: [_person_vec(0, seed=1), _person_vec(0, seed=3)],
             2: [_person_vec(9, seed=2)]}
        )
        assert ds1.fingerprint() != ds2.fingerprint()
