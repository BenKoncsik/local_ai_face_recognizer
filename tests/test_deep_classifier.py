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
    base_prob_threshold: float = 0.60
    min_prob_threshold: float = 0.35
    min_margin: float = 0.10
    min_prototype_similarity: float = 0.55
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
        probe = _person_vec(0, noise=0.18, seed=999)

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
