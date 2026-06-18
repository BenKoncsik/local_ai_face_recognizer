"""Tests for deep recognition debug/diagnostics helpers."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier

from app.deep.classifier import DeepPrediction
from app.deep.debug_info import (
    DECISION_VERSION,
    DeepDebugInfo,
    GateResult,
    compute_decision_path,
    compute_layer_activations,
    decision_to_dict,
)


def _prediction(**overrides) -> DeepPrediction:
    base = dict(
        person_id=1,
        person_name="Alice",
        score=0.91,
        probability=0.88,
        similarity=0.82,
        margin=0.35,
        reason="assigned",
    )
    base.update(overrides)
    return DeepPrediction(**base)


def _debug_info(**overrides) -> DeepDebugInfo:
    base = dict(
        face_id=42,
        crop_path="/tmp/crop.jpg",
        embedding_norm=1.0,
        embedding_top_dims=[(0, 0.9), (3, 0.4)],
        all_similarities={"Alice": 0.82, "Bob": 0.31},
        layer_activations=[np.array([0.5, 0.1, 0.0])],
        output_probs={"Alice": 0.88, "Bob": 0.12},
        gates=[
            GateResult(name="prob", passed=True, value=0.88, threshold=0.60),
            GateResult(name="margin", passed=True, value=0.35, threshold=0.10),
        ],
        prediction=_prediction(),
        mode="ensemble",
    )
    base.update(overrides)
    return DeepDebugInfo(**base)


class TestDecisionToDict:
    def test_serialises_compact_summary(self) -> None:
        payload = decision_to_dict(_debug_info())
        assert payload["version"] == DECISION_VERSION
        assert payload["engine"] == "deep"
        assert payload["mode"] == "ensemble"
        assert payload["person_name"] == "Alice"
        assert payload["gates"][0]["name"] == "prob"
        assert payload["top_persons"][0] == ["Alice", pytest.approx(0.88)]
        assert payload["top_similarities"][0][0] == "Alice"

    def test_limits_ranked_lists_to_eight(self) -> None:
        probs = {f"P{i}": 1.0 - i * 0.05 for i in range(12)}
        sims = {f"S{i}": 1.0 - i * 0.04 for i in range(12)}
        payload = decision_to_dict(_debug_info(output_probs=probs, all_similarities=sims))
        assert len(payload["top_persons"]) == 8
        assert len(payload["top_similarities"]) == 8


class TestComputeLayerActivations:
    def test_returns_relu_hidden_layers(self) -> None:
        mlp = MLPClassifier(hidden_layer_sizes=(4, 3), activation="relu", max_iter=1)
        x = np.random.default_rng(0).normal(size=(8, 6))
        y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        mlp.fit(x, y)
        vec = x[0]
        activations = compute_layer_activations(mlp, vec)
        assert len(activations) == 2
        assert activations[0].shape == (4,)
        assert activations[1].shape == (3,)
        assert np.all(activations[0] >= 0.0)

    def test_empty_for_unfitted_mlp(self) -> None:
        mlp = MLPClassifier(hidden_layer_sizes=(4,), max_iter=1)
        activations = compute_layer_activations(mlp, np.zeros(6))
        assert activations == []

    def test_empty_for_non_mlp_object(self) -> None:
        class _NotAnMlp:
            pass

        assert compute_layer_activations(_NotAnMlp(), np.zeros(3)) == []


class _FakeMlp:
    def __init__(self) -> None:
        self.coefs_ = [
            np.array([[1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]),
            np.array([[0.2, 0.8], [0.1, 0.9]]),
        ]


class TestComputeDecisionPath:
    def test_traces_winner_through_hidden_layers(self) -> None:
        mlp = _FakeMlp()
        vec = np.array([0.9, 0.1, 0.4], dtype=np.float64)
        layer_activations = [
            np.array([0.9, 0.0, 0.4]),
            np.array([0.9, 0.4]),
        ]
        embedding_top_dims = [(0, 0.9), (2, 0.4), (1, 0.1)]
        output_probs = {"Alice": 0.2, "Bob": 0.8}
        path = compute_decision_path(
            mlp,
            vec,
            layer_activations,
            embedding_top_dims,
            output_probs,
            class_names=["Alice", "Bob"],
            top_n=3,
            fan_k=2,
        )
        assert path is not None
        assert path["winner"] == "Bob"
        assert path["nodes"][-1] == ["Bob"]
        assert path["edges"]

    def test_returns_none_without_activations(self) -> None:
        mlp = MLPClassifier(hidden_layer_sizes=(4,), max_iter=1)
        x = np.zeros((4, 3))
        y = np.array([0, 0, 1, 1])
        mlp.fit(x, y)
        assert (
            compute_decision_path(
                mlp,
                x[0],
                [],
                [(0, 1.0)],
                {"Class0": 0.5, "Class1": 0.5},
                class_names=["Class0", "Class1"],
            )
            is None
        )
