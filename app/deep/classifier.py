"""DeepFaceClassifier — neural-network identity classifier with open-set rejection.

The classifier is the learning core of the new recognition path:

* **Model** — an ensemble of feed-forward neural networks (multi-layer
  perceptrons) trained on the deep CNN face embeddings of every labeled face.
  Ensemble averaging smooths out individual network quirks and gives usable
  confidence estimates.

* **Continual learning** — the ensemble is retrained from scratch on every
  run with *all* trusted labels, so each newly confirmed face immediately
  improves the model.  A fingerprint of the labeled data lets callers skip
  retraining when nothing changed.

* **Open-set rejection** — a photo archive always contains people the model
  has never seen (and the detector occasionally produces non-face boxes).
  Three independent gates keep those from being force-assigned:

  1. *outlier gate* — the embedding must resemble at least one known training
     example at all (non-faces and complete strangers fail here);
  2. *probability gate* — the ensemble probability for the winning person must
     exceed a per-person threshold calibrated by cross-validation;
  3. *prototype gate* — the embedding must be cosine-close to at least one
     real training example of the winning person.

* **Accuracy over speed** — training uses every CPU core (ensemble members and
  calibration folds run in parallel) and is allowed to take its time.

Small classes are augmented with jittered / interpolated synthetic embeddings
so people with only a handful of confirmed faces still train a stable class.
"""

from __future__ import annotations

import logging
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.deep.dataset import TrainingDataset

log = logging.getLogger(__name__)

_MODEL_FILE_VERSION = 1


@dataclass
class DeepPrediction:
    """Result of classifying one embedding."""

    person_id: Optional[int]      # None → rejected (stays unknown)
    person_name: Optional[str]
    score: float                  # blended confidence [0, 1] for the assignment
    probability: float            # raw ensemble probability of the winner
    similarity: float             # best cosine similarity to the winner's examples
    margin: float                 # probability gap to the runner-up person
    reason: str                   # "assigned" | rejection reason


@dataclass
class DeepTrainingReport:
    """Diagnostics from one training run."""

    n_persons: int = 0
    n_examples: int = 0
    n_augmented: int = 0
    validation_accuracy: Optional[float] = None
    duration_seconds: float = 0.0
    fingerprint: str = ""
    mode: str = "ensemble"        # "ensemble" | "prototype" | "empty"


@dataclass
class _TrainedState:
    """Everything predict() needs, picklable as one unit."""

    mode: str = "empty"
    classes: List[int] = field(default_factory=list)
    person_names: Dict[int, str] = field(default_factory=dict)
    members: List[object] = field(default_factory=list)
    train_matrix: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float32)
    )
    train_labels: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    # True for each row of train_matrix that came from a manual/human-confirmed source.
    # None when loaded from an older persisted model that predates this field.
    train_manual_mask: Optional[np.ndarray] = field(default=None)
    prob_thresholds: Dict[int, float] = field(default_factory=dict)
    sim_floors: Dict[int, float] = field(default_factory=dict)
    fingerprint: str = ""
    trained_at: float = 0.0
    report: Optional[DeepTrainingReport] = None


class DeepFaceClassifier:
    """Ensemble MLP classifier over face embeddings with open-set rejection."""

    def __init__(self, config) -> None:
        # ``config`` is an app.config.DeepRecognitionConfig; kept duck-typed so
        # the module stays importable in isolation (tests pass a stub).
        self._config = config
        self._state = _TrainedState()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._state.mode in ("ensemble", "prototype")

    @property
    def person_ids(self) -> tuple[int, ...]:
        """Person IDs this model was trained to recognise (its class labels).

        Used to detect a *stale* model — one whose people no longer exist in the
        current database (e.g. after the DB was reset but the model kept).
        """
        return tuple(int(c) for c in self._state.classes)

    @property
    def fingerprint(self) -> str:
        return self._state.fingerprint

    @property
    def trained_hidden_layers(self) -> Optional[tuple]:
        """Hidden-layer layout of the persisted ensemble (None without MLPs)."""
        if not self._state.members:
            return None
        sizes = getattr(self._state.members[0], "hidden_layer_sizes", None)
        if sizes is None:
            return None
        if isinstance(sizes, (list, tuple)):
            return tuple(int(s) for s in sizes)
        return (int(sizes),)

    @property
    def report(self) -> Optional[DeepTrainingReport]:
        return self._state.report

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        dataset: TrainingDataset,
        progress_cb=None,
    ) -> DeepTrainingReport:
        """(Re)train the model from *dataset*; returns a diagnostics report.

        progress_cb: optional ``(current: int, total: int, detail: str)``.
        """
        started = time.monotonic()
        report = DeepTrainingReport(
            n_persons=dataset.n_persons,
            n_examples=dataset.n_examples,
            fingerprint=dataset.fingerprint(),
        )

        if dataset.n_examples == 0 or dataset.n_persons == 0:
            self._state = _TrainedState(mode="empty", report=report)
            report.mode = "empty"
            log.info("Deep training skipped: no labeled examples")
            return report

        manual_mask = (
            np.asarray(dataset.manual_flags, dtype=bool)
            if dataset.manual_flags
            else None
        )
        state = _TrainedState(
            mode="prototype",
            classes=sorted(set(dataset.labels.tolist())),
            person_names=dict(dataset.person_names),
            train_matrix=dataset.embeddings.copy(),
            train_labels=dataset.labels.copy(),
            train_manual_mask=manual_mask,
            fingerprint=report.fingerprint,
            trained_at=time.time(),
        )
        state.sim_floors = self._calibrate_sim_floors(dataset)

        # A discriminative network needs a real cohort to produce meaningful
        # probabilities: with a handful of people (or examples) the softmax is
        # grossly overconfident for the dominant person and open-set rejection
        # degenerates — stay in pure prototype matching until there is data.
        enough_for_ensemble = (
            dataset.n_persons >= int(self._cfg("min_persons_for_ensemble", 4))
            and dataset.n_examples >= int(self._cfg("min_examples_for_ensemble", 30))
        )
        if enough_for_ensemble:
            x_aug, y_aug, n_synthetic = self._augment(
                dataset.embeddings, dataset.labels
            )
            report.n_augmented = n_synthetic

            total_steps = self._ensemble_size() + self._n_folds(dataset) + 1
            step = [0]

            def tick(detail: str) -> None:
                step[0] += 1
                if progress_cb is not None:
                    progress_cb(step[0], total_steps, detail)

            state.members = self._fit_ensemble(x_aug, y_aug, tick)
            state.mode = "ensemble"
            report.mode = "ensemble"

            thresholds, accuracy = self._calibrate_prob_thresholds(dataset, tick)
            state.prob_thresholds = thresholds
            report.validation_accuracy = accuracy
            tick("finalising")
        else:
            # Too few people/examples for a discriminative network — pure
            # prototype matching with the hard similarity floor is safer.
            report.mode = "prototype"
            if progress_cb is not None:
                progress_cb(1, 1, "prototype mode")

        report.duration_seconds = time.monotonic() - started
        state.report = report
        self._state = state
        log.info(
            "Deep training done: mode=%s persons=%d examples=%d (+%d synthetic) "
            "val_acc=%s in %.1fs",
            report.mode, report.n_persons, report.n_examples, report.n_augmented,
            f"{report.validation_accuracy:.3f}" if report.validation_accuracy is not None else "n/a",
            report.duration_seconds,
        )
        return report

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, embedding: Optional[np.ndarray]) -> DeepPrediction:
        """Classify one embedding; rejected predictions carry person_id=None."""
        rejected = DeepPrediction(
            person_id=None, person_name=None,
            score=0.0, probability=0.0, similarity=0.0, margin=0.0,
            reason="untrained",
        )
        if not self.is_trained:
            return rejected
        vec = self._unit(embedding)
        if vec is None:
            rejected.reason = "no_embedding"
            return rejected

        state = self._state
        if state.train_matrix.ndim == 2 and state.train_matrix.shape[1] != vec.shape[0]:
            rejected.reason = "no_embedding"
            return rejected
        sims = state.train_matrix @ vec
        best_global_sim = float(np.max(sims)) if sims.size else 0.0

        # Gate 1: must resemble *someone* we know — non-faces and complete
        # strangers fail here and are never force-assigned.
        if best_global_sim < self._cfg("outlier_similarity", 0.42):
            return DeepPrediction(
                person_id=None, person_name=None, score=0.0,
                probability=0.0, similarity=best_global_sim, margin=0.0,
                reason="outlier",
            )

        if state.mode == "prototype":
            return self._predict_prototype(vec, sims)
        return self._predict_ensemble(vec, sims)

    def class_probabilities(self, embedding: Optional[np.ndarray]) -> Dict[int, float]:
        """Return ``{person_id: ensemble probability}`` for *embedding*.

        This is the calibrated open-set softmax the AI engine itself uses to
        decide assignments — useful for showing a real "AI probability" next to
        each candidate in person-selector popups.

        Only the **ensemble** mode produces meaningful probabilities; prototype
        mode (too few people/examples for a discriminative network) has no
        softmax, so an empty mapping is returned and callers fall back to plain
        similarity ordering.  An empty mapping is likewise returned for a
        missing / degenerate embedding or an untrained model.
        """
        state = self._state
        vec = self._unit(embedding)
        if vec is None or state.mode != "ensemble" or not state.members:
            return {}
        probs = self._ensemble_proba(vec.reshape(1, -1))[0]
        return {
            int(state.classes[i]): float(p)
            for i, p in enumerate(probs)
        }

    def predict_debug(
        self,
        embedding: Optional[np.ndarray],
        face_id: int = -1,
        crop_path: Optional[str] = None,
    ) -> Tuple["DeepPrediction", "DeepDebugInfo"]:
        """Like :meth:`predict` but also returns a :class:`DeepDebugInfo` object.

        The extra work (activation forward pass, gate recording) only happens
        here — the hot path :meth:`predict` is not affected.
        """
        from app.deep.debug_info import (
            DeepDebugInfo,
            GateResult,
            compute_decision_path,
            compute_layer_activations,
        )

        state = self._state
        gates: List[GateResult] = []
        layer_activations: List[np.ndarray] = []
        output_probs: Dict[int, float] = {}
        all_similarities: Dict[str, float] = {}

        vec = self._unit(embedding)
        dim_mismatch = (
            vec is not None
            and state.train_matrix.ndim == 2
            and state.train_matrix.shape[1] != vec.shape[0]
        )
        if vec is None or not self.is_trained or dim_mismatch:
            pred = self.predict(embedding)
            return pred, DeepDebugInfo(
                face_id=face_id, crop_path=crop_path,
                embedding_norm=0.0, embedding_top_dims=[],
                all_similarities={}, layer_activations=[],
                output_probs={}, gates=[], prediction=pred,
                mode=state.mode,
            )

        sims = state.train_matrix @ vec
        # Per-person max similarity
        for pid in state.classes:
            mask = state.train_labels == pid
            if mask.any():
                all_similarities[state.person_names.get(pid, str(pid))] = float(np.max(sims[mask]))

        # Gate 1: outlier
        best_global_sim = float(np.max(sims)) if sims.size else 0.0
        outlier_thr = self._cfg("outlier_similarity", 0.42)
        gates.append(GateResult("outlier", best_global_sim >= outlier_thr,
                                best_global_sim, outlier_thr))

        # Activations from first ensemble member (ensemble mode only)
        if state.mode == "ensemble" and state.members:
            layer_activations = compute_layer_activations(state.members[0], vec)
            probs_avg = self._ensemble_proba(vec.reshape(1, -1))[0]
            output_probs = {
                state.person_names.get(int(state.classes[i]), str(state.classes[i])): float(p)
                for i, p in enumerate(probs_avg)
            }
            if best_global_sim >= outlier_thr:
                order = np.argsort(probs_avg)[::-1]
                top_idx = int(order[0])
                second_idx = int(order[1]) if len(order) > 1 else top_idx
                pid = int(state.classes[top_idx])
                prob = float(probs_avg[top_idx])
                margin = prob - float(probs_avg[second_idx])
                thr = state.prob_thresholds.get(pid, self._cfg("base_prob_threshold", 0.60))
                gates.append(GateResult("prob", prob >= thr, prob, thr))
                min_margin = self._cfg("min_margin", 0.10)
                gates.append(GateResult("margin", margin >= min_margin, margin, min_margin))
                mask = state.train_labels == pid
                sim = float(np.max(sims[mask])) if mask.any() else 0.0
                default_floor = self._cfg("min_prototype_similarity", 0.55)
                floor = max(state.sim_floors.get(pid, default_floor), default_floor)
                # Manual-anchor bypass: gate passes when nearest training example
                # for the winner is a human-confirmed face above the anchor threshold.
                sim_floor_passed = sim >= floor or self._has_manual_anchor(sims, pid)
                gates.append(GateResult("sim_floor", sim_floor_passed, sim, floor))
                other_mask = state.train_labels != pid
                runner_sim = float(np.max(sims[other_mask])) if other_mask.any() else 0.0
                sim_margin_thr = self._cfg("min_sim_margin", 0.05)
                gates.append(GateResult("sim_margin", sim - runner_sim >= sim_margin_thr,
                                        sim - runner_sim, sim_margin_thr))

        # Embedding stats
        emb_arr = np.asarray(embedding, dtype=np.float32).ravel()
        emb_norm = float(np.linalg.norm(emb_arr)) if emb_arr.size else 0.0
        top_idx_arr = np.argsort(np.abs(emb_arr))[::-1][:10]
        emb_top = [(int(i), float(emb_arr[i])) for i in top_idx_arr]

        # Contribution path to the winning output (green overlay in the NN graph)
        decision_path = None
        if state.mode == "ensemble" and state.members and layer_activations:
            member = state.members[0]
            class_names = [
                state.person_names.get(int(c), str(c)) for c in member.classes_
            ]
            decision_path = compute_decision_path(
                member, vec, layer_activations, emb_top, output_probs, class_names
            )

        pred = self.predict(embedding)
        info = DeepDebugInfo(
            face_id=face_id,
            crop_path=crop_path,
            embedding_norm=emb_norm,
            embedding_top_dims=emb_top,
            all_similarities=all_similarities,
            layer_activations=layer_activations,
            output_probs=output_probs,
            gates=gates,
            prediction=pred,
            mode=state.mode,
            decision_path=decision_path,
        )
        return pred, info

    def _predict_prototype(self, vec: np.ndarray, sims: np.ndarray) -> DeepPrediction:
        state = self._state
        best: Dict[int, float] = {}
        for label in state.classes:
            mask = state.train_labels == label
            best[label] = float(np.max(sims[mask])) if mask.any() else 0.0
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        person_id, similarity = ranked[0]
        runner_sim = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = similarity - runner_sim
        # The configured minimum is a hard floor even when a stale persisted
        # model carries a lower calibrated value.
        default_floor = self._cfg("min_prototype_similarity", 0.55)
        floor = max(state.sim_floors.get(person_id, default_floor), default_floor)
        # When the nearest match is a manually confirmed face, the sim_floor gate
        # is bypassed: a human labelled this face as ground truth, so we trust it.
        has_manual_anchor = self._has_manual_anchor(sims, person_id)
        if similarity < floor and not has_manual_anchor:
            return DeepPrediction(
                person_id=None, person_name=None, score=similarity,
                probability=0.0, similarity=similarity, margin=margin,
                reason="prototype",
            )
        if len(ranked) > 1 and margin < self._cfg("min_sim_margin", 0.05):
            return DeepPrediction(
                person_id=None, person_name=None, score=similarity,
                probability=0.0, similarity=similarity, margin=margin,
                reason="margin",
            )
        return DeepPrediction(
            person_id=person_id,
            person_name=state.person_names.get(person_id),
            score=similarity,
            probability=1.0,
            similarity=similarity,
            margin=margin,
            reason="assigned",
        )

    def _predict_ensemble(self, vec: np.ndarray, sims: np.ndarray) -> DeepPrediction:
        state = self._state
        probs = self._ensemble_proba(vec.reshape(1, -1))[0]
        order = np.argsort(probs)[::-1]
        top_idx, second_idx = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
        person_id = int(state.classes[top_idx])
        probability = float(probs[top_idx])
        margin = probability - float(probs[second_idx]) if len(order) > 1 else probability

        mask = state.train_labels == person_id
        similarity = float(np.max(sims[mask])) if mask.any() else 0.0
        score = 0.5 * probability + 0.5 * similarity
        name = state.person_names.get(person_id)

        threshold = state.prob_thresholds.get(
            person_id, self._cfg("base_prob_threshold", 0.60)
        )
        if probability < threshold:
            return DeepPrediction(
                person_id=None, person_name=name, score=score,
                probability=probability, similarity=similarity, margin=margin,
                reason="below_threshold",
            )
        if margin < self._cfg("min_margin", 0.10):
            return DeepPrediction(
                person_id=None, person_name=name, score=score,
                probability=probability, similarity=similarity, margin=margin,
                reason="margin",
            )
        # The configured minimum is a hard floor even when a stale persisted
        # model carries a lower calibrated value.
        default_floor = self._cfg("min_prototype_similarity", 0.55)
        floor = max(state.sim_floors.get(person_id, default_floor), default_floor)
        # When the nearest match is a manually confirmed face, the sim_floor gate
        # is bypassed: a human labelled this face as ground truth, so we trust it.
        has_manual_anchor = self._has_manual_anchor(sims, person_id)
        if similarity < floor and not has_manual_anchor:
            return DeepPrediction(
                person_id=None, person_name=name, score=score,
                probability=probability, similarity=similarity, margin=margin,
                reason="prototype",
            )
        # MLP probabilities are blind to "nobody we know" — require the winner
        # to also beat every other person on raw embedding similarity.
        other_mask = state.train_labels != person_id
        runner_sim = float(np.max(sims[other_mask])) if other_mask.any() else 0.0
        if other_mask.any() and similarity - runner_sim < self._cfg("min_sim_margin", 0.05):
            return DeepPrediction(
                person_id=None, person_name=name, score=score,
                probability=probability, similarity=similarity, margin=margin,
                reason="margin",
            )
        return DeepPrediction(
            person_id=person_id, person_name=name, score=score,
            probability=probability, similarity=similarity, margin=margin,
            reason="assigned",
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": _MODEL_FILE_VERSION, "state": self._state}
        with open(path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("Deep model saved: %s", path)

    def load(self, path: Path | str) -> bool:
        """Load a previously saved model; returns False when unusable."""
        path = Path(path)
        if not path.exists():
            return False
        try:
            with open(path, "rb") as fh:
                payload = pickle.load(fh)
            if payload.get("version") != _MODEL_FILE_VERSION:
                log.warning("Deep model version mismatch — retraining required")
                return False
            self._state = payload["state"]
            return self.is_trained
        except Exception as exc:  # noqa: BLE001 — a corrupt model just retrains
            log.warning("Failed to load deep model %s: %s", path, exc)
            return False

    # ------------------------------------------------------------------
    # Training internals
    # ------------------------------------------------------------------

    def _ensemble_size(self) -> int:
        return max(1, int(self._cfg("ensemble_size", 5)))

    def _n_folds(self, dataset: TrainingDataset) -> int:
        counts = dataset.examples_per_person()
        min_count = min(counts.values()) if counts else 0
        return max(0, min(int(self._cfg("calibration_folds", 3)), min_count))

    def _make_mlp(self, seed: int, max_iter: Optional[int] = None):
        from sklearn.neural_network import MLPClassifier

        hidden = tuple(self._cfg("hidden_layers", (256, 128)))
        return MLPClassifier(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=int(max_iter or self._cfg("max_iter", 600)),
            shuffle=True,
            random_state=seed,
            early_stopping=False,
            n_iter_no_change=25,
            tol=1e-5,
        )

    def _fit_ensemble(self, x: np.ndarray, y: np.ndarray, tick) -> List[object]:
        """Train all ensemble members in parallel — saturates the CPU on purpose."""
        import os

        members: List[Optional[object]] = [None] * self._ensemble_size()

        def fit_one(idx: int):
            model = self._make_mlp(seed=1000 + idx)
            model.fit(x, y)
            tick(f"network {idx + 1}/{len(members)}")
            return idx, model

        workers = max(1, min(len(members), os.cpu_count() or 1))
        from app.tasks.resource_governor import get_resource_governor

        workers = get_resource_governor().recommended_workers(workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for idx, model in pool.map(fit_one, range(len(members))):
                members[idx] = model
        return [m for m in members if m is not None]

    def _ensemble_proba(self, x: np.ndarray) -> np.ndarray:
        state = self._state
        classes = np.asarray(state.classes)
        total = np.zeros((x.shape[0], len(classes)), dtype=np.float64)
        for model in state.members:
            proba = model.predict_proba(x)
            # Map the member's class order onto the canonical class order.
            col_of = {int(c): i for i, c in enumerate(model.classes_)}
            for j, cls in enumerate(classes):
                col = col_of.get(int(cls))
                if col is not None:
                    total[:, j] += proba[:, col]
        total /= max(1, len(state.members))
        return total

    def _augment(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Grow small classes with jittered / interpolated synthetic embeddings."""
        rng = np.random.default_rng(42)
        min_size = max(2, int(self._cfg("min_class_size", 8)))
        sigma = float(self._cfg("augment_noise_sigma", 0.03))

        extra_x: List[np.ndarray] = []
        extra_y: List[int] = []
        for label in sorted(set(y.tolist())):
            rows = x[y == label]
            need = min_size - rows.shape[0]
            for k in range(max(0, need)):
                if rows.shape[0] >= 2 and k % 2 == 1:
                    # Convex combination of two real examples of the person.
                    i, j = rng.choice(rows.shape[0], size=2, replace=False)
                    w = rng.uniform(0.3, 0.7)
                    sample = w * rows[i] + (1.0 - w) * rows[j]
                else:
                    sample = rows[k % rows.shape[0]] + rng.normal(
                        0.0, sigma, size=rows.shape[1]
                    )
                norm = float(np.linalg.norm(sample))
                if norm > 1e-8:
                    extra_x.append((sample / norm).astype(np.float32))
                    extra_y.append(label)

        if not extra_x:
            return x, y, 0
        x_aug = np.vstack([x, np.vstack(extra_x)])
        y_aug = np.concatenate([y, np.asarray(extra_y, dtype=y.dtype)])
        return x_aug, y_aug, len(extra_x)

    def _calibrate_prob_thresholds(
        self, dataset: TrainingDataset, tick
    ) -> Tuple[Dict[int, float], Optional[float]]:
        """Cross-validate to find a safe per-person probability threshold.

        Each person's threshold sits below the probabilities the model assigns
        to that person's *held-out* genuine faces, so real matches pass while
        imposters (which score lower) are rejected.  People with too few faces
        keep the global base threshold.
        """
        base = float(self._cfg("base_prob_threshold", 0.60))
        floor = float(self._cfg("min_prob_threshold", 0.35))
        n_folds = self._n_folds(dataset)
        if n_folds < 2:
            return {}, None

        from sklearn.model_selection import StratifiedKFold

        x, y = dataset.embeddings, dataset.labels
        true_probs: Dict[int, List[float]] = {}
        n_correct = 0
        n_total = 0

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=7)
        folds = list(skf.split(x, y))

        def run_fold(fold_idx: int):
            train_idx, test_idx = folds[fold_idx]
            x_aug, y_aug, _ = self._augment(x[train_idx], y[train_idx])
            model = self._make_mlp(seed=2000 + fold_idx)
            model.fit(x_aug, y_aug)
            proba = model.predict_proba(x[test_idx])
            tick(f"calibration {fold_idx + 1}/{n_folds}")
            return test_idx, model.classes_, proba

        import os

        workers = max(1, min(n_folds, os.cpu_count() or 1))
        from app.tasks.resource_governor import get_resource_governor

        workers = get_resource_governor().recommended_workers(workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(run_fold, range(n_folds)))

        for test_idx, classes_, proba in results:
            col_of = {int(c): i for i, c in enumerate(classes_)}
            for row, sample_idx in enumerate(test_idx):
                label = int(y[sample_idx])
                col = col_of.get(label)
                if col is None:
                    continue
                p_true = float(proba[row, col])
                true_probs.setdefault(label, []).append(p_true)
                n_total += 1
                if int(classes_[int(np.argmax(proba[row]))]) == label:
                    n_correct += 1

        thresholds: Dict[int, float] = {}
        for label, probs in true_probs.items():
            if len(probs) < 2:
                continue
            # 20th percentile of genuine probabilities, with headroom, clamped
            # into [floor, base] — never stricter than the global base.
            q = float(np.percentile(probs, 20))
            thresholds[label] = float(np.clip(q * 0.8, floor, base))

        accuracy = (n_correct / n_total) if n_total else None
        return thresholds, accuracy

    def _calibrate_sim_floors(self, dataset: TrainingDataset) -> Dict[int, float]:
        """Derive per-person cosine-similarity floors from intra-class spread."""
        default_floor = float(self._cfg("min_prototype_similarity", 0.55))
        floors: Dict[int, float] = {}
        x, y = dataset.embeddings, dataset.labels
        for label in sorted(set(y.tolist())):
            rows = x[y == label]
            if rows.shape[0] < 2:
                continue
            sims = rows @ rows.T
            iu = np.triu_indices(rows.shape[0], k=1)
            pair_sims = sims[iu]
            if pair_sims.size == 0:
                continue
            # A new face of this person should look at least as similar as the
            # person's own weaker pairs (with slack).  Calibration may only
            # TIGHTEN the gate: a person whose labeled faces are noisy/diverse
            # must not drag the floor below the configured minimum, or every
            # vaguely face-like crop ends up assigned to them.  The upper cap
            # keeps burst-photo training sets (near-duplicate crops, intra-pair
            # sims ≈ 1.0) from rejecting every genuinely new photo.
            q = float(np.percentile(pair_sims, 10)) - 0.08
            floors[label] = float(np.clip(q, default_floor, 0.80))
        return floors

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _has_manual_anchor(self, sims: np.ndarray, person_id: int) -> bool:
        """True when the best-matching training example for *person_id* is a
        manually confirmed face and its similarity meets the anchor threshold.

        Old persisted models lack ``train_manual_mask`` (the field was added
        later) — they return False, preserving the original gate behaviour.
        """
        state = self._state
        manual_mask = getattr(state, "train_manual_mask", None)
        if manual_mask is None or not manual_mask.any():
            return False
        person_manual = manual_mask & (state.train_labels == person_id)
        if not person_manual.any():
            return False
        best_manual_sim = float(np.max(sims[person_manual]))
        anchor_thr = self._cfg("manual_anchor_min_similarity", 0.48)
        return best_manual_sim >= anchor_thr

    def _cfg(self, name: str, default):
        return getattr(self._config, name, default) if self._config is not None else default

    @staticmethod
    def _unit(vector: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if vector is None:
            return None
        vec = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-8:
            return None
        return vec / norm
