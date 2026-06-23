"""Deep-learning recognition service (the "new" recognition path).

Orchestrates the :class:`~app.deep.classifier.DeepFaceClassifier` against the
database:

* :meth:`train` — collects every trusted labeled face, retrains the neural
  ensemble (continual learning) and records a :class:`TrainingRun`;
* :meth:`recognize` — classifies the still-unknown faces and assigns the
  confident matches, logging every decision as an :class:`AutoAssignment`
  for later review;
* :meth:`confirm_assignment` / :meth:`correct_assignment` /
  :meth:`revert_assignment` — the review actions of the "automatic
  groupings" tab.  Confirmations become trusted training data; corrections
  additionally record :class:`FaceCorrection` constraints so the same
  mistake is not repeated.

Safety rules
------------
* A face already assigned to a *named* person is never touched — an earlier
  recognition always wins over a newer one.
* Faces failing the open-set gates (strangers, non-faces) stay unknown.
* The user's "different person" judgements are hard vetoes.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.config import DeepRecognitionConfig
from app.db.models import (
    AUTO_ASSIGN_STATUS_AUTO,
    AUTO_ASSIGN_STATUS_CONFIRMED,
    AUTO_ASSIGN_STATUS_CORRECTED,
    AUTO_ASSIGN_STATUS_REVERTED,
    AutoAssignment,
    Face,
    FaceCorrection,
    Person,
    TrainingRun,
)
from app.deep.classifier import DeepFaceClassifier
from app.deep.dataset import build_training_dataset

log = logging.getLogger(__name__)

_MODEL_FILENAME = "deep_face_model.pkl"

# Assignment source written by this engine.
DEEP_ASSIGNMENT_SOURCE = "deep_recognition"
# Source after the user confirmed an automatic assignment (trusted training).
DEEP_CONFIRMED_SOURCE = "deep_confirmed"


def _utcnow_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


@dataclass
class DeepTrainStats:
    """Counters returned by :meth:`DeepRecognitionService.train`."""

    n_persons: int = 0
    n_examples: int = 0
    n_augmented: int = 0
    validation_accuracy: Optional[float] = None
    duration_seconds: float = 0.0
    reused_existing_model: bool = False


@dataclass
class DeepRecognitionStats:
    """Counters returned by :meth:`DeepRecognitionService.recognize`."""

    n_candidates: int = 0
    n_assigned: int = 0
    n_rejected_outlier: int = 0       # stranger / non-face — stayed unknown
    n_rejected_threshold: int = 0
    n_rejected_margin: int = 0
    n_rejected_prototype: int = 0
    n_rejected_correction: int = 0    # vetoed by a user "different" judgement
    n_skipped_low_confidence: int = 0  # detector confidence too low to trust
    n_no_embedding: int = 0
    # Predicted a person that no longer exists in the DB — a stale model trained
    # on a previous database. Skipped instead of creating a ghost assignment.
    n_skipped_unknown_person: int = 0


@dataclass
class AutoAssignmentDTO:
    """Display payload for one reviewable automatic assignment."""

    assignment_id: int
    face_id: int
    person_id: int
    person_name: str
    previous_person_name: Optional[str]
    score: float
    status: str
    crop_path: Optional[str]
    image_path: Optional[str]
    bbox: Optional[Tuple[int, int, int, int]]
    run_id: int = 0
    run_finished_at: Optional[datetime] = None
    corrected_person_name: Optional[str] = None
    decided_at: Optional[datetime] = None
    # Parsed deep-engine decision graph (None for rows predating capture).
    decision: Optional[Dict] = None


def _parse_decision_json(raw: Optional[str]) -> Optional[Dict]:
    """Parse a stored decision JSON, tolerating missing/corrupt values."""
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


@dataclass
class TrainAndRecognizeResult:
    """Combined result of one full train + recognize cycle."""

    training_run_id: Optional[int] = None
    train: DeepTrainStats = field(default_factory=DeepTrainStats)
    recognition: DeepRecognitionStats = field(default_factory=DeepRecognitionStats)


class DeepRecognitionService:
    """DB-facing orchestration of the deep recognition engine."""

    def __init__(
        self,
        session: Session,
        config: Optional[DeepRecognitionConfig] = None,
        model_dir: Optional[Path | str] = None,
    ) -> None:
        self._session = session
        self._config = config or DeepRecognitionConfig()
        self._model_dir = Path(model_dir) if model_dir else Path(self._config.model_dir)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @property
    def model_path(self) -> Path:
        return self._model_dir / _MODEL_FILENAME

    def delete_model(self) -> bool:
        """Delete the persisted model file so the next train starts from scratch.

        Returns True when a file was removed.  Used by the "rebuild from scratch"
        flow: without this, an empty/unlabeled DB would leave the previous
        model on disk, and any disk-loading path would still trust its stale,
        now-deleted people.  Never raises.
        """
        path = self.model_path
        try:
            if path.exists():
                path.unlink()
                log.info("Deep model deleted (rebuild from scratch): %s", path)
                return True
        except OSError as exc:
            log.warning("Could not delete deep model %s: %s", path, exc)
        return False

    def decision_for_face(self, face_id: int) -> Optional[Dict]:
        """Recompute the deep decision graph for *face_id* from the saved model.

        Used by the review tab to show a graph for assignments made *before*
        the per-row decision was persisted (their ``decision_json`` is NULL).
        Returns None — never raises — when there is no trained model, no face,
        or no embedding, so the caller can show a friendly "no graph" notice.
        """
        from app.deep.debug_info import decision_to_dict

        info = self.debug_info_for_face(face_id)
        if info is None:
            return None
        decision = decision_to_dict(info)
        decision["recomputed"] = True
        return decision

    def debug_info_for_face(self, face_id: int):
        """Run a debug forward pass for *face_id* on the saved model.

        Returns the full :class:`~app.deep.debug_info.DeepDebugInfo` (layer
        activations, decision path, gates) or None — never raises — when there
        is no trained model, no face, or no embedding.
        """
        from app.deep.classifier import DeepFaceClassifier

        try:
            face = self._session.get(Face, face_id)
            if face is None:
                return None
            embedding = face.get_embedding()
            if embedding is None or not self.model_path.exists():
                return None
            classifier = DeepFaceClassifier(self._config)
            if not classifier.load(self.model_path) or not classifier.is_trained:
                return None
            crop_path = str(face.crop_path) if face.crop_path else None
            _pred, info = classifier.predict_debug(
                embedding, face_id=face.id, crop_path=crop_path
            )
            return info
        except Exception:  # noqa: BLE001
            log.warning("Could not recompute debug info for face %s", face_id)
            return None

    def debug_sample_face(self):
        """Debug info for a representative embedded face on the *current* model.

        Lets the neural-net graph refresh to the live model — its current set of
        persons — without needing a full visualization scan.  Prefers a face
        assigned to a real (named, unprotected) person so the output layer shows
        a meaningful prediction; falls back to any embedded face.  Returns None
        (never raises) when there is no trained model or no embedded face.
        """
        if not self.model_path.exists():
            return None
        try:
            face = (
                self._session.query(Face)
                .join(Person, Face.person_id == Person.id)
                .filter(Person.is_auto_named == False)  # noqa: E712
                .filter(Person.is_protected == False)  # noqa: E712
                .filter(Face.embedding_exists())
                .order_by(Face.assignment_confidence.desc())
                .first()
            )
            if face is None:
                face = (
                    self._session.query(Face)
                    .filter(Face.embedding_exists())
                    .first()
                )
        except Exception:  # noqa: BLE001
            log.warning("debug_sample_face: face lookup failed", exc_info=True)
            return None
        if face is None:
            return None
        return self.debug_info_for_face(face.id)

    # ------------------------------------------------------------------
    # Model export / import
    # ------------------------------------------------------------------

    def export_model(self, dest_path: Path | str) -> bool:
        """Copy the current model file to *dest_path*.

        Returns True on success, False when no trained model exists yet.
        """
        dest = Path(dest_path)
        src = self.model_path
        if not src.exists():
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        log.info("Deep model exported: %s → %s", src, dest)
        return True

    def import_model(self, src_path: Path | str) -> bool:
        """Replace the active model with the file at *src_path*.

        Validates the file version before installing.  Returns True on success,
        False when the file is missing, corrupt, or a version mismatch.
        """
        import pickle

        from app.deep.classifier import _MODEL_FILE_VERSION

        src = Path(src_path)
        if not src.exists():
            log.warning("Import failed — file not found: %s", src)
            return False
        try:
            with open(src, "rb") as fh:
                payload = pickle.load(fh)
            if payload.get("version") != _MODEL_FILE_VERSION:
                log.warning(
                    "Import failed — version mismatch (got %s, expected %s)",
                    payload.get("version"),
                    _MODEL_FILE_VERSION,
                )
                return False
        except Exception as exc:  # noqa: BLE001
            log.warning("Import failed — cannot read model file: %s", exc)
            return False

        dest = self.model_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        log.info("Deep model imported: %s → %s", src, dest)
        return True

    def train(
        self,
        mode: str = "rescan",
        progress_cb=None,
        force: bool = False,
    ) -> Tuple[TrainingRun, DeepTrainStats, DeepFaceClassifier]:
        """Retrain the neural ensemble on every trusted labeled face.

        Returns the persisted :class:`TrainingRun`, the statistics and the
        ready-to-use classifier.  When the labeled data is unchanged since the
        last run and ``skip_unchanged`` is on, the saved model is reused —
        unless ``force`` is set (the explicit "train the model" action must
        always retrain, e.g. to pick up new engine defaults).
        """
        dataset = build_training_dataset(
            self._session,
            auto_min_confidence=self._config.auto_training_min_confidence,
            use_auto_assignments=self._config.use_auto_assignments_for_training,
            exclude_low_quality=self._config.strict_quality_filter,
        )

        run = TrainingRun(mode=mode, started_at=_utcnow_naive())
        self._session.add(run)

        classifier = DeepFaceClassifier(self._config)
        stats = DeepTrainStats(
            n_persons=dataset.n_persons, n_examples=dataset.n_examples
        )

        fingerprint = dataset.fingerprint()
        if (
            not force
            and self._config.skip_unchanged
            and classifier.load(self.model_path)
            and classifier.fingerprint == fingerprint
            and self._architecture_unchanged(classifier)
            and self._model_persons_exist(classifier)
        ):
            stats.reused_existing_model = True
            report = classifier.report
            if report is not None:
                stats.n_augmented = report.n_augmented
                stats.validation_accuracy = report.validation_accuracy
            log.info("Deep training skipped: labeled data unchanged — model reused")
        else:
            report = classifier.train(dataset, progress_cb=progress_cb)
            stats.n_augmented = report.n_augmented
            stats.validation_accuracy = report.validation_accuracy
            stats.duration_seconds = report.duration_seconds
            if classifier.is_trained:
                try:
                    classifier.save(self.model_path)
                except OSError as exc:
                    log.warning("Could not persist deep model: %s", exc)

        run.finished_at = _utcnow_naive()
        run.n_persons = stats.n_persons
        run.n_examples = stats.n_examples
        run.n_augmented = stats.n_augmented
        run.validation_accuracy = stats.validation_accuracy
        run.data_fingerprint = fingerprint
        run.model_path = str(self.model_path)
        self._session.commit()
        return run, stats, classifier

    def _model_persons_exist(self, classifier: DeepFaceClassifier) -> bool:
        """False when none of the model's people still exist in the database.

        Prevents reusing a stale on-disk model (trained on a previous DB) after
        the database was reset — such a model must be retrained from scratch.
        """
        ids = set(classifier.person_ids)
        if not ids:
            return True  # empty model carries no stale people
        existing = {pid for (pid,) in self._session.query(Person.id).all()}
        return bool(ids & existing)

    def _architecture_unchanged(self, classifier: DeepFaceClassifier) -> bool:
        """False when the saved ensemble was trained with a different layer
        layout than the current config — the model must then be rebuilt even
        though the labeled data is unchanged."""
        trained = classifier.trained_hidden_layers
        return trained is None or trained == tuple(self._config.hidden_layers)

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------

    def recognize(
        self,
        classifier: DeepFaceClassifier,
        training_run: TrainingRun,
        progress_cb=None,
        debug_cb=None,
    ) -> DeepRecognitionStats:
        """Assign still-unknown faces to known people where the model is sure.

        Faces already assigned to a named person are never candidates — an
        existing recognition always survives a newer run untouched.
        """
        stats = DeepRecognitionStats()
        if not classifier.is_trained:
            log.info("Deep recognition skipped: no trained model")
            return stats

        # Guard against a stale model: one trained on a previous database whose
        # people no longer exist (e.g. the DB was reset but the model kept).
        # Assigning to those vanished person IDs would create ghost links.
        valid_person_ids = {
            pid for (pid,) in self._session.query(Person.id).all()
        }
        model_person_ids = set(classifier.person_ids)
        if model_person_ids and not (model_person_ids & valid_person_ids):
            log.warning(
                "Deep recognition skipped: the model was trained on people that "
                "no longer exist in this database (model persons=%s). Rebuild the "
                "AI model from scratch.",
                sorted(model_person_ids),
            )
            return stats

        candidates = self._load_candidate_faces()
        stats.n_candidates = len(candidates)
        if not candidates:
            return stats

        vetoes = self._load_correction_vetoes(candidates)
        now = _utcnow_naive()

        for idx, face in enumerate(candidates):
            if progress_cb is not None:
                progress_cb(idx + 1, len(candidates), f"face #{face.id}")

            # "Never recognise non-faces": a weak detection may not even be a
            # face — it may stay in the unknown pool but is never auto-assigned.
            if (
                face.confidence is not None
                and face.confidence < self._config.min_face_confidence
            ):
                stats.n_skipped_low_confidence += 1
                continue

            debug_info = None
            if debug_cb is not None:
                crop_path = str(face.crop_path) if face.crop_path else None
                prediction, debug_info = classifier.predict_debug(
                    face.get_embedding(), face_id=face.id, crop_path=crop_path
                )
                debug_cb(debug_info)
            else:
                prediction = classifier.predict(face.get_embedding())
            if prediction.reason == "no_embedding":
                stats.n_no_embedding += 1
                continue
            if prediction.person_id is None:
                if prediction.reason == "outlier":
                    stats.n_rejected_outlier += 1
                elif prediction.reason == "below_threshold":
                    stats.n_rejected_threshold += 1
                elif prediction.reason == "margin":
                    stats.n_rejected_margin += 1
                elif prediction.reason == "prototype":
                    stats.n_rejected_prototype += 1
                continue
            if prediction.person_id in vetoes.get(face.id, set()):
                stats.n_rejected_correction += 1
                continue
            # Never attach a face to a person that no longer exists (stale model).
            if prediction.person_id not in valid_person_ids:
                stats.n_skipped_unknown_person += 1
                continue

            previous_person_id = face.person_id
            previous_person_name = (
                face.person.name if face.person is not None else None
            )
            face.person_id = prediction.person_id
            face.assignment_source = DEEP_ASSIGNMENT_SOURCE
            face.assignment_confidence = prediction.score
            face.assigned_at = now

            # Capture the decision graph for this assignment so the review tab
            # can explain it later.  Reuse the debug info already computed for
            # the live viz; otherwise compute it once for this assigned face
            # (rejected faces stay on the cheap predict() path).
            decision_json = self._serialize_decision(
                classifier, face, debug_info
            )

            self._session.add(
                AutoAssignment(
                    training_run=training_run,
                    face_id=face.id,
                    person_id=prediction.person_id,
                    previous_person_id=previous_person_id,
                    previous_person_name=previous_person_name,
                    score=prediction.score,
                    status=AUTO_ASSIGN_STATUS_AUTO,
                    decision_json=decision_json,
                    created_at=now,
                )
            )
            stats.n_assigned += 1

        self._delete_orphan_auto_persons()
        self._session.commit()
        log.info(
            "Deep recognition: %d/%d assigned (outlier=%d, threshold=%d, margin=%d, "
            "prototype=%d, correction-veto=%d, low-conf=%d, no-emb=%d, "
            "unknown-person=%d)",
            stats.n_assigned, stats.n_candidates,
            stats.n_rejected_outlier, stats.n_rejected_threshold,
            stats.n_rejected_margin, stats.n_rejected_prototype,
            stats.n_rejected_correction, stats.n_skipped_low_confidence,
            stats.n_no_embedding, stats.n_skipped_unknown_person,
        )
        return stats

    @staticmethod
    def _serialize_decision(
        classifier: DeepFaceClassifier, face, debug_info
    ) -> Optional[str]:
        """JSON for *face*'s assignment decision graph, or None on failure.

        Reuses *debug_info* when it was already computed for the live viz;
        otherwise runs ``predict_debug`` once for this single assigned face.
        Never raises — a diagnostics blob must not abort a recognition run.
        """
        from app.deep.debug_info import DeepDebugInfo, decision_to_dict

        try:
            if not isinstance(debug_info, DeepDebugInfo):
                crop_path = str(face.crop_path) if face.crop_path else None
                _pred, debug_info = classifier.predict_debug(
                    face.get_embedding(), face_id=face.id, crop_path=crop_path
                )
            return json.dumps(decision_to_dict(debug_info), ensure_ascii=False)
        except Exception:  # noqa: BLE001
            log.warning("Could not serialize decision graph for face %s", face.id)
            return None

    def train_and_recognize(
        self,
        mode: str = "rescan",
        train_progress_cb=None,
        recognize_progress_cb=None,
        debug_cb=None,
    ) -> TrainAndRecognizeResult:
        """Convenience wrapper: one full continual-learning cycle.

        A *rebuild* always retrains from scratch (``force``), never reusing the
        previous on-disk model — the rebuild flow has already deleted it.
        """
        result = TrainAndRecognizeResult()
        run, train_stats, classifier = self.train(
            mode=mode,
            progress_cb=train_progress_cb,
            force=(mode in ("rebuild", "rebuild_model")),
        )
        result.training_run_id = run.id
        result.train = train_stats
        result.recognition = self.recognize(
            classifier, run,
            progress_cb=recognize_progress_cb,
            debug_cb=debug_cb,
        )
        return result

    # ------------------------------------------------------------------
    # Review actions ("automatic groupings" tab)
    # ------------------------------------------------------------------

    def latest_run(self) -> Optional[TrainingRun]:
        return (
            self._session.query(TrainingRun)
            .filter(TrainingRun.finished_at.isnot(None))
            .order_by(TrainingRun.id.desc())
            .first()
        )

    def latest_assignment_run_id(self) -> Optional[int]:
        """Id of the newest run that actually produced auto-assignments.

        Train-only runs create a :class:`TrainingRun` without assignments;
        the review tab must keep showing the last run that has any.
        """
        row = (
            self._session.query(AutoAssignment.training_run_id)
            .order_by(AutoAssignment.training_run_id.desc())
            .first()
        )
        return row[0] if row is not None else None

    def list_auto_assignments(
        self,
        run_id: Optional[int] = None,
        only_open: bool = True,
    ) -> List[AutoAssignmentDTO]:
        """Reviewable assignments of *run_id* (default: last run with any)."""
        if run_id is None:
            run_id = self.latest_assignment_run_id()
            if run_id is None:
                return []

        query = (
            self._session.query(AutoAssignment)
            .filter(AutoAssignment.training_run_id == run_id)
        )
        if only_open:
            query = query.filter(AutoAssignment.status == AUTO_ASSIGN_STATUS_AUTO)

        dtos: List[AutoAssignmentDTO] = []
        for assignment in query.order_by(AutoAssignment.score.desc()).all():
            face = assignment.face
            if face is None:
                continue
            # Skip stale rows: the face was meanwhile moved elsewhere manually.
            if (
                assignment.status == AUTO_ASSIGN_STATUS_AUTO
                and face.person_id != assignment.person_id
            ):
                continue
            person = assignment.person
            dtos.append(
                AutoAssignmentDTO(
                    assignment_id=assignment.id,
                    face_id=face.id,
                    person_id=assignment.person_id,
                    person_name=person.name if person else "?",
                    previous_person_name=(
                        assignment.previous_person_name
                        or (
                            assignment.previous_person.name
                            if assignment.previous_person is not None
                            else None
                        )
                    ),
                    score=assignment.score,
                    status=assignment.status,
                    crop_path=face.crop_path,
                    image_path=face.image.file_path if face.image else None,
                    bbox=(face.bbox_x, face.bbox_y, face.bbox_w, face.bbox_h),
                    run_id=run_id,
                    run_finished_at=(
                        assignment.training_run.finished_at
                        if assignment.training_run
                        else None
                    ),
                    decision=_parse_decision_json(assignment.decision_json),
                )
            )
        return dtos

    def list_decided_assignments(self, limit: int = 100) -> List[AutoAssignmentDTO]:
        """Already-reviewed assignments (any run), newest decision first."""
        rows = (
            self._session.query(AutoAssignment)
            .filter(AutoAssignment.status != AUTO_ASSIGN_STATUS_AUTO)
            .order_by(AutoAssignment.decided_at.desc(), AutoAssignment.id.desc())
            .limit(limit)
            .all()
        )
        dtos: List[AutoAssignmentDTO] = []
        for assignment in rows:
            face = assignment.face
            person = assignment.person
            dtos.append(
                AutoAssignmentDTO(
                    assignment_id=assignment.id,
                    face_id=assignment.face_id,
                    person_id=assignment.person_id,
                    person_name=person.name if person else "?",
                    previous_person_name=(
                        assignment.previous_person_name
                        or (
                            assignment.previous_person.name
                            if assignment.previous_person is not None
                            else None
                        )
                    ),
                    score=assignment.score,
                    status=assignment.status,
                    crop_path=face.crop_path if face is not None else None,
                    image_path=(
                        face.image.file_path
                        if face is not None and face.image
                        else None
                    ),
                    bbox=(
                        (face.bbox_x, face.bbox_y, face.bbox_w, face.bbox_h)
                        if face is not None
                        else None
                    ),
                    run_id=assignment.training_run_id,
                    run_finished_at=(
                        assignment.training_run.finished_at
                        if assignment.training_run
                        else None
                    ),
                    corrected_person_name=(
                        assignment.corrected_person.name
                        if assignment.corrected_person is not None
                        else None
                    ),
                    decided_at=assignment.decided_at,
                    decision=_parse_decision_json(assignment.decision_json),
                )
            )
        return dtos

    def count_open_assignments(self, run_id: Optional[int] = None) -> int:
        if run_id is None:
            run_id = self.latest_assignment_run_id()
            if run_id is None:
                return 0
        return (
            self._session.query(AutoAssignment)
            .filter(AutoAssignment.training_run_id == run_id)
            .filter(AutoAssignment.status == AUTO_ASSIGN_STATUS_AUTO)
            .count()
        )

    def confirm_assignment(self, assignment_id: int) -> AutoAssignment:
        """User says the grouping is right → becomes trusted training data."""
        assignment = self._require_assignment(assignment_id)
        face = assignment.face
        if face is not None and face.person_id == assignment.person_id:
            face.assignment_source = DEEP_CONFIRMED_SOURCE
            face.assignment_confidence = 1.0
        assignment.status = AUTO_ASSIGN_STATUS_CONFIRMED
        assignment.decided_at = _utcnow_naive()
        self._session.commit()
        return assignment

    def confirm_assignments(self, assignment_ids: List[int]) -> int:
        """Confirm many groupings in one transaction. Returns the count confirmed.

        Skips ids that no longer exist (e.g. removed by a meanwhile-running
        re-scan), so a stale selection can never abort the whole batch.
        """
        confirmed = 0
        for assignment_id in assignment_ids:
            assignment = self._session.get(AutoAssignment, assignment_id)
            if assignment is None:
                continue
            face = assignment.face
            if face is not None and face.person_id == assignment.person_id:
                face.assignment_source = DEEP_CONFIRMED_SOURCE
                face.assignment_confidence = 1.0
            assignment.status = AUTO_ASSIGN_STATUS_CONFIRMED
            assignment.decided_at = _utcnow_naive()
            confirmed += 1
        self._session.commit()
        log.info("Deep recognition: batch-confirmed %d assignments", confirmed)
        return confirmed

    def correct_assignment(
        self, assignment_id: int, new_person_id: int
    ) -> AutoAssignment:
        """User moves the face to the right person — and the engine learns.

        Records a "different person" judgement against the wrong person and a
        "same person" judgement towards the right one, so future runs are
        vetoed from repeating the mistake and gain a confirmed example.
        """
        assignment = self._require_assignment(assignment_id)
        if assignment.face is None:
            raise ValueError(f"Face for assignment {assignment_id} no longer exists")
        new_person = self._session.get(Person, new_person_id)
        if new_person is None:
            raise ValueError(f"Person id={new_person_id} not found")

        self._correct_one(assignment, new_person_id)
        self._delete_orphan_auto_persons()
        self._session.commit()
        return assignment

    def correct_assignment_to_new_person(
        self, assignment_id: int, new_person_name: str
    ) -> AutoAssignment:
        """Correction variant that first creates a brand-new named person."""
        name = new_person_name.strip()
        if not name:
            raise ValueError("New person name must not be empty")
        person = Person(name=name, is_auto_named=False)
        self._session.add(person)
        self._session.flush()
        return self.correct_assignment(assignment_id, person.id)

    def correct_assignments(
        self, assignment_ids: List[int], new_person_id: int
    ) -> int:
        """Move many groupings to *new_person_id* in one transaction.

        Skips ids whose row or face no longer exists. Returns the count moved.
        """
        new_person = self._session.get(Person, new_person_id)
        if new_person is None:
            raise ValueError(f"Person id={new_person_id} not found")
        moved = 0
        for assignment_id in assignment_ids:
            assignment = self._session.get(AutoAssignment, assignment_id)
            if assignment is None or assignment.face is None:
                continue
            self._correct_one(assignment, new_person_id)
            moved += 1
        self._delete_orphan_auto_persons()
        self._session.commit()
        log.info("Deep recognition: batch-corrected %d assignments", moved)
        return moved

    def correct_assignments_to_new_person(
        self, assignment_ids: List[int], new_person_name: str
    ) -> int:
        """Batch correction that first creates a brand-new named person."""
        name = new_person_name.strip()
        if not name:
            raise ValueError("New person name must not be empty")
        person = Person(name=name, is_auto_named=False)
        self._session.add(person)
        self._session.flush()
        return self.correct_assignments(assignment_ids, person.id)

    def _correct_one(self, assignment: AutoAssignment, new_person_id: int) -> None:
        """Shared correction core; caller validates the person and commits.

        Records a "different person" judgement against the wrong person and a
        "same person" judgement towards the right one, so future runs are
        vetoed from repeating the mistake and gain a confirmed example.
        """
        face = assignment.face
        wrong_person_id = assignment.person_id
        face.person_id = new_person_id
        face.assignment_source = "manual"
        face.assignment_confidence = 1.0
        face.assigned_at = _utcnow_naive()

        self._record_correction_pair(face, wrong_person_id, same_person=False)
        self._record_correction_pair(face, new_person_id, same_person=True)

        assignment.status = AUTO_ASSIGN_STATUS_CORRECTED
        assignment.corrected_person_id = new_person_id
        assignment.decided_at = _utcnow_naive()

    def revert_assignment(self, assignment_id: int) -> AutoAssignment:
        """User sends the face back to where it was before the run."""
        assignment = self._require_assignment(assignment_id)
        self._revert_one(assignment)
        self._session.commit()
        return assignment

    def revert_assignments(self, assignment_ids: List[int]) -> int:
        """Reject (undo) many groupings in one transaction.

        Skips ids that no longer exist. Returns the count reverted.
        """
        reverted = 0
        for assignment_id in assignment_ids:
            assignment = self._session.get(AutoAssignment, assignment_id)
            if assignment is None:
                continue
            self._revert_one(assignment)
            reverted += 1
        self._session.commit()
        log.info("Deep recognition: batch-reverted %d assignments", reverted)
        return reverted

    def revert_all_open(self, run_id: Optional[int] = None) -> int:
        """Revert every still-open (auto) assignment of *run_id* in one go.

        Recovery hatch for a run gone wrong (e.g. an avalanche of faces onto
        one person): each face returns to where it was before the run and a
        "different person" judgement is recorded so the engine learns from
        the mass rejection too. Returns the number of reverted assignments.
        """
        if run_id is None:
            run_id = self.latest_assignment_run_id()
            if run_id is None:
                return 0
        open_assignments = (
            self._session.query(AutoAssignment)
            .filter(AutoAssignment.training_run_id == run_id)
            .filter(AutoAssignment.status == AUTO_ASSIGN_STATUS_AUTO)
            .all()
        )
        for assignment in open_assignments:
            self._revert_one(assignment)
        self._session.commit()
        log.info(
            "Deep recognition: bulk-reverted %d open assignments of run %s",
            len(open_assignments), run_id,
        )
        return len(open_assignments)

    def _revert_one(self, assignment: AutoAssignment) -> None:
        """Shared revert core; caller commits."""
        face = assignment.face
        if face is not None and face.person_id == assignment.person_id:
            previous = (
                self._session.get(Person, assignment.previous_person_id)
                if assignment.previous_person_id is not None
                else None
            )
            face.person_id = previous.id if previous is not None else None
            face.assignment_source = None
            face.assignment_confidence = None
            face.assigned_at = None
            self._record_correction_pair(
                face, assignment.person_id, same_person=False
            )
        assignment.status = AUTO_ASSIGN_STATUS_REVERTED
        assignment.decided_at = _utcnow_naive()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_assignment(self, assignment_id: int) -> AutoAssignment:
        assignment = self._session.get(AutoAssignment, assignment_id)
        if assignment is None:
            raise ValueError(f"AutoAssignment id={assignment_id} not found")
        return assignment

    def _record_correction_pair(
        self, face: Face, person_id: int, *, same_person: bool
    ) -> None:
        """Persist a FaceCorrection between *face* and a face of *person_id*."""
        other = (
            self._session.query(Face)
            .filter(Face.person_id == person_id)
            .filter(Face.id != face.id)
            .filter(Face.is_excluded == False)  # noqa: E712
            .order_by(Face.id)
            .first()
        )
        if other is None:
            return
        a, b = (face.id, other.id) if face.id < other.id else (other.id, face.id)
        existing = (
            self._session.query(FaceCorrection)
            .filter(FaceCorrection.face_id_a == a)
            .filter(FaceCorrection.face_id_b == b)
            .first()
        )
        if existing is not None:
            existing.same_person = same_person
            return
        self._session.add(
            FaceCorrection(face_id_a=a, face_id_b=b, same_person=same_person)
        )

    def _load_candidate_faces(self) -> List[Face]:
        """Embedded faces that are still unknown (or in an auto-named group)."""
        query = (
            self._session.query(Face)
            .outerjoin(Person, Face.person_id == Person.id)
            .filter(Face.embedding_exists())
            .filter(Face.is_excluded == False)  # noqa: E712
            .filter(
                or_(
                    Face.person_id.is_(None),
                    Person.is_auto_named == True,  # noqa: E712
                    Person.is_protected == True,  # noqa: E712
                )
            )
        )
        if self._config.strict_quality_filter:
            query = query.filter(
                or_(Face.is_low_quality.is_(None), Face.is_low_quality == False)  # noqa: E712
            )
        return query.all()

    def _load_correction_vetoes(
        self, candidates: List[Face]
    ) -> Dict[int, Set[int]]:
        """Map candidate face id → person ids vetoed by 'different' judgements."""
        candidate_ids = {face.id for face in candidates}
        if not candidate_ids:
            return {}

        corrections: List[FaceCorrection] = (
            self._session.query(FaceCorrection)
            .filter(FaceCorrection.same_person == False)  # noqa: E712
            .filter(
                or_(
                    FaceCorrection.face_id_a.in_(candidate_ids),
                    FaceCorrection.face_id_b.in_(candidate_ids),
                )
            )
            .all()
        )
        if not corrections:
            return {}

        other_ids: Set[int] = set()
        for corr in corrections:
            if corr.face_id_a in candidate_ids:
                other_ids.add(corr.face_id_b)
            if corr.face_id_b in candidate_ids:
                other_ids.add(corr.face_id_a)

        person_of: Dict[int, Optional[int]] = {
            face.id: face.person_id
            for face in self._session.query(Face)
            .filter(Face.id.in_(other_ids))
            .all()
        }

        vetoes: Dict[int, Set[int]] = {}
        for corr in corrections:
            if corr.face_id_a in candidate_ids:
                target = person_of.get(corr.face_id_b)
                if target is not None:
                    vetoes.setdefault(corr.face_id_a, set()).add(target)
            if corr.face_id_b in candidate_ids:
                target = person_of.get(corr.face_id_a)
                if target is not None:
                    vetoes.setdefault(corr.face_id_b, set()).add(target)
        return vetoes

    def _delete_orphan_auto_persons(self) -> int:
        orphans: List[Person] = (
            self._session.query(Person)
            .filter(Person.is_auto_named == True)  # noqa: E712
            .filter(~Person.faces.any())
            .all()
        )
        for person in orphans:
            self._session.delete(person)
        return len(orphans)
