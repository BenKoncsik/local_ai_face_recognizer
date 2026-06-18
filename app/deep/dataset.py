"""Training dataset construction for the deep recognition engine.

Collects every *trusted* labeled face from the database and turns it into a
matrix of L2-normalised embeddings plus person labels.  The trust rules mirror
the rest of the application:

* faces assigned to a real (non-auto-named, non-protected) person count;
* manual sources are always trusted — the user explicitly chose those faces;
* automatic assignments only count above a confidence floor, so one early
  mistake cannot poison the model;
* excluded faces and faces without embeddings never count.

The more faces a person has, the more training examples the network sees for
them — recognition quality grows with every confirmed face.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sqlalchemy.orm import Session

from app.db.models import Face, FaceCorrection, Person

log = logging.getLogger(__name__)

# Sources that mark a human decision (legacy rows have NULL source).
TRUSTED_MANUAL_SOURCES = {
    None,
    "manual",
    "manual_merge",
    "suggestion_approved",
    "deep_confirmed",
}

# Automatic sources that may be reused for training above a confidence floor.
AUTO_SOURCES = {"recognition", "deep_recognition", "clustering"}


@dataclass
class TrainingDataset:
    """Labeled embeddings ready for classifier training."""

    embeddings: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float32)
    )
    labels: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    face_ids: List[int] = field(default_factory=list)
    person_names: Dict[int, str] = field(default_factory=dict)
    # True for each training example that comes from a manual (human-confirmed) source.
    manual_flags: List[bool] = field(default_factory=list)
    # face_id pairs the user marked as "different person" / "same person"
    diff_pairs: Set[Tuple[int, int]] = field(default_factory=set)
    same_pairs: Set[Tuple[int, int]] = field(default_factory=set)

    @property
    def n_examples(self) -> int:
        return int(self.labels.shape[0])

    @property
    def n_persons(self) -> int:
        return len(set(self.labels.tolist())) if self.n_examples else 0

    def examples_per_person(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for label in self.labels.tolist():
            counts[label] = counts.get(label, 0) + 1
        return counts

    def fingerprint(self) -> str:
        """Stable hash of the labeled data — used to skip redundant retraining."""
        hasher = hashlib.sha256()
        order = np.argsort(np.asarray(self.face_ids))
        for idx in order:
            hasher.update(str(self.face_ids[idx]).encode())
            hasher.update(str(int(self.labels[idx])).encode())
        hasher.update(str(self.embeddings.shape).encode())
        return hasher.hexdigest()


def _normalise(vector: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if vector is None:
        return None
    vec = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return None
    return vec / norm


def is_trusted_training_face(
    face: Face,
    *,
    auto_min_confidence: float,
    use_auto_assignments: bool,
    exclude_low_quality: bool,
) -> bool:
    """Whether *face* may serve as a training example for its person."""
    if face.is_excluded or face.person is None:
        return False
    if face.person.is_auto_named or face.person.is_protected:
        return False
    if face.get_embedding() is None:
        return False

    if face.assignment_source in TRUSTED_MANUAL_SOURCES:
        # Human-confirmed faces bypass the quality filter on purpose.
        return True

    if not use_auto_assignments:
        return False
    if exclude_low_quality and face.is_low_quality:
        return False
    if face.assignment_source not in AUTO_SOURCES:
        return False
    confidence = face.assignment_confidence
    return confidence is not None and confidence >= auto_min_confidence


def build_training_dataset(
    session: Session,
    *,
    auto_min_confidence: float = 0.92,
    use_auto_assignments: bool = True,
    exclude_low_quality: bool = True,
) -> TrainingDataset:
    """Collect all trusted labeled faces into a :class:`TrainingDataset`."""
    persons: List[Person] = (
        session.query(Person)
        .filter(Person.is_auto_named == False)  # noqa: E712
        .filter(Person.is_protected == False)  # noqa: E712
        .all()
    )

    vectors: List[np.ndarray] = []
    labels: List[int] = []
    face_ids: List[int] = []
    manual_flags: List[bool] = []
    person_names: Dict[int, str] = {}

    for person in persons:
        for face in person.faces:
            if not is_trusted_training_face(
                face,
                auto_min_confidence=auto_min_confidence,
                use_auto_assignments=use_auto_assignments,
                exclude_low_quality=exclude_low_quality,
            ):
                continue
            vec = _normalise(face.get_embedding())
            if vec is None:
                continue
            vectors.append(vec)
            labels.append(person.id)
            face_ids.append(face.id)
            manual_flags.append(face.assignment_source in TRUSTED_MANUAL_SOURCES)
            person_names[person.id] = person.name

    dataset = TrainingDataset(
        embeddings=(
            np.vstack(vectors).astype(np.float32)
            if vectors
            else np.zeros((0, 0), dtype=np.float32)
        ),
        labels=np.asarray(labels, dtype=np.int64),
        face_ids=face_ids,
        manual_flags=manual_flags,
        person_names=person_names,
    )
    _attach_correction_pairs(session, dataset)

    log.info(
        "Deep training dataset: %d example(s) across %d person(s)",
        dataset.n_examples,
        dataset.n_persons,
    )
    return dataset


def _attach_correction_pairs(session: Session, dataset: TrainingDataset) -> None:
    """Load the user's same/not-same judgements as training constraints."""
    corrections: List[FaceCorrection] = session.query(FaceCorrection).all()
    for corr in corrections:
        pair = (
            (corr.face_id_a, corr.face_id_b)
            if corr.face_id_a < corr.face_id_b
            else (corr.face_id_b, corr.face_id_a)
        )
        if corr.same_person:
            dataset.same_pairs.add(pair)
        else:
            dataset.diff_pairs.add(pair)
