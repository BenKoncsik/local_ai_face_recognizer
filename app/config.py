"""Application configuration.

All tuneable parameters live here.  Load from a YAML file at startup;
fall back to sensible defaults when no file is present.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

from app import paths


@dataclass
class DetectionConfig:
    """Parameters for face detection."""

    # Minimum confidence score [0.0 – 1.0] to accept a detection
    confidence_threshold: float = 0.65

    # Minimum face size in pixels (width and height must both exceed this)
    min_face_size: int = 50

    # Path to the Edge TPU compiled face-detection model (.tflite).
    # Set to None to force CPU-only mode regardless of hardware.
    coral_model_path: Optional[str] = None

    # Path to the CPU TFLite / OpenCV DNN model used for fallback detection.
    # Default: OpenCV's bundled res10_300x300_ssd deploy.prototxt / caffemodel.
    cpu_model_path: Optional[str] = None

    # Input size expected by the CPU DNN model (width, height)
    cpu_model_input_size: tuple[int, int] = (300, 300)


@dataclass
class EmbeddingConfig:
    """Parameters for face embedding generation.

    NOTE: Embedding runs on CPU via a local TFLite model.
          Coral is NOT used for embeddings — only for detection.
    """

    # Path to MobileFaceNet or compatible embedding TFLite model.
    # Download instructions in README; set this to a local path.
    model_path: Optional[str] = None

    # Size to which face crops are resized before embedding
    input_size: tuple[int, int] = (112, 112)

    # Length of the embedding vector produced by the model.
    # MobileFaceNet: 192.  ArcFace variants: 512.
    embedding_dim: int = 192


@dataclass
class ClusteringConfig:
    """Parameters for DBSCAN face clustering."""

    # Maximum cosine distance between two faces in the same cluster.
    # Lower → stricter (more clusters).  Tune to your dataset.
    epsilon: float = 0.4

    # Minimum faces required to form a cluster core point.
    min_samples: int = 2

    # Distance metric passed to DBSCAN
    metric: str = "cosine"


@dataclass
class RecognitionConfig:
    """Parameters for learned person recognition.

    The recognizer builds person profiles from already labeled faces and
    assigns currently unassigned / auto-named faces to known people when the
    match is strong and unambiguous.
    """

    # Minimum combined cosine similarity required for automatic assignment.
    auto_assign_threshold: float = 0.72

    # Required gap between the best and second-best person match.
    # Higher values reduce ambiguous automatic assignments.
    min_margin: float = 0.08

    # Minimum trusted training faces before a person can be recognized.
    min_examples_per_person: int = 1

    # Blend between the person's centroid and the best individual example.
    # 1.0 = centroid only, 0.0 = nearest example only.
    centroid_weight: float = 0.70

    # Whether very confident automatic assignments can strengthen future
    # profiles. Manual / legacy assignments are always trusted.
    use_recognized_faces_for_training: bool = True

    # Minimum confidence for an automatic assignment to be reused as training.
    profile_auto_min_confidence: float = 0.85


@dataclass
class SuggestionConfig:
    """Parameters for the unknown-person name-suggestion feature.

    The feature compares automatically-named ("Unknown N") persons against
    the embedding profiles of manually-named persons and proposes likely
    identity matches.  No merge happens without explicit user approval.
    """

    # Minimum cosine similarity [0.0 – 1.0] for a match to be suggested.
    # Below this value nothing is proposed.  Higher → stricter.
    similarity_threshold: float = 0.5

    # Maximum number of ranked target candidates proposed per unknown person.
    max_suggestions_per_person: int = 3


@dataclass
class StorageConfig:
    """Paths for persistent data."""

    # Directory where face crop thumbnails are stored
    crops_dir: str = "data/crops"

    # SQLite database file
    db_path: str = "data/faces.db"


@dataclass
class ScanConfig:
    """Parameters controlling image discovery."""

    # File extensions treated as images (lowercase, including the dot)
    image_extensions: List[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp"]
    )

    # Number of parallel worker threads for the processing pipeline
    worker_threads: int = 2

    # Size (width, height) of stored face crop thumbnails
    thumbnail_size: tuple[int, int] = (128, 128)


@dataclass
class AppConfig:
    """Top-level application configuration."""

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    suggestions: SuggestionConfig = field(default_factory=SuggestionConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)

    # Base directory used to resolve relative paths in sub-configs.
    # Defaults to the current working directory.
    base_dir: str = field(default_factory=lambda: str(Path.cwd()))

    def resolve(self, relative: str) -> Path:
        """Return *relative* resolved against *base_dir*."""
        p = Path(relative)
        return p if p.is_absolute() else Path(self.base_dir) / p

    @property
    def db_path_resolved(self) -> Path:
        return self.resolve(self.storage.db_path)

    @property
    def crops_dir_resolved(self) -> Path:
        return self.resolve(self.storage.crops_dir)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load configuration from a YAML file, falling back to defaults.

    Args:
        config_path: Path to a YAML file.  ``None`` → pure defaults.

    Returns:
        Populated :class:`AppConfig`.
    """
    cfg = AppConfig()
    explicit_path = config_path
    discovered_path: Optional[Path] = None

    if config_path is not None:
        candidate = Path(config_path).expanduser()
        if candidate.exists():
            discovered_path = candidate.resolve()
    else:
        env_config = os.environ.get("FACE_LOCAL_CONFIG")
        candidates: list[Path] = []
        if env_config:
            candidates.append(Path(env_config).expanduser())

        if paths.is_frozen():
            candidates.extend(
                [
                    paths.user_config_dir() / "config.yaml",
                    paths.bundle_root() / "config.yaml",
                    paths.bundle_root() / "config.example.yaml",
                    Path("config.yaml"),
                    Path("config.example.yaml"),
                ]
            )
        else:
            candidates.extend(
                [
                    Path("config.yaml"),
                    Path("config.example.yaml"),
                ]
            )

        for candidate in candidates:
            if candidate.exists():
                discovered_path = candidate.resolve()
                break

    if discovered_path and discovered_path.exists():
        cfg.base_dir = str(discovered_path.parent)

        with open(discovered_path, "r", encoding="utf-8") as fh:
            raw: dict = yaml.safe_load(fh) or {}

        det = raw.get("detection", {})
        cfg.detection = DetectionConfig(
            confidence_threshold=det.get(
                "confidence_threshold", cfg.detection.confidence_threshold
            ),
            min_face_size=det.get("min_face_size", cfg.detection.min_face_size),
            coral_model_path=det.get("coral_model_path"),
            cpu_model_path=det.get("cpu_model_path"),
            cpu_model_input_size=tuple(
                det.get("cpu_model_input_size", list(cfg.detection.cpu_model_input_size))
            ),
        )

        emb = raw.get("embedding", {})
        cfg.embedding = EmbeddingConfig(
            model_path=emb.get("model_path"),
            input_size=tuple(emb.get("input_size", list(cfg.embedding.input_size))),
            embedding_dim=emb.get("embedding_dim", cfg.embedding.embedding_dim),
        )

        clu = raw.get("clustering", {})
        cfg.clustering = ClusteringConfig(
            epsilon=clu.get("epsilon", cfg.clustering.epsilon),
            min_samples=clu.get("min_samples", cfg.clustering.min_samples),
            metric=clu.get("metric", cfg.clustering.metric),
        )

        rec = raw.get("recognition", {})
        cfg.recognition = RecognitionConfig(
            auto_assign_threshold=rec.get(
                "auto_assign_threshold", cfg.recognition.auto_assign_threshold
            ),
            min_margin=rec.get("min_margin", cfg.recognition.min_margin),
            min_examples_per_person=rec.get(
                "min_examples_per_person", cfg.recognition.min_examples_per_person
            ),
            centroid_weight=rec.get(
                "centroid_weight", cfg.recognition.centroid_weight
            ),
            use_recognized_faces_for_training=rec.get(
                "use_recognized_faces_for_training",
                cfg.recognition.use_recognized_faces_for_training,
            ),
            profile_auto_min_confidence=rec.get(
                "profile_auto_min_confidence",
                cfg.recognition.profile_auto_min_confidence,
            ),
        )

        sug = raw.get("suggestions", {})
        cfg.suggestions = SuggestionConfig(
            similarity_threshold=sug.get(
                "similarity_threshold", cfg.suggestions.similarity_threshold
            ),
            max_suggestions_per_person=sug.get(
                "max_suggestions_per_person",
                cfg.suggestions.max_suggestions_per_person,
            ),
        )

        sto = raw.get("storage", {})
        cfg.storage = StorageConfig(
            crops_dir=sto.get("crops_dir", cfg.storage.crops_dir),
            db_path=sto.get("db_path", cfg.storage.db_path),
        )

        sc = raw.get("scan", {})
        cfg.scan = ScanConfig(
            image_extensions=sc.get(
                "image_extensions", cfg.scan.image_extensions
            ),
            worker_threads=sc.get("worker_threads", cfg.scan.worker_threads),
            thumbnail_size=tuple(
                sc.get("thumbnail_size", list(cfg.scan.thumbnail_size))
            ),
        )

        if "base_dir" in raw:
            cfg.base_dir = raw["base_dir"]
    elif paths.is_frozen():
        cfg.base_dir = str(paths.bundle_root())

    if paths.is_frozen():
        _apply_frozen_storage_defaults(
            cfg=cfg,
            discovered_path=discovered_path,
            explicit_path=explicit_path,
        )

    return cfg


def _apply_frozen_storage_defaults(
    cfg: AppConfig,
    discovered_path: Optional[Path],
    explicit_path: Optional[str],
) -> None:
    """Redirect default writable paths out of the app bundle."""
    bundle = paths.bundle_root().resolve()
    use_user_data_dir = explicit_path is None and (
        discovered_path is None or bundle == discovered_path.parent
    )
    if not use_user_data_dir:
        return

    data_root = paths.user_data_dir()

    if not Path(cfg.storage.db_path).is_absolute():
        cfg.storage.db_path = str(data_root / Path(cfg.storage.db_path))

    if not Path(cfg.storage.crops_dir).is_absolute():
        cfg.storage.crops_dir = str(data_root / Path(cfg.storage.crops_dir))


def _user_config_file() -> Path:
    """Return the writable config file path for the current runtime."""
    if paths.is_frozen():
        cfg_dir = paths.user_config_dir()
        cfg_dir.mkdir(parents=True, exist_ok=True)
        return cfg_dir / "config.yaml"
    # Dev: prefer existing config.yaml next to cwd
    for c in [
        Path(os.environ.get("FACE_LOCAL_CONFIG", "")),
        Path("config.yaml"),
    ]:
        if c.name and c.exists():
            return c
    return Path("config.yaml")


def save_db_path(new_db_path: str, config_path: Optional[str] = None) -> None:
    """Persist *new_db_path* into the storage.db_path field of the YAML config."""
    path = Path(config_path) if config_path else _user_config_file()
    path.parent.mkdir(parents=True, exist_ok=True)

    raw: dict = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    raw.setdefault("storage", {})["db_path"] = new_db_path

    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(raw, fh, allow_unicode=True, default_flow_style=False, sort_keys=False)
