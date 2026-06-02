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

    # Path to the YuNet ONNX model (face_detection_yunet_2023mar.onnx).  YuNet
    # returns 5 facial landmarks per face, which the "aligned" embedding crop
    # mode needs.  When None the detector is auto-located in models/; set this
    # to point elsewhere.  Set use_yunet=False to force the Caffe/Haar CPU path.
    yunet_model_path: Optional[str] = None

    # Prefer the landmark-capable YuNet detector over the plain Caffe SSD / Haar
    # CPU detector when a YuNet model is available.  Required for crop_mode
    # "aligned" to produce real alignment.  Coral (when configured) still wins.
    use_yunet: bool = True

    # Input size expected by the CPU DNN model (width, height)
    cpu_model_input_size: tuple[int, int] = (300, 300)

    # --- High accuracy mode parameters ---

    # Lower confidence threshold used in high-accuracy multi-pass detection.
    # More detections are kept; IoU-based deduplication removes overlaps.
    # 0.25 works well for old / B&W photos where the SSD model gives lower scores.
    # Tested against res10_300x300_ssd: genuine faces typically score 0.25–0.95,
    # while most false positives are below 0.20.
    high_accuracy_confidence_threshold: float = 0.25

    # IoU threshold for merging overlapping bounding boxes from multiple
    # preprocessing variants in high-accuracy mode.
    iou_merge_threshold: float = 0.35

    # IoU threshold for the manual cleanup action that finds unassigned
    # question-mark boxes overlapping already named faces.
    duplicate_unknown_iou_threshold: float = 0.35

    # Containment threshold for the same cleanup action. A small box nested
    # inside a much larger one has a low IoU but a near-1.0 containment ratio
    # (intersection / smaller-box area); this catches that case.
    duplicate_unknown_containment_threshold: float = 0.80


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

    # How face crops are extracted before embedding.  Embedding models are
    # trained on *aligned* faces, so the crop geometry directly affects how
    # stable a person's embedding is across photos.
    #   "legacy"  — bbox + margin stretched to a square (original behaviour;
    #               distorts non-square boxes).
    #   "square"  — aspect-preserving square crop (no landmarks needed).
    #   "aligned" — 5-point similarity-transform alignment to the ArcFace
    #               template (requires landmarks; falls back to "square").
    # WARNING: changing this on an existing library changes every embedding —
    # a full re-detect + re-embed is required for results to stay comparable.
    crop_mode: str = "legacy"


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

    # Maximum cosine distance to assign an unassigned face to an *existing*
    # Unknown person (incremental clustering in the pipeline).
    # Should be >= epsilon so faces that would cluster together also match existing ones.
    unknown_assign_threshold: float = 0.45

    # Minimum cluster size to create a new Unknown person.
    # Clusters smaller than this remain unassigned (avoid singleton Unknown spam).
    create_unknown_min_cluster_size: int = 2


@dataclass
class IntraImageConsistencyConfig:
    """Parameters for the same-image identity consistency pass.

    After clustering, two faces of the *same* person on the *same* photo can
    end up under different identities (e.g. one matched ``Unknown 98`` at
    cosine distance 0.44 while a sibling face fell just past the threshold and
    spawned ``Unknown 155``).  This pass re-unifies faces on one image when
    their embeddings are mutually near-identical, healing that fragmentation
    without re-embedding.
    """

    # When False the pipeline skips the pass entirely.
    enabled: bool = True

    # Minimum cosine similarity between two faces *on the same image* for them
    # to be treated as the same identity.  Deliberately stricter than the
    # clustering boundary (epsilon=0.4 → 0.60 similarity) so that two genuinely
    # different people who merely co-occur are never merged.
    merge_similarity: float = 0.62

    # Do not act on images with more faces than this (group photos blow up the
    # O(n²) pairwise comparison and rarely suffer the boundary-split bug).
    max_faces_per_image: int = 40


@dataclass
class IdentityRepairConfig:
    """Parameters for the global Identity Repair Scan.

    Walks every auto-named ("Unknown N") person and proposes merges between
    those whose embedding centroids are highly similar, consolidating identity
    fragments that accumulated across many incremental pipeline runs.
    """

    # Minimum centroid cosine similarity for two Unknown persons to be proposed
    # as the same identity.  Stricter than clustering to keep suggestions safe.
    merge_similarity: float = 0.66

    # Also require the closest *individual* face pair between the two persons to
    # reach this similarity (guards against centroid blur on mixed clusters).
    min_pair_similarity: float = 0.60

    # Maximum merge candidates returned per person.
    max_candidates_per_person: int = 5


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

    # --- Adaptive threshold (Pass 1) ---
    # When True, low-quality / small / profile faces get a lower threshold so
    # they are not unfairly penalised by the fixed base threshold.
    adaptive_threshold_enabled: bool = True
    # Absolute floor for the adaptive threshold — never goes below this value.
    adaptive_min_threshold: float = 0.55

    # --- Same-image context-assisted recognition (Pass 2) ---
    # When True, faces that survive pass 1 unresolved are retried with a lower
    # threshold when the same image already has at least one manually confirmed
    # person.  Matching is restricted to those confirmed persons only.
    same_image_assist_enabled: bool = True
    # Threshold used in the assisted second pass.
    same_image_assist_threshold: float = 0.62
    # Minimum number of distinct confirmed persons on the image to activate assist.
    same_image_assist_min_confirmed: int = 1
    # Margin requirement for the assisted pass (looser than main margin since
    # the candidate set is already restricted by image context).
    same_image_assist_margin: float = 0.05


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
class MatchingConfig:
    """Parameters for the background person merge-suggestion engine.

    The engine compares persons in a background worker pool and persists
    *suggestions* (never automatic merges).  Confidence combines face
    similarity (dominant) with fuzzy name similarity.
    """

    # Minimum centroid cosine similarity for a face-driven suggestion.
    face_threshold: float = 0.5
    # Minimum fuzzy name similarity for a name to count as supporting evidence.
    name_threshold: float = 0.85
    # A name match only contributes when face similarity is at least this high
    # (a name alone must never be enough to suggest a confident merge).
    name_supported_face_floor: float = 0.35
    # Minimum combined confidence required to persist a suggestion.
    min_confidence: float = 0.45
    # Maximum ranked target suggestions kept per candidate person.
    max_suggestions_per_person: int = 3

    # --- Background worker tuning ---
    # Persons scored per chunk handed to the thread pool.
    chunk_size: int = 64
    # CPUs left free for the UI; worker pool size = max(1, cpu_count - reserved).
    reserved_cpus: int = 1
    # Hard cap on worker threads (None → derive from CPU count).
    max_workers: Optional[int] = None
    # Minimum interval between progress signals to the UI (milliseconds).
    progress_throttle_ms: int = 250


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
class RecordingConfig:
    """Screen-recording (documentation capture) parameters.

    The recorder shells out to the system ``ffmpeg`` binary.  Audio capture
    always includes the microphone; system/speaker audio is best-effort and
    only used when a virtual loopback device is auto-detected (macOS: BlackHole
    / Loopback, Windows: a WASAPI ``virtual-audio-capturer``).
    """

    # Last-used / configured output directory.  ``None`` → ask on first start.
    output_dir: Optional[str] = None
    # Capture frame rate (documentation quality; 15–20 recommended).
    fps: int = 18
    # Quality preset: ``"low"`` | ``"normal"`` | ``"better"``.
    quality: str = "normal"
    # Length of each on-disk segment in seconds (crash-protection granularity).
    segment_seconds: int = 8
    # Draw the mouse cursor into the recording.
    capture_cursor: bool = True
    # Record the microphone (mandatory by design — kept as a flag for clarity).
    capture_microphone: bool = True
    # Best-effort system/speaker audio (needs a virtual loopback device).
    capture_system_audio: bool = True
    # Explicit ffmpeg path; ``None`` → resolve from PATH.
    ffmpeg_path: Optional[str] = None
    # Concatenate the segments into a single final mp4 when recording stops.
    concat_on_stop: bool = True


@dataclass
class AppConfig:
    """Top-level application configuration."""

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    intra_image: IntraImageConsistencyConfig = field(
        default_factory=IntraImageConsistencyConfig
    )
    identity_repair: IdentityRepairConfig = field(default_factory=IdentityRepairConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    suggestions: SuggestionConfig = field(default_factory=SuggestionConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)

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
            yunet_model_path=det.get("yunet_model_path"),
            use_yunet=det.get("use_yunet", cfg.detection.use_yunet),
            cpu_model_input_size=tuple(
                det.get("cpu_model_input_size", list(cfg.detection.cpu_model_input_size))
            ),
            high_accuracy_confidence_threshold=det.get(
                "high_accuracy_confidence_threshold",
                cfg.detection.high_accuracy_confidence_threshold,
            ),
            iou_merge_threshold=det.get(
                "iou_merge_threshold", cfg.detection.iou_merge_threshold
            ),
            duplicate_unknown_iou_threshold=det.get(
                "duplicate_unknown_iou_threshold",
                cfg.detection.duplicate_unknown_iou_threshold,
            ),
            duplicate_unknown_containment_threshold=det.get(
                "duplicate_unknown_containment_threshold",
                cfg.detection.duplicate_unknown_containment_threshold,
            ),
        )

        emb = raw.get("embedding", {})
        cfg.embedding = EmbeddingConfig(
            model_path=emb.get("model_path"),
            input_size=tuple(emb.get("input_size", list(cfg.embedding.input_size))),
            embedding_dim=emb.get("embedding_dim", cfg.embedding.embedding_dim),
            crop_mode=emb.get("crop_mode", cfg.embedding.crop_mode),
        )

        clu = raw.get("clustering", {})
        cfg.clustering = ClusteringConfig(
            epsilon=clu.get("epsilon", cfg.clustering.epsilon),
            min_samples=clu.get("min_samples", cfg.clustering.min_samples),
            metric=clu.get("metric", cfg.clustering.metric),
            unknown_assign_threshold=clu.get(
                "unknown_assign_threshold", cfg.clustering.unknown_assign_threshold
            ),
            create_unknown_min_cluster_size=clu.get(
                "create_unknown_min_cluster_size",
                cfg.clustering.create_unknown_min_cluster_size,
            ),
        )

        iic = raw.get("intra_image", {})
        cfg.intra_image = IntraImageConsistencyConfig(
            enabled=iic.get("enabled", cfg.intra_image.enabled),
            merge_similarity=iic.get(
                "merge_similarity", cfg.intra_image.merge_similarity
            ),
            max_faces_per_image=iic.get(
                "max_faces_per_image", cfg.intra_image.max_faces_per_image
            ),
        )

        rep = raw.get("identity_repair", {})
        cfg.identity_repair = IdentityRepairConfig(
            merge_similarity=rep.get(
                "merge_similarity", cfg.identity_repair.merge_similarity
            ),
            min_pair_similarity=rep.get(
                "min_pair_similarity", cfg.identity_repair.min_pair_similarity
            ),
            max_candidates_per_person=rep.get(
                "max_candidates_per_person",
                cfg.identity_repair.max_candidates_per_person,
            ),
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
            adaptive_threshold_enabled=rec.get(
                "adaptive_threshold_enabled",
                cfg.recognition.adaptive_threshold_enabled,
            ),
            adaptive_min_threshold=rec.get(
                "adaptive_min_threshold",
                cfg.recognition.adaptive_min_threshold,
            ),
            same_image_assist_enabled=rec.get(
                "same_image_assist_enabled",
                cfg.recognition.same_image_assist_enabled,
            ),
            same_image_assist_threshold=rec.get(
                "same_image_assist_threshold",
                cfg.recognition.same_image_assist_threshold,
            ),
            same_image_assist_min_confirmed=rec.get(
                "same_image_assist_min_confirmed",
                cfg.recognition.same_image_assist_min_confirmed,
            ),
            same_image_assist_margin=rec.get(
                "same_image_assist_margin",
                cfg.recognition.same_image_assist_margin,
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

        mat = raw.get("matching", {})
        cfg.matching = MatchingConfig(
            face_threshold=mat.get("face_threshold", cfg.matching.face_threshold),
            name_threshold=mat.get("name_threshold", cfg.matching.name_threshold),
            name_supported_face_floor=mat.get(
                "name_supported_face_floor", cfg.matching.name_supported_face_floor
            ),
            min_confidence=mat.get("min_confidence", cfg.matching.min_confidence),
            max_suggestions_per_person=mat.get(
                "max_suggestions_per_person", cfg.matching.max_suggestions_per_person
            ),
            chunk_size=mat.get("chunk_size", cfg.matching.chunk_size),
            reserved_cpus=mat.get("reserved_cpus", cfg.matching.reserved_cpus),
            max_workers=mat.get("max_workers", cfg.matching.max_workers),
            progress_throttle_ms=mat.get(
                "progress_throttle_ms", cfg.matching.progress_throttle_ms
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

        rec = raw.get("recording", {})
        cfg.recording = RecordingConfig(
            output_dir=rec.get("output_dir", cfg.recording.output_dir),
            fps=rec.get("fps", cfg.recording.fps),
            quality=rec.get("quality", cfg.recording.quality),
            segment_seconds=rec.get(
                "segment_seconds", cfg.recording.segment_seconds
            ),
            capture_cursor=rec.get(
                "capture_cursor", cfg.recording.capture_cursor
            ),
            capture_microphone=rec.get(
                "capture_microphone", cfg.recording.capture_microphone
            ),
            capture_system_audio=rec.get(
                "capture_system_audio", cfg.recording.capture_system_audio
            ),
            ffmpeg_path=rec.get("ffmpeg_path", cfg.recording.ffmpeg_path),
            concat_on_stop=rec.get(
                "concat_on_stop", cfg.recording.concat_on_stop
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
