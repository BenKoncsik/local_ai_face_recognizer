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

    # --- Adaptive escalation ---

    # When the strict first pass finds ZERO faces, progressively relax the
    # detector (multi-variant preprocessing + lower confidence + smaller min
    # face size) until at least one face appears, then STOP.  This rescues
    # group / old / faded photos where the strict pass is blind, WITHOUT
    # loosening detection on normal photos that already succeed strictly —
    # those never reach the ladder, so no extra false positives there.
    adaptive_escalation: bool = True

    # Escalation ladder, tried in order, each only if the previous found 0 faces.
    # Each rung is (confidence_threshold, min_face_fraction).  The pixel min face
    # size at a rung is max(adaptive_min_face_floor, round(short_side * fraction))
    # where short_side = min(image_width, image_height).  Image-size aware: the
    # same fraction yields a larger pixel floor on big images, keeping tiny noise
    # blobs out.  We deliberately stop at 0.25 — lower thresholds produced false
    # positives (bushes, shirt patterns) in testing on real group photos.
    adaptive_ladder: tuple[tuple[float, float], ...] = (
        (0.40, 0.045),
        (0.30, 0.037),
        (0.25, 0.037),
    )

    # Absolute floor for the dynamic min face size (pixels). Never go below this
    # regardless of image size — guards against accepting sub-face noise.
    adaptive_min_face_floor: int = 22

    # Safety cap: if an escalation rung suddenly yields MORE detections than this
    # multiple of the previous rung's count (or this absolute number from zero),
    # treat the jump as likely false positives and keep the previous result.
    adaptive_max_faces: int = 60


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
class IntraImageDuplicateConfig:
    """Parameters for the same-image duplicate-detection cleanup.

    Re-detecting an already-processed photo can produce a *second* box over a
    face the user already marked, when the new box is shifted enough that the
    geometric IoU dedup in detection misses it (or when crop sizes differ a
    lot).  This embedding-based pass runs after embeddings exist and removes a
    freshly-detected, still-unassigned face when it both spatially overlaps and
    is embedding-near-identical to a retained (assigned or manual) face on the
    *same* image — i.e. it is the same physical face detected twice.

    It is deliberately conservative: a genuine second appearance of the same
    person elsewhere in the frame does not overlap, so it is never removed.
    """

    # When False the pipeline skips the pass entirely.
    enabled: bool = True

    # Minimum cosine similarity between the new face and a retained face for
    # them to be considered the *same physical face*.  High on purpose: two
    # crops of one face score well above this, while two different people score
    # far below it.
    duplicate_similarity: float = 0.90

    # Minimum bounding-box IoU required in addition to the embedding match.
    # Guards against deleting a real, non-overlapping second appearance of the
    # same person.  Set below the detection-time iou_merge_threshold (0.35) so
    # this pass catches exactly the shifted/contained boxes that slip past it.
    min_overlap: float = 0.10

    # Skip images with more faces than this (keeps the pairwise work bounded).
    max_faces_per_image: int = 80


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

    # --- Image-browser "re-recognize faces" workflow ---
    # Master switch for the user-triggered re-recognition of Unknown faces in
    # the image browser context menu.
    rerecognition_enabled: bool = True
    # Score at/above which an Unknown face is auto-merged into the matched
    # person without asking (same cosine scale as auto_assign_threshold).
    rerecognition_auto_threshold: float = 0.72
    # Score at/above which a match is *suggested* for user review.  Below this
    # nothing is proposed.  Must be < rerecognition_auto_threshold.
    rerecognition_suggest_threshold: float = 0.55

    # --- Auto-merge from Unknown (reviewable) ---
    # Master switch for the intelligent auto-confirm of faces dragged along when
    # one face of an "Unknown N" cluster is manually assigned to a named person.
    # When off, every auto-moved sibling stays "pending" for manual review.
    unknown_auto_merge_enabled: bool = True
    # Cosine similarity at/above which an auto-moved sibling face is
    # automatically confirmed (pending flag removed) — but only when the target
    # person has at least one manually confirmed reference face and the match is
    # unambiguous (margin >= min_margin).  Deliberately high.
    unknown_auto_confirm_threshold: float = 0.80


@dataclass
class DeepRecognitionConfig:
    """Parameters for the deep-learning recognition engine (the "new" path).

    A neural-network (MLP) ensemble is trained on the embeddings of every
    trusted labeled face on each run, with cross-validated per-person
    thresholds and open-set rejection.  Accuracy is preferred over speed:
    training may take minutes and saturate the CPU by design.
    """

    enabled: bool = True

    # Directory (relative to base_dir) where the trained model is persisted.
    model_dir: str = "data/deep_model"

    # --- Training ---
    # Number of independently seeded networks in the ensemble.
    ensemble_size: int = 5
    # Hidden layer sizes of each member network.
    hidden_layers: tuple = (256, 128)
    # Maximum optimiser iterations per network.
    max_iter: int = 600
    # Small classes are augmented (jitter + interpolation) up to this size.
    min_class_size: int = 8
    # Gaussian noise applied to synthetic samples (on unit-norm embeddings).
    augment_noise_sigma: float = 0.03
    # Cross-validation folds used to calibrate per-person thresholds.
    calibration_folds: int = 3
    # Skip retraining when the labeled data did not change since the last run.
    skip_unchanged: bool = True
    # The MLP ensemble only activates with at least this many labeled people
    # AND examples; below that, pure prototype matching is used (a tiny
    # discriminative network is overconfident and over-assigns).
    min_persons_for_ensemble: int = 4
    min_examples_for_ensemble: int = 30

    # --- Open-set / assignment gates ---
    # Fallback ensemble-probability threshold (per-person calibration may lower it).
    base_prob_threshold: float = 0.60
    # Hard floor for calibrated per-person probability thresholds.
    min_prob_threshold: float = 0.35
    # Required probability gap between the best and second-best person.
    min_margin: float = 0.10
    # Minimum cosine similarity to a real training example of the winner.
    # HARD floor: per-person calibration may only raise it, never lower it.
    min_prototype_similarity: float = 0.55
    # Required cosine-similarity gap between the winner and the most similar
    # training example of any OTHER person (ambiguous faces stay unknown).
    min_sim_margin: float = 0.05
    # Below this best-similarity to *anyone* known, the face is treated as an
    # outlier (stranger or non-face) and never auto-assigned.
    outlier_similarity: float = 0.42

    # --- Candidate filtering ("never recognise non-faces") ---
    # Faces below this detector confidence are never auto-assigned.
    min_face_confidence: float = 0.60
    # Skip low-quality faces (blurry / tiny / sideways) during auto-assignment.
    strict_quality_filter: bool = True

    # --- Continual learning from automatic assignments ---
    # Reuse very confident automatic assignments as extra training data.
    use_auto_assignments_for_training: bool = True
    # Confidence floor for an automatic assignment to count as training data.
    auto_training_min_confidence: float = 0.92

    # Use the slower high-accuracy detector pass in the deep pipeline.
    high_accuracy_detection: bool = False


@dataclass
class AiFaceDetectionConfig:
    """Parameters for the AI (deep learning) face-detection analysis pass.

    A standalone, analysis-only step of the AI pipeline: it answers *whether*
    there are faces on an image, *how many*, *where* (bounding boxes) and with
    what confidence — using a pretrained deep-learning detector (YuNet ONNX).
    It never assigns identities and never touches the classic detection /
    recognition results; its findings are stored separately in the
    ``ai_face_detections`` table.
    """

    enabled: bool = True

    # Minimum detector confidence for a detection to be recorded.
    confidence_threshold: float = 0.60

    # Minimum face bounding-box width AND height in pixels.
    min_face_size: int = 20

    # Optional explicit path to the YuNet ``.onnx`` model.  None → the bundled
    # default (models/face_detection_yunet_2023mar.onnx).
    model_path: Optional[str] = None

    # Also run this pass when the image browser's "re-recognize faces" action
    # is used (best-effort: an AI failure never breaks re-recognition).
    run_on_rerecognition: bool = True


@dataclass
class OverlapResolutionConfig:
    """Parameters for the same-image overlapping-box resolution pass.

    When the detector produced several boxes over the same physical face, only
    one survives: a manually drawn or person-assigned box always wins over an
    unknown one; between two unknown boxes the better-quality one is kept.
    """

    enabled: bool = True
    # Boxes overlapping at or above this IoU are considered the same face.
    iou_threshold: float = 0.35
    # A small box nested inside a larger one has low IoU but high containment
    # (intersection / smaller-box area); this catches that case.
    containment_threshold: float = 0.80
    # When both faces have embeddings, require at least this cosine similarity
    # before deleting — protects two genuinely different, tightly cropped faces.
    embedding_guard_similarity: float = 0.80
    # Above this IoU the boxes are geometrically the same spot, so the pair is
    # resolved even when embeddings disagree (e.g. one crop is corrupt).
    hard_iou_threshold: float = 0.65
    # Skip pathological images with more boxes than this.
    max_faces_per_image: int = 120


@dataclass
class IgnoredFaceConfig:
    """Parameters for the permanently-ignored faces filter.

    Faces the user excluded "forever" are stored as embeddings; the pipeline
    suppresses freshly embedded, still-unassigned faces that match one of the
    stored vectors so the same person never resurfaces as a new "Unknown N".
    """

    # When False the pipeline skips the filter entirely (the ignore list is
    # kept, just not applied).
    enabled: bool = True

    # Minimum cosine similarity between a new face and an ignored embedding
    # for the new face to be suppressed.  Deliberately stricter than the
    # recognition auto-assign threshold (0.72) so genuinely new people are
    # never silently swallowed by the ignore list.
    ignore_similarity: float = 0.80


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

    # --- Auto-merge settings ---
    # Minimum confidence threshold for automatic merging of unknown persons.
    auto_merge_min_confidence: float = 0.60
    # Maximum number of faces in an unknown person to allow automatic merging.
    auto_merge_max_unknown_faces: int = 5

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
    # Explicit audio device names (``None`` → auto-pick).  ``audio_input_device``
    # selects the microphone; ``system_audio_device`` selects the loopback.
    audio_input_device: Optional[str] = None
    system_audio_device: Optional[str] = None
    # Linear gain applied to each source before mixing (1.0 = unity).
    mic_volume: float = 1.0
    system_volume: float = 1.0
    # Drop a source from the mix without changing device selection.
    mute_microphone: bool = False
    mute_system_audio: bool = False
    # Explicit ffmpeg path; ``None`` → resolve from PATH.
    ffmpeg_path: Optional[str] = None
    # Concatenate the segments into a single final mp4 when recording stops.
    concat_on_stop: bool = True
    # Which part of the screen to capture: ``"active_window"`` | ``"all"`` |
    # ``"selected"`` (see ``RecordingDisplayMode``).
    display_mode: str = "all"
    # Monitor ids to record when ``display_mode == "selected"``.
    selected_display_ids: list[str] = field(default_factory=list)
    # Drop the frame rate automatically when capturing multiple monitors.
    auto_reduce_fps: bool = True
    # Frame-rate ceiling applied to multi-monitor captures.
    multi_monitor_fps_cap: int = 15


@dataclass
class AppConfig:
    """Top-level application configuration."""

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    intra_image: IntraImageConsistencyConfig = field(
        default_factory=IntraImageConsistencyConfig
    )
    intra_image_duplicate: IntraImageDuplicateConfig = field(
        default_factory=IntraImageDuplicateConfig
    )
    identity_repair: IdentityRepairConfig = field(default_factory=IdentityRepairConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    deep_recognition: DeepRecognitionConfig = field(
        default_factory=DeepRecognitionConfig
    )
    ai_face_detection: AiFaceDetectionConfig = field(
        default_factory=AiFaceDetectionConfig
    )
    overlap_resolution: OverlapResolutionConfig = field(
        default_factory=OverlapResolutionConfig
    )
    ignored_faces: IgnoredFaceConfig = field(default_factory=IgnoredFaceConfig)
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
            adaptive_escalation=det.get(
                "adaptive_escalation", cfg.detection.adaptive_escalation
            ),
            adaptive_ladder=tuple(
                tuple(rung)
                for rung in det.get(
                    "adaptive_ladder",
                    [list(r) for r in cfg.detection.adaptive_ladder],
                )
            ),
            adaptive_min_face_floor=det.get(
                "adaptive_min_face_floor", cfg.detection.adaptive_min_face_floor
            ),
            adaptive_max_faces=det.get(
                "adaptive_max_faces", cfg.detection.adaptive_max_faces
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

        iid = raw.get("intra_image_duplicate", {})
        cfg.intra_image_duplicate = IntraImageDuplicateConfig(
            enabled=iid.get("enabled", cfg.intra_image_duplicate.enabled),
            duplicate_similarity=iid.get(
                "duplicate_similarity",
                cfg.intra_image_duplicate.duplicate_similarity,
            ),
            min_overlap=iid.get(
                "min_overlap", cfg.intra_image_duplicate.min_overlap
            ),
            max_faces_per_image=iid.get(
                "max_faces_per_image",
                cfg.intra_image_duplicate.max_faces_per_image,
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
            rerecognition_enabled=rec.get(
                "rerecognition_enabled",
                cfg.recognition.rerecognition_enabled,
            ),
            rerecognition_auto_threshold=rec.get(
                "rerecognition_auto_threshold",
                cfg.recognition.rerecognition_auto_threshold,
            ),
            rerecognition_suggest_threshold=rec.get(
                "rerecognition_suggest_threshold",
                cfg.recognition.rerecognition_suggest_threshold,
            ),
            unknown_auto_merge_enabled=rec.get(
                "unknown_auto_merge_enabled",
                cfg.recognition.unknown_auto_merge_enabled,
            ),
            unknown_auto_confirm_threshold=rec.get(
                "unknown_auto_confirm_threshold",
                cfg.recognition.unknown_auto_confirm_threshold,
            ),
        )

        deep = raw.get("deep_recognition", {})
        cfg.deep_recognition = DeepRecognitionConfig(
            enabled=deep.get("enabled", cfg.deep_recognition.enabled),
            model_dir=deep.get("model_dir", cfg.deep_recognition.model_dir),
            ensemble_size=deep.get(
                "ensemble_size", cfg.deep_recognition.ensemble_size
            ),
            hidden_layers=tuple(
                deep.get("hidden_layers", list(cfg.deep_recognition.hidden_layers))
            ),
            max_iter=deep.get("max_iter", cfg.deep_recognition.max_iter),
            min_class_size=deep.get(
                "min_class_size", cfg.deep_recognition.min_class_size
            ),
            augment_noise_sigma=deep.get(
                "augment_noise_sigma", cfg.deep_recognition.augment_noise_sigma
            ),
            calibration_folds=deep.get(
                "calibration_folds", cfg.deep_recognition.calibration_folds
            ),
            skip_unchanged=deep.get(
                "skip_unchanged", cfg.deep_recognition.skip_unchanged
            ),
            min_persons_for_ensemble=deep.get(
                "min_persons_for_ensemble",
                cfg.deep_recognition.min_persons_for_ensemble,
            ),
            min_examples_for_ensemble=deep.get(
                "min_examples_for_ensemble",
                cfg.deep_recognition.min_examples_for_ensemble,
            ),
            base_prob_threshold=deep.get(
                "base_prob_threshold", cfg.deep_recognition.base_prob_threshold
            ),
            min_prob_threshold=deep.get(
                "min_prob_threshold", cfg.deep_recognition.min_prob_threshold
            ),
            min_margin=deep.get("min_margin", cfg.deep_recognition.min_margin),
            min_prototype_similarity=deep.get(
                "min_prototype_similarity",
                cfg.deep_recognition.min_prototype_similarity,
            ),
            min_sim_margin=deep.get(
                "min_sim_margin", cfg.deep_recognition.min_sim_margin
            ),
            outlier_similarity=deep.get(
                "outlier_similarity", cfg.deep_recognition.outlier_similarity
            ),
            min_face_confidence=deep.get(
                "min_face_confidence", cfg.deep_recognition.min_face_confidence
            ),
            strict_quality_filter=deep.get(
                "strict_quality_filter", cfg.deep_recognition.strict_quality_filter
            ),
            use_auto_assignments_for_training=deep.get(
                "use_auto_assignments_for_training",
                cfg.deep_recognition.use_auto_assignments_for_training,
            ),
            auto_training_min_confidence=deep.get(
                "auto_training_min_confidence",
                cfg.deep_recognition.auto_training_min_confidence,
            ),
            high_accuracy_detection=deep.get(
                "high_accuracy_detection",
                cfg.deep_recognition.high_accuracy_detection,
            ),
        )

        aifd = raw.get("ai_face_detection", {})
        cfg.ai_face_detection = AiFaceDetectionConfig(
            enabled=aifd.get("enabled", cfg.ai_face_detection.enabled),
            confidence_threshold=aifd.get(
                "confidence_threshold", cfg.ai_face_detection.confidence_threshold
            ),
            min_face_size=aifd.get(
                "min_face_size", cfg.ai_face_detection.min_face_size
            ),
            model_path=aifd.get("model_path", cfg.ai_face_detection.model_path),
            run_on_rerecognition=aifd.get(
                "run_on_rerecognition", cfg.ai_face_detection.run_on_rerecognition
            ),
        )

        ovr = raw.get("overlap_resolution", {})
        cfg.overlap_resolution = OverlapResolutionConfig(
            enabled=ovr.get("enabled", cfg.overlap_resolution.enabled),
            iou_threshold=ovr.get(
                "iou_threshold", cfg.overlap_resolution.iou_threshold
            ),
            containment_threshold=ovr.get(
                "containment_threshold",
                cfg.overlap_resolution.containment_threshold,
            ),
            embedding_guard_similarity=ovr.get(
                "embedding_guard_similarity",
                cfg.overlap_resolution.embedding_guard_similarity,
            ),
            hard_iou_threshold=ovr.get(
                "hard_iou_threshold", cfg.overlap_resolution.hard_iou_threshold
            ),
            max_faces_per_image=ovr.get(
                "max_faces_per_image", cfg.overlap_resolution.max_faces_per_image
            ),
        )

        ign = raw.get("ignored_faces", {})
        cfg.ignored_faces = IgnoredFaceConfig(
            enabled=ign.get("enabled", cfg.ignored_faces.enabled),
            ignore_similarity=ign.get(
                "ignore_similarity", cfg.ignored_faces.ignore_similarity
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
            audio_input_device=rec.get(
                "audio_input_device", cfg.recording.audio_input_device
            ),
            system_audio_device=rec.get(
                "system_audio_device", cfg.recording.system_audio_device
            ),
            mic_volume=rec.get("mic_volume", cfg.recording.mic_volume),
            system_volume=rec.get(
                "system_volume", cfg.recording.system_volume
            ),
            mute_microphone=rec.get(
                "mute_microphone", cfg.recording.mute_microphone
            ),
            mute_system_audio=rec.get(
                "mute_system_audio", cfg.recording.mute_system_audio
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
