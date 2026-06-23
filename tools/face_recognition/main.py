"""Example CLI: compare a probe image against a reference.

Usage
-----
    python main.py --image probe.jpg --ref reference.jpg [options]

Options
-------
    --no-double-check   Disable the augmented double-embedding gate
    --no-liveness       Disable the liveness heuristic filter
    --gpu               Force GPU (default: auto-detect)
    --cpu               Force CPU
    --thresh-cos FLOAT  Override cosine similarity threshold (default 0.45)
    --thresh-l2  FLOAT  Override euclidean distance threshold (default 0.85)
    --verbose           Print per-step details
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger("face_recognition.main")


def _load_or_abort(path: str, label: str) -> np.ndarray:
    from utils.image import load_image_rgb

    img = load_image_rgb(path)
    if img is None:
        log.error("Cannot load %s: %s", label, path)
        sys.exit(1)
    return img


def _extract_reference_embedding(ref_img, detector, embedder, quality_filter):
    """Detect the face in the reference image and return its L2 embedding."""
    from face_quality import FaceQualityFilter

    faces = detector.detect(ref_img)
    if not faces:
        log.error("No face detected in reference image.")
        sys.exit(1)

    face = faces[0]
    log.info("Reference: detected %d face(s), using highest-confidence (conf=%.3f)",
             len(faces), face.confidence)

    qresult = quality_filter.check(face, ref_img, run_liveness=False)
    if not qresult.accepted:
        log.warning(
            "Reference face failed quality check: %s — embedding anyway.",
            ", ".join(qresult.reasons),
        )

    emb = embedder.embed(ref_img, face)
    if emb is None:
        log.error("Could not extract embedding from reference image.")
        sys.exit(1)

    log.info("Reference embedding extracted (dim=%d, norm=%.4f)",
             len(emb), float(np.linalg.norm(emb)))
    return emb


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Face verification: compare a probe image against a reference."
    )
    parser.add_argument("--image", required=True, help="Probe image path")
    parser.add_argument("--ref",   required=True, help="Reference image path")
    parser.add_argument("--no-double-check", action="store_true",
                        help="Disable augmented double-embedding gate")
    parser.add_argument("--no-liveness",     action="store_true",
                        help="Disable liveness heuristic filter")
    parser.add_argument("--gpu",  action="store_true",  help="Force GPU")
    parser.add_argument("--cpu",  action="store_true",  help="Force CPU")
    parser.add_argument("--thresh-cos", type=float, default=0.45,
                        help="Cosine similarity threshold (default 0.45)")
    parser.add_argument("--thresh-l2",  type=float, default=0.85,
                        help="Euclidean distance threshold (default 0.85)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    gpu: bool | None = None
    if args.gpu:
        gpu = True
    elif args.cpu:
        gpu = False

    # ── Initialise models ────────────────────────────────────────────────────
    log.info("Loading InsightFace buffalo_l …")
    from face_detector import FaceDetector
    from face_embedder import FaceEmbedder
    from face_quality import FaceQualityFilter
    from face_matcher import FaceMatcher, verify_face

    detector = FaceDetector(gpu=gpu)
    embedder = FaceEmbedder(detector)
    quality_filter = FaceQualityFilter()
    matcher = FaceMatcher(
        cosine_threshold=args.thresh_cos,
        euclidean_threshold=args.thresh_l2,
    )

    # ── Load images ──────────────────────────────────────────────────────────
    ref_img   = _load_or_abort(args.ref,   "reference")
    probe_img = _load_or_abort(args.image, "probe")

    log.info("Reference image: %s (%dx%d)", args.ref,   ref_img.shape[1],   ref_img.shape[0])
    log.info("Probe image:     %s (%dx%d)", args.image, probe_img.shape[1], probe_img.shape[0])

    # ── Build reference embedding ────────────────────────────────────────────
    ref_emb = _extract_reference_embedding(ref_img, detector, embedder, quality_filter)

    # ── Verify probe ────────────────────────────────────────────────────────
    result = verify_face(
        img_rgb=probe_img,
        reference_embedding=ref_emb,
        detector=detector,
        embedder=embedder,
        quality_filter=quality_filter,
        matcher=matcher,
        double_check=not args.no_double_check,
        run_liveness=not args.no_liveness,
    )

    # ── Output ───────────────────────────────────────────────────────────────
    verdict = "MATCH ✓" if result["is_match"] else "NO MATCH ✗"
    print()
    print(f"  Result:            {verdict}")
    if result["error"]:
        print(f"  Pipeline error:    {result['error']}")
    if result["cosine"] is not None:
        print(f"  Cosine similarity: {result['cosine']:.4f}  (threshold > {args.thresh_cos})")
        print(f"  Euclidean dist.:   {result['euclidean']:.4f}  (threshold < {args.thresh_l2})")
    if result["cosine_aug0"] is not None:
        print(f"  Augment-0 cosine:  {result['cosine_aug0']:.4f}")
        print(f"  Augment-0 L2:      {result['euclidean_aug0']:.4f}")
        print(f"  Augment-1 cosine:  {result['cosine_aug1']:.4f}")
        print(f"  Augment-1 L2:      {result['euclidean_aug1']:.4f}")
    if result["quality"]:
        q = result["quality"]
        status = "accepted" if q.accepted else f"rejected ({', '.join(q.reasons)})"
        print(f"  Quality:           {status}")
        if q.blur_score is not None:
            print(f"  Blur score:        {q.blur_score:.1f}")
        if q.pose is not None:
            print(f"  Pose (yaw/pitch/roll): {q.pose[0]:.1f}° / {q.pose[1]:.1f}° / {q.pose[2]:.1f}°")
    print(f"  Faces in probe:    {result['n_faces']}")
    print()

    sys.exit(0 if result["is_match"] else 1)


if __name__ == "__main__":
    main()
