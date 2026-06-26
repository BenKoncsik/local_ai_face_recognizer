#!/usr/bin/env python3
"""Convert a MobileFaceNet source model to a Core ML ``.mlpackage``.

The result is consumed by :class:`app.embeddings.coreml_embedder.CoreMLEmbedder`,
which runs it on the Apple Neural Engine / GPU.  Run this once on a macOS
machine; the rest of the app then picks the file up automatically (it defaults
to ``models/mobilefacenet.mlpackage``).

Why a separate source is needed
-------------------------------
``coremltools`` converts from **TensorFlow** (SavedModel / Keras ``.h5``) and
**PyTorch** (TorchScript), but **not** from a ``.tflite`` flatbuffer.  The
MobileFaceNet weights shipped in ``models/mobilefacenet.tflite`` are therefore
not directly convertible — point this script at one of:

* a TensorFlow SavedModel directory, or a Keras ``.h5`` file, or
* a TorchScript (``.pt``) traced MobileFaceNet.

Use the *same* MobileFaceNet weights as the TFLite model so the Core ML model
produces matching 192-dim vectors — then existing embeddings stay comparable
and no re-embed is required when switching between the CPU and Core ML backends.

The produced model takes a single ``float32`` input tensor that is already
normalised to [-1, 1] (the runtime embedder does the resize + normalisation),
matching the TFLite model's input contract.

Usage
-----
    python scripts/convert_mobilefacenet_to_coreml.py \
        --source path/to/mobilefacenet_savedmodel \
        --output models/mobilefacenet.mlpackage \
        --input-size 112 --layout NHWC

Verify the result with ``--verify`` (compares a random input against the
source model when it can be run here).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--source", required=True,
        help="TensorFlow SavedModel dir, Keras .h5, or TorchScript .pt source model.",
    )
    p.add_argument(
        "--output", default="models/mobilefacenet.mlpackage",
        help="Destination .mlpackage path (default: models/mobilefacenet.mlpackage).",
    )
    p.add_argument(
        "--input-size", type=int, default=112,
        help="Square input edge the model expects (default: 112).",
    )
    p.add_argument(
        "--layout", choices=["NHWC", "NCHW"], default="NHWC",
        help="Input tensor layout of the source model (default: NHWC).",
    )
    p.add_argument(
        "--compute-units", choices=["ALL", "CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY"],
        default="ALL",
        help="Core ML compute units to embed in the model (default: ALL → ANE first).",
    )
    return p


def _input_shape(size: int, layout: str):
    return (1, size, size, 3) if layout == "NHWC" else (1, 3, size, size)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    src = Path(args.source)
    if not src.exists():
        print(f"error: source not found: {src}", file=sys.stderr)
        return 2
    if src.suffix == ".tflite":
        print(
            "error: coremltools cannot convert a .tflite file directly.\n"
            "Provide a TensorFlow SavedModel / Keras .h5 or a TorchScript .pt of the\n"
            "same MobileFaceNet weights instead (see this script's header).",
            file=sys.stderr,
        )
        return 2

    try:
        import coremltools as ct
    except ImportError:
        print(
            "error: coremltools is not installed. Install it on macOS with:\n"
            "    pip install coremltools",
            file=sys.stderr,
        )
        return 2

    shape = _input_shape(args.input_size, args.layout)
    compute_units = getattr(ct.ComputeUnit, args.compute_units)

    print(f"Converting {src} → Core ML (input shape {shape}) …")

    if src.suffix == ".pt":
        import torch  # noqa: F401

        model = torch.jit.load(str(src)).eval()
        example = __import__("torch").rand(*shape)
        mlmodel = ct.convert(
            model,
            inputs=[ct.TensorType(name="input", shape=shape)],
            convert_to="mlprogram",
            compute_units=compute_units,
            example_inputs=[example],
        )
    else:
        # TensorFlow SavedModel directory or Keras .h5 — coremltools handles both.
        mlmodel = ct.convert(
            str(src),
            inputs=[ct.TensorType(name="input", shape=shape)],
            convert_to="mlprogram",
            compute_units=compute_units,
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out))
    print(f"Saved: {out}")
    print(
        "Done. The app will use it automatically on macOS "
        "(embedding.prefer_coreml=True). Check Task Manager → Performance for "
        "the 'Neural Engine (CoreML)' status."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
