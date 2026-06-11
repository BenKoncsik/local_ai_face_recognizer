"""Ad-hoc detektálási próba egy képen.

Futás:
    python scripts/detect_probe.py /eleresi/ut/a/kephez.jpg

Kiírja, hány arcot talál a YuNet és a CPU detektor különböző
konfidencia- és min_face_size-beállításokkal — a "fast" és a
"high_accuracy" többmenetes preprocessinggel együtt.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.utils.image_utils import (  # noqa: E402
    load_image_bgr_normalized,
    apply_clahe,
    apply_gamma,
    apply_histeq,
    apply_bilateral,
)
from app.detectors.yunet_detector import YuNetDetector  # noqa: E402
from app.detectors.cpu_detector import CpuDetector  # noqa: E402


def iou(a, b) -> float:
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union else 0.0


def merge(dets, thr=0.35):
    dets = sorted(dets, key=lambda d: d.confidence, reverse=True)
    kept = []
    for d in dets:
        if all(iou(d, k) <= thr for k in kept):
            kept.append(d)
    return kept


def high_accuracy(det, img, conf, min_size):
    out = []
    for _, v in [
        ("orig", img),
        ("clahe", apply_clahe(img)),
        ("gamma", apply_gamma(img, 0.7)),
        ("histeq", apply_histeq(img)),
        ("bilat", apply_bilateral(img)),
    ]:
        try:
            out.extend(det.detect(v, confidence_threshold=conf, min_face_size=min_size))
        except Exception as e:  # noqa: BLE001
            print(f"    variant hiba: {e}")
    return merge(out)


def main(path: str) -> None:
    img = load_image_bgr_normalized(path)
    if img is None:
        print(f"Nem olvasható: {path}")
        return
    h, w = img.shape[:2]
    print(f"Kép: {w}x{h}\n")

    detectors = []
    try:
        detectors.append(("YuNet", YuNetDetector()))
    except Exception as e:  # noqa: BLE001
        print(f"YuNet nem elérhető: {e}")
    try:
        detectors.append(("CPU/SSD", CpuDetector()))
    except Exception as e:  # noqa: BLE001
        print(f"CPU nem elérhető: {e}")

    print(f"{'detektor':10} {'mód':14} {'conf':>5} {'minsize':>7} {'arcok':>6}")
    print("-" * 50)
    for name, det in detectors:
        # Jelenlegi alapértelmezés
        for conf, ms in [(0.65, 50)]:
            n = len(det.detect(img, confidence_threshold=conf, min_face_size=ms))
            print(f"{name:10} {'fast (alap)':14} {conf:>5} {ms:>7} {n:>6}")
        # Lazább fast
        for conf, ms in [(0.40, 30), (0.30, 24)]:
            n = len(det.detect(img, confidence_threshold=conf, min_face_size=ms))
            print(f"{name:10} {'fast':14} {conf:>5} {ms:>7} {n:>6}")
        # High accuracy
        for conf, ms in [(0.25, 30), (0.25, 24), (0.20, 20)]:
            n = len(high_accuracy(det, img, conf, ms))
            print(f"{name:10} {'high_accuracy':14} {conf:>5} {ms:>7} {n:>6}")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Használat: python scripts/detect_probe.py <kép útvonal>")
        sys.exit(1)
    main(sys.argv[1])
