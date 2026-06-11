"""Az ADAPTÍV detektálás ellenőrzése a valódi DetectionService kódjával.

Futás:
    python scripts/detect_adaptive_probe.py <kép> [<kép2> ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_config  # noqa: E402
from app.detectors.factory import create_detector  # noqa: E402
from app.services.detection_service import DetectionService  # noqa: E402
from app.utils.image_utils import load_image_bgr_normalized  # noqa: E402


def main(paths) -> None:
    cfg = load_config()
    detector = create_detector(cfg.detection)
    print(f"Aktív detektor: {detector.backend_name}\n")

    svc = DetectionService(session=None, detector=detector, config=cfg, high_accuracy=False)

    for p in paths:
        img = load_image_bgr_normalized(p)
        if img is None:
            print(f"{Path(p).name}: nem olvasható")
            continue
        h, w = img.shape[:2]
        dets = svc._run_detection(img)  # strict → ha 0, adaptív
        print(f"{Path(p).name}  ({w}x{h})  →  {len(dets)} arc")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Használat: python scripts/detect_adaptive_probe.py <kép> [...]")
        sys.exit(1)
    main(sys.argv[1:])
