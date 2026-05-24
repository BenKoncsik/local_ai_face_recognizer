from __future__ import annotations

from pathlib import Path

from app import config as config_module
from app import paths


def test_load_config_resolves_relative_paths_against_config_location(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "\n".join(
            [
                "storage:",
                "  db_path: custom/faces.db",
                "  crops_dir: custom/crops",
                "embedding:",
                "  model_path: models/mobilefacenet.tflite",
                "recognition:",
                "  auto_assign_threshold: 0.81",
                "  min_margin: 0.12",
                "  use_recognized_faces_for_training: false",
            ]
        ),
        encoding="utf-8",
    )

    cfg = config_module.load_config(str(cfg_file))

    assert Path(cfg.base_dir) == tmp_path
    assert cfg.db_path_resolved == tmp_path / "custom" / "faces.db"
    assert cfg.crops_dir_resolved == tmp_path / "custom" / "crops"
    assert cfg.resolve(cfg.embedding.model_path) == tmp_path / "models" / "mobilefacenet.tflite"
    assert cfg.recognition.auto_assign_threshold == 0.81
    assert cfg.recognition.min_margin == 0.12
    assert cfg.recognition.use_recognized_faces_for_training is False


def test_frozen_bundle_defaults_use_user_data_dir(tmp_path: Path, monkeypatch) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "config.example.yaml").write_text(
        "\n".join(
            [
                "storage:",
                "  db_path: data/faces.db",
                "  crops_dir: data/crops",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(paths, "is_frozen", lambda: True)
    monkeypatch.setattr(paths, "bundle_root", lambda: bundle_dir)
    monkeypatch.setattr(paths, "user_data_dir", lambda: tmp_path / "user-data")
    monkeypatch.setattr(paths, "user_config_dir", lambda: tmp_path / "user-config")

    cfg = config_module.load_config()

    assert Path(cfg.base_dir) == bundle_dir
    assert cfg.db_path_resolved == tmp_path / "user-data" / "data" / "faces.db"
    assert cfg.crops_dir_resolved == tmp_path / "user-data" / "data" / "crops"
