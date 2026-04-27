from __future__ import annotations

from pathlib import Path

import yaml

from anime_frame_qa.config import load_config
from anime_frame_qa.pipeline import PipelineConfig
from anime_frame_qa.sweep import run_sweep


def test_load_config(tmp_path: Path) -> None:
    cfg = {
        "deflicker": True,
        "deflicker_alpha": 0.8,
        "denoise": True,
        "denoise_method": "nlm",
        "extract_edges": True,
        "color_consistency": True,
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)

    result = load_config(config_path)
    assert isinstance(result, PipelineConfig)
    assert result.deflicker is True
    assert result.deflicker_alpha == 0.8
    assert result.denoise_enabled is True
    assert result.extract_edges is True


def test_run_sweep(sample_image: Path, tmp_path: Path) -> None:
    sweep_cfg = {
        "parameters": [
            {"denoise": True, "denoise_method": "bilateral"},
            {"denoise": True, "denoise_method": "nlm"},
        ]
    }
    sweep_path = tmp_path / "sweep.yaml"
    with open(sweep_path, "w") as f:
        yaml.dump(sweep_cfg, f)

    output_dir = tmp_path / "sweep_out"
    results = run_sweep(sample_image, sweep_path, output_dir)

    assert len(results) == 2
    assert all(r.psnr > 0 for r in results)
    assert all(0 <= r.ssim <= 1 for r in results)
    assert (output_dir / "sweep_000.png").exists()
    assert (output_dir / "sweep_001.png").exists()
