"""Parameter sweep with optional W&B logging."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from anime_frame_qa.io import read_image, write_image
from anime_frame_qa.metrics import compute_psnr, compute_ssim
from anime_frame_qa.modules.denoise import DenoiseMethod
from anime_frame_qa.pipeline import PipelineConfig, process_image


@dataclass
class SweepResult:
    params: dict[str, Any]
    psnr: float
    ssim: float


def _build_config(params: dict[str, Any]) -> PipelineConfig:
    method = params.get("denoise_method", "bilateral")
    if isinstance(method, str):
        method = DenoiseMethod(method)
    return PipelineConfig(
        denoise_enabled=params.get("denoise", True),
        denoise_method=method,
    )


def load_sweep_config(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def run_sweep(
    input_path: Path,
    sweep_config_path: Path,
    output_dir: Path,
    wandb_project: str | None = None,
) -> list[SweepResult]:
    sweep_cfg = load_sweep_config(sweep_config_path)
    original = read_image(input_path)
    param_grid = sweep_cfg.get("parameters", [])

    wb_run = None
    if wandb_project:
        try:
            import wandb
            wb_run = wandb.init(project=wandb_project, config=sweep_cfg)
        except ImportError:
            raise ImportError(
                "wandb is required for experiment tracking. "
                "Install with: uv pip install anime-frame-qa[wandb]"
            )

    results: list[SweepResult] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, params in enumerate(param_grid):
        config = _build_config(params)
        processed = process_image(original, config)

        psnr = compute_psnr(original, processed)
        ssim = compute_ssim(original, processed)

        out_path = output_dir / f"sweep_{i:03d}.png"
        write_image(out_path, processed)

        result = SweepResult(params=params, psnr=psnr, ssim=ssim)
        results.append(result)

        if wb_run:
            import wandb
            wandb.log({"psnr": psnr, "ssim": ssim, **params, "step": i})

    if wb_run:
        import wandb
        wandb.finish()

    return results
