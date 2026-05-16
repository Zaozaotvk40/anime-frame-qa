from __future__ import annotations

from pathlib import Path

import click

from anime_frame_qa.modules.denoise import DenoiseMethod
from anime_frame_qa.pipeline import PipelineConfig, run


@click.group()
def main() -> None:
    """anime-frame-qa: Quality assurance pipeline for AI-generated anime frames."""


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", "output_path", type=click.Path(path_type=Path), required=True)
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), help="YAML config file")
@click.option("--deflicker", is_flag=True, help="Enable flicker suppression (video only)")
@click.option("--denoise", "denoise_enabled", is_flag=True, help="Enable noise reduction")
@click.option(
    "--denoise-method",
    type=click.Choice(["bilateral", "nlm", "banding"]),
    default="bilateral",
    help="Denoising method",
)
@click.option("--extract-edges", is_flag=True, help="Extract and analyze edges/contours")
@click.option("--color-consistency", is_flag=True, help="Enforce inter-frame color consistency (video only)")
@click.option("--remove-bg", is_flag=True, help="Remove background (requires cnn extras)")
@click.option("--inpaint", is_flag=True, help="Inpaint masked regions (requires cnn extras)")
@click.option("--inpaint-mask", type=click.Path(exists=True, path_type=Path), help="Mask image for inpainting (white=repair)")
@click.option("--all", "all_modules", is_flag=True, help="Enable all core OpenCV modules (deflicker and color-consistency apply to video only)")
def process(
    input_path: Path,
    output_path: Path,
    config_path: Path | None,
    deflicker: bool,
    denoise_enabled: bool,
    denoise_method: str,
    extract_edges: bool,
    color_consistency: bool,
    remove_bg: bool,
    inpaint: bool,
    inpaint_mask: Path | None,
    all_modules: bool,
) -> None:
    """Process an image or video file."""
    if config_path:
        from anime_frame_qa.config import load_config
        config = load_config(config_path)
    else:
        config = PipelineConfig(
            deflicker=deflicker or all_modules,
            denoise_enabled=denoise_enabled or all_modules,
            denoise_method=DenoiseMethod(denoise_method),
            extract_edges=extract_edges or all_modules,
            color_consistency=color_consistency or all_modules,
            remove_bg=remove_bg,
            inpaint=inpaint,
            inpaint_mask_path=inpaint_mask,
        )

    click.echo(f"Processing: {input_path}")
    run(input_path, output_path, config)
    click.echo(f"Output: {output_path}")


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("sweep_config", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output-dir", type=click.Path(path_type=Path), required=True)
@click.option("--wandb-project", type=str, default=None, help="W&B project name for logging")
@click.option("--reference", "reference_path", type=click.Path(exists=True, path_type=Path), default=None, help="Ground truth image for PSNR/SSIM evaluation")
def sweep(
    input_path: Path,
    sweep_config: Path,
    output_dir: Path,
    wandb_project: str | None,
    reference_path: Path | None,
) -> None:
    """Run parameter sweep on an image."""
    from anime_frame_qa.sweep import run_sweep

    click.echo(f"Running sweep: {sweep_config}")
    results = run_sweep(input_path, sweep_config, output_dir, wandb_project, reference_path)

    click.echo(f"\nResults ({len(results)} runs):")
    for i, r in enumerate(results):
        click.echo(f"  [{i}] PSNR={r.psnr:.2f} SSIM={r.ssim:.4f} {r.params}")
