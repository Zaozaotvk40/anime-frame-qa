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
@click.option("--deflicker", is_flag=True, help="Enable flicker suppression (video only)")
@click.option("--denoise", "denoise_enabled", is_flag=True, help="Enable noise reduction")
@click.option(
    "--denoise-method",
    type=click.Choice(["bilateral", "nlm"]),
    default="bilateral",
    help="Denoising method",
)
@click.option("--all", "all_modules", is_flag=True, help="Enable all modules")
def process(
    input_path: Path,
    output_path: Path,
    deflicker: bool,
    denoise_enabled: bool,
    denoise_method: str,
    all_modules: bool,
) -> None:
    """Process an image or video file."""
    config = PipelineConfig(
        deflicker=deflicker or all_modules,
        denoise_enabled=denoise_enabled or all_modules,
        denoise_method=DenoiseMethod(denoise_method),
    )

    click.echo(f"Processing: {input_path}")
    run(input_path, output_path, config)
    click.echo(f"Output: {output_path}")
