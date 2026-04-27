"""Core pipeline that chains modules together."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from anime_frame_qa.io import (
    VideoWriter,
    get_video_info,
    read_image,
    read_video_frames,
    write_image,
)
from anime_frame_qa.modules.deflicker import suppress_flicker_ema
from anime_frame_qa.modules.denoise import DenoiseMethod, denoise


@dataclass
class PipelineConfig:
    deflicker: bool = False
    deflicker_alpha: float = 0.7
    deflicker_threshold: float = 0.3
    denoise_enabled: bool = False
    denoise_method: DenoiseMethod = DenoiseMethod.BILATERAL


def process_image(image: np.ndarray, config: PipelineConfig) -> np.ndarray:
    result = image
    if config.denoise_enabled:
        result = denoise(result, method=config.denoise_method)
    return result


def process_video(
    input_path: Path,
    output_path: Path,
    config: PipelineConfig,
    chunk_size: int = 16,
) -> None:
    info = get_video_info(input_path)

    with VideoWriter(output_path, info["fps"], info["width"], info["height"]) as writer:
        for chunk in read_video_frames(input_path, chunk_size=chunk_size):
            processed = [process_image(f, config) for f in chunk]

            if config.deflicker:
                processed = suppress_flicker_ema(
                    processed,
                    alpha=config.deflicker_alpha,
                    threshold=config.deflicker_threshold,
                )

            for frame in processed:
                writer.write(frame)


def run(input_path: Path, output_path: Path, config: PipelineConfig) -> None:
    suffix = input_path.suffix.lower()
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    if suffix in video_exts:
        process_video(input_path, output_path, config)
    else:
        image = read_image(input_path)
        result = process_image(image, config)
        write_image(output_path, result)
