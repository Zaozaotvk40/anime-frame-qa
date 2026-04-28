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
from anime_frame_qa.modules.color import enforce_color_consistency
from anime_frame_qa.modules.deflicker import suppress_flicker_ema
from anime_frame_qa.modules.denoise import DenoiseMethod, denoise
from anime_frame_qa.modules.edge import process_edges


@dataclass
class PipelineConfig:
    deflicker: bool = False
    deflicker_alpha: float = 0.7
    deflicker_threshold: float = 0.3
    denoise_enabled: bool = False
    denoise_method: DenoiseMethod = DenoiseMethod.BILATERAL
    extract_edges: bool = False
    edge_connect_gaps: bool = True
    color_consistency: bool = False
    color_window: int = 5
    remove_bg: bool = False
    inpaint: bool = False
    inpaint_mask_path: Path | None = None


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

            if config.color_consistency:
                processed = enforce_color_consistency(
                    processed, window_size=config.color_window
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

        if config.extract_edges:
            edge_result = process_edges(
                image, connect=config.edge_connect_gaps
            )
            stem = output_path.stem
            edge_dir = output_path.parent
            write_image(edge_dir / f"{stem}_edges.png", edge_result.edges)
            write_image(edge_dir / f"{stem}_thinned.png", edge_result.thinned)

        if config.remove_bg:
            from anime_frame_qa.modules.background import remove_background

            bg_removed = remove_background(result)
            stem = output_path.stem
            bg_dir = output_path.parent
            write_image(bg_dir / f"{stem}_nobg.png", bg_removed)

        if config.inpaint and config.inpaint_mask_path:
            from anime_frame_qa.modules.inpaint import inpaint as run_inpaint

            mask = read_image(config.inpaint_mask_path)
            import cv2
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            result = run_inpaint(result, mask_gray)

        write_image(output_path, result)
