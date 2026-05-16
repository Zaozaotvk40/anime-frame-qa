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
from anime_frame_qa.modules.edge import process_edges, visualize_gaps


@dataclass
class PipelineConfig:
    deflicker: bool = False
    deflicker_alpha: float = 0.7
    deflicker_threshold: float = 0.3
    denoise_enabled: bool = False
    denoise_method: DenoiseMethod = DenoiseMethod.BILATERAL
    # bilateral params
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    # nlm params
    nlm_h: float = 10.0
    nlm_template_window: int = 7
    nlm_search_window: int = 21
    # banding params
    banding_blur_radius: int = 5
    banding_edge_low: int = 30
    banding_edge_high: int = 90
    extract_edges: bool = False
    color_consistency: bool = False
    color_window: int = 5
    remove_bg: bool = False
    inpaint: bool = False
    inpaint_mask_path: Path | None = None


def process_image(image: np.ndarray, config: PipelineConfig) -> np.ndarray:
    result = image
    if config.denoise_enabled:
        if config.denoise_method == DenoiseMethod.BILATERAL:
            result = denoise(
                result,
                method=config.denoise_method,
                d=config.bilateral_d,
                sigma_color=config.bilateral_sigma_color,
                sigma_space=config.bilateral_sigma_space,
            )
        elif config.denoise_method == DenoiseMethod.NLM:
            result = denoise(
                result,
                method=config.denoise_method,
                h=config.nlm_h,
                template_window=config.nlm_template_window,
                search_window=config.nlm_search_window,
            )
        elif config.denoise_method == DenoiseMethod.BANDING:
            result = denoise(
                result,
                method=config.denoise_method,
                blur_radius=config.banding_blur_radius,
                edge_low=config.banding_edge_low,
                edge_high=config.banding_edge_high,
            )
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


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def _process_single_image(
    input_path: Path, output_path: Path, config: PipelineConfig
) -> None:
    image = read_image(input_path)
    result = process_image(image, config)

    if config.extract_edges:
        edge_result = process_edges(image)
        vis = visualize_gaps(edge_result.thinned, edge_result.gaps)
        print(f"  edges: {edge_result.gap_count} gap(s) detected")
        write_image(output_path, vis)
        return

    if config.remove_bg:
        from anime_frame_qa.modules.background import remove_background

        bg_removed = remove_background(result)
        write_image(output_path, bg_removed)
        return

    if config.inpaint:
        if config.inpaint_mask_path is None:
            raise ValueError("--inpaint-mask が必要です: マスク画像のパスを指定してください")
        from anime_frame_qa.modules.inpaint import inpaint as run_inpaint
        import cv2
        mask_gray = cv2.cvtColor(read_image(config.inpaint_mask_path), cv2.COLOR_BGR2GRAY)
        result = run_inpaint(result, mask_gray)

    write_image(output_path, result)


def run(input_path: Path, output_path: Path, config: PipelineConfig) -> None:
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    if input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        for img_path in sorted(input_path.iterdir()):
            if img_path.suffix.lower() in _IMAGE_EXTS:
                _process_single_image(img_path, output_path / img_path.name, config)
        return

    if input_path.suffix.lower() in video_exts:
        process_video(input_path, output_path, config)
    else:
        _process_single_image(input_path, output_path, config)
