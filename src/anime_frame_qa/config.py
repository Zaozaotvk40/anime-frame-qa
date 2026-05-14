"""YAML configuration loading for pipeline parameters."""

from __future__ import annotations

from pathlib import Path

import yaml

from anime_frame_qa.modules.denoise import DenoiseMethod
from anime_frame_qa.pipeline import PipelineConfig


def load_config(path: Path) -> PipelineConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(raw)}")

    denoise_method = raw.get("denoise_method", "bilateral")
    if isinstance(denoise_method, str):
        denoise_method = DenoiseMethod(denoise_method)

    return PipelineConfig(
        deflicker=raw.get("deflicker", False),
        deflicker_alpha=float(raw.get("deflicker_alpha", 0.7)),
        deflicker_threshold=float(raw.get("deflicker_threshold", 0.3)),
        denoise_enabled=raw.get("denoise", False),
        denoise_method=denoise_method,
        extract_edges=raw.get("extract_edges", False),
        edge_connect_gaps=raw.get("edge_connect_gaps", False),
        color_consistency=raw.get("color_consistency", False),
        color_window=int(raw.get("color_window", 5)),
        remove_bg=raw.get("remove_bg", False),
        inpaint=raw.get("inpaint", False),
    )
