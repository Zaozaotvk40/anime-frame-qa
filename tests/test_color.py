from __future__ import annotations

import numpy as np

from anime_frame_qa.modules.color import (
    detect_color_unevenness,
    enforce_color_consistency,
    match_histograms,
)


def test_detect_color_unevenness() -> None:
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    result = detect_color_unevenness(img)
    assert result.unevenness_mask.shape == (100, 100)
    assert result.unevenness_score >= 0


def test_detect_color_unevenness_with_patch() -> None:
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    img[0:50, 0:50] = [255, 0, 0]
    result = detect_color_unevenness(img)
    assert result.unevenness_score > 0


def test_match_histograms() -> None:
    source = np.full((100, 100, 3), 50, dtype=np.uint8)
    reference = np.full((100, 100, 3), 200, dtype=np.uint8)
    matched = match_histograms(source, reference)
    assert matched.shape == source.shape
    assert matched.dtype == np.uint8


def test_enforce_color_consistency() -> None:
    frames = [
        np.full((100, 100, 3), b, dtype=np.uint8)
        for b in [100, 200, 100, 200, 100]
    ]
    result = enforce_color_consistency(frames, window_size=3)
    assert len(result) == 5
    assert all(f.shape == (100, 100, 3) for f in result)
