"""Noise suppression for AI-generated anime images.

Provides three methods targeting different artifact types:
  bilateral : Gaussian noise reduction while preserving edges
  nlm       : Mosquito noise (high-frequency artifacts around edges)
  banding   : Quantization step artifacts in gradient areas (sky, skin)
"""

from __future__ import annotations

from enum import Enum

import cv2
import numpy as np


class DenoiseMethod(str, Enum):
    BILATERAL = "bilateral"
    NLM = "nlm"
    BANDING = "banding"


def denoise_bilateral(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """Apply bilateral filter for Gaussian noise reduction.

    Preserves edges while smoothing flat regions. Suitable for general
    Gaussian-like noise. NOT effective for banding artifacts (step edges
    are treated as valid edges and preserved).
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def denoise_nlm(
    image: np.ndarray,
    h: float = 10.0,
    template_window: int = 7,
    search_window: int = 21,
) -> np.ndarray:
    """Apply Non-Local Means denoising.

    Effective for mosquito noise (high-frequency ringing artifacts around
    edges caused by JPEG compression or AI generation artifacts).
    """
    return cv2.fastNlMeansDenoisingColored(
        image, None, h, h, template_window, search_window
    )


def denoise_banding(
    image: np.ndarray,
    blur_radius: int = 5,
    edge_low: int = 30,
    edge_high: int = 90,
) -> np.ndarray:
    """Remove banding artifacts in gradient areas.

    Detects edges (line art, contours) and applies Gaussian blur only to
    flat/gradient regions. This preserves sharp edges while smoothing the
    quantization steps that cause banding in areas like sky and skin.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), edge_low, edge_high)

    # dilate to protect pixels adjacent to edges
    kernel = np.ones((3, 3), np.uint8)
    edge_mask = cv2.dilate(edges, kernel, iterations=1)

    flat = (edge_mask == 0).astype(np.float32)[:, :, np.newaxis]

    ksize = blur_radius * 2 + 1
    blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
    return (blurred * flat + image * (1.0 - flat)).astype(np.uint8)


def denoise(
    image: np.ndarray,
    method: DenoiseMethod = DenoiseMethod.BILATERAL,
    **kwargs: object,
) -> np.ndarray:
    if method == DenoiseMethod.BILATERAL:
        return denoise_bilateral(image, **kwargs)  # type: ignore[arg-type]
    if method == DenoiseMethod.NLM:
        return denoise_nlm(image, **kwargs)  # type: ignore[arg-type]
    if method == DenoiseMethod.BANDING:
        return denoise_banding(image, **kwargs)  # type: ignore[arg-type]
    raise ValueError(f"Unknown method: {method}")