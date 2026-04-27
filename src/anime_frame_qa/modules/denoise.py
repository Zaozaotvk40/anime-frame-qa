"""Noise suppression for AI-generated anime images.

Targets banding artifacts and mosquito noise common in AI-generated anime.
Provides bilateral filter and Non-Local Means denoising.
"""

from __future__ import annotations

from enum import Enum

import cv2
import numpy as np


class DenoiseMethod(str, Enum):
    BILATERAL = "bilateral"
    NLM = "nlm"


def denoise_bilateral(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """Apply bilateral filter — preserves edges while smoothing flat regions.

    Good for banding artifacts in anime (e.g., gradient areas in sky/skin).
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def denoise_nlm(
    image: np.ndarray,
    h: float = 10.0,
    template_window: int = 7,
    search_window: int = 21,
) -> np.ndarray:
    """Apply Non-Local Means denoising.

    Better for mosquito noise / high-frequency artifacts around edges.
    """
    return cv2.fastNlMeansDenoisingColored(
        image, None, h, h, template_window, search_window
    )


def denoise(
    image: np.ndarray,
    method: DenoiseMethod = DenoiseMethod.BILATERAL,
    **kwargs: object,
) -> np.ndarray:
    if method == DenoiseMethod.BILATERAL:
        return denoise_bilateral(image, **kwargs)  # type: ignore[arg-type]
    if method == DenoiseMethod.NLM:
        return denoise_nlm(image, **kwargs)  # type: ignore[arg-type]
    raise ValueError(f"Unknown method: {method}")
