"""Color unevenness detection and inter-frame color consistency.

Uses HSV/Lab color space analysis for detecting color irregularities,
and histogram matching for maintaining color consistency across frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ColorAnalysis:
    unevenness_score: float
    unevenness_mask: np.ndarray


def detect_color_unevenness(
    image: np.ndarray,
    kernel_size: int = 31,
    threshold: float = 30.0,
) -> ColorAnalysis:
    """Detect color unevenness by comparing local vs global color distribution.

    Converts to Lab color space and compares each pixel's local neighborhood
    mean against the global mean. Regions with large deviation are flagged.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab).astype(np.float32)

    local_mean = cv2.blur(lab, (kernel_size, kernel_size))
    global_mean = cv2.mean(lab)[:3]
    global_arr = np.full_like(lab, global_mean)

    diff = np.sqrt(np.sum((local_mean - global_arr) ** 2, axis=2))
    mask = (diff > threshold).astype(np.uint8) * 255
    score = float(np.mean(diff))

    return ColorAnalysis(unevenness_score=score, unevenness_mask=mask)


def match_histograms(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match source frame's color distribution to a reference frame.

    Operates in Lab color space to preserve perceptual color relationships.
    """
    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2Lab)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2Lab)

    result = np.zeros_like(src_lab)
    for c in range(3):
        src_ch = src_lab[:, :, c]
        ref_ch = ref_lab[:, :, c]

        src_hist, _ = np.histogram(src_ch.flatten(), 256, (0, 256))
        ref_hist, _ = np.histogram(ref_ch.flatten(), 256, (0, 256))

        src_cdf = np.cumsum(src_hist).astype(np.float64)
        src_cdf /= src_cdf[-1]
        ref_cdf = np.cumsum(ref_hist).astype(np.float64)
        ref_cdf /= ref_cdf[-1]

        lut = np.zeros(256, dtype=np.uint8)
        for s_val in range(256):
            closest = np.argmin(np.abs(ref_cdf - src_cdf[s_val]))
            lut[s_val] = closest

        result[:, :, c] = lut[src_ch]

    return cv2.cvtColor(result, cv2.COLOR_Lab2BGR)


def enforce_color_consistency(
    frames: list[np.ndarray],
    window_size: int = 5,
) -> list[np.ndarray]:
    """Enforce color consistency across a sequence of frames.

    Uses a sliding window of recent frames to build a reference histogram,
    then matches each frame to it.
    """
    if len(frames) <= 1:
        return [f.copy() for f in frames]

    result = [frames[0].copy()]

    for i in range(1, len(frames)):
        start = max(0, i - window_size)
        window = frames[start:i]

        ref_lab_sum = np.zeros_like(window[0], dtype=np.float64)
        for wf in window:
            ref_lab_sum += cv2.cvtColor(wf, cv2.COLOR_BGR2Lab).astype(np.float64)
        ref_mean = (ref_lab_sum / len(window)).astype(np.uint8)
        reference = cv2.cvtColor(ref_mean, cv2.COLOR_Lab2BGR)

        matched = match_histograms(frames[i], reference)
        result.append(matched)

    return result
