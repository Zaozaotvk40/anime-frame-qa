"""Flicker detection and suppression for consecutive video frames.

Uses histogram comparison to detect brightness/color flicker between frames,
and temporal smoothing (exponential moving average) to suppress it.
"""

from __future__ import annotations

import cv2
import numpy as np


def compute_histogram(frame: np.ndarray, bins: int = 256) -> np.ndarray:
    """Compute a normalized grayscale histogram for a frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def detect_flicker(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    threshold: float = 0.3,
) -> tuple[bool, float]:
    """Detect flicker between two consecutive frames.

    Returns (is_flickering, distance) where distance is the Bhattacharyya
    distance between histograms.
    """
    hist_prev = compute_histogram(prev_frame)
    hist_curr = compute_histogram(curr_frame)
    distance = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_BHATTACHARYYA)
    return distance > threshold, float(distance)


def compute_brightness_ratio(
    prev_frame: np.ndarray, curr_frame: np.ndarray
) -> np.ndarray:
    """Compute per-channel mean brightness ratio between frames."""
    mean_prev = cv2.mean(prev_frame)[:3]
    mean_curr = cv2.mean(curr_frame)[:3]
    ratios = []
    for mp, mc in zip(mean_prev, mean_curr):
        if mc > 1e-6:
            ratios.append(mp / mc)
        else:
            ratios.append(1.0)
    return np.array(ratios, dtype=np.float64)


def suppress_flicker_ema(
    frames: list[np.ndarray],
    alpha: float = 0.7,
    threshold: float = 0.3,
) -> list[np.ndarray]:
    """Suppress flicker in a sequence of frames using exponential moving average.

    Only applies correction when flicker is detected above threshold.
    The first frame is returned as-is and used as the brightness anchor.
    """
    if not frames:
        return []

    result = [frames[0].copy()]
    ema_mean = np.array(cv2.mean(frames[0])[:3], dtype=np.float64)

    for i in range(1, len(frames)):
        flickering, _ = detect_flicker(frames[i - 1], frames[i], threshold)

        curr_mean = np.array(cv2.mean(frames[i])[:3], dtype=np.float64)
        ema_mean = alpha * ema_mean + (1 - alpha) * curr_mean

        if flickering:
            correction = np.ones(3, dtype=np.float64)
            for c in range(3):
                if curr_mean[c] > 1e-6:
                    correction[c] = ema_mean[c] / curr_mean[c]

            corrected = frames[i].astype(np.float64)
            for c in range(3):
                corrected[:, :, c] *= correction[c]
            corrected = np.clip(corrected, 0, 255).astype(np.uint8)
            result.append(corrected)
        else:
            result.append(frames[i].copy())

    return result
