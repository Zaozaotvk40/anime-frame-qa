"""Image quality metrics for pipeline evaluation."""

from __future__ import annotations

import cv2
import numpy as np


def compute_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    return float(cv2.PSNR(original, processed))


def compute_ssim(original: np.ndarray, processed: np.ndarray) -> float:
    """Compute structural similarity (simplified, grayscale)."""
    gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray_proc = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY).astype(np.float64)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu_orig = cv2.GaussianBlur(gray_orig, (11, 11), 1.5)
    mu_proc = cv2.GaussianBlur(gray_proc, (11, 11), 1.5)

    mu_orig_sq = mu_orig ** 2
    mu_proc_sq = mu_proc ** 2
    mu_cross = mu_orig * mu_proc

    sigma_orig_sq = cv2.GaussianBlur(gray_orig ** 2, (11, 11), 1.5) - mu_orig_sq
    sigma_proc_sq = cv2.GaussianBlur(gray_proc ** 2, (11, 11), 1.5) - mu_proc_sq
    sigma_cross = cv2.GaussianBlur(gray_orig * gray_proc, (11, 11), 1.5) - mu_cross

    numerator = (2 * mu_cross + c1) * (2 * sigma_cross + c2)
    denominator = (mu_orig_sq + mu_proc_sq + c1) * (sigma_orig_sq + sigma_proc_sq + c2)

    ssim_map = numerator / denominator
    return float(np.mean(ssim_map))
