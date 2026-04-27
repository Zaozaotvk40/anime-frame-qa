"""Edge/contour extraction, line cleanup, and gap detection for anime frames.

Canny edge detection + morphological processing, thinning, and
contour gap detection/connection for anime line art quality inspection.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class EdgeResult:
    edges: np.ndarray
    thinned: np.ndarray
    gaps: list[tuple[int, int]]
    gap_count: int


def extract_edges(
    image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.Canny(blurred, low_threshold, high_threshold)


def thin_edges(edges: np.ndarray) -> np.ndarray:
    """Morphological thinning to get single-pixel-width lines (Zhang-Suen)."""
    return cv2.ximgproc.thinning(edges, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)


def detect_gaps(
    thinned: np.ndarray, min_gap: int = 3, max_gap: int = 15
) -> list[tuple[int, int]]:
    """Detect gaps (breaks) in thinned contour lines.

    Finds endpoint pixels, then checks if another endpoint is nearby (within
    max_gap distance) — these are likely unintended line breaks.
    """
    kernel = np.array(
        [[1, 1, 1],
         [1, 0, 1],
         [1, 1, 1]], dtype=np.uint8
    )
    neighbor_count = cv2.filter2D(
        (thinned > 0).astype(np.uint8), -1, kernel
    )
    endpoints = np.argwhere((thinned > 0) & (neighbor_count == 1))

    gaps = []
    used = set()
    for i, (y1, x1) in enumerate(endpoints):
        if i in used:
            continue
        for j, (y2, x2) in enumerate(endpoints[i + 1 :], start=i + 1):
            if j in used:
                continue
            dist = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            if min_gap <= dist <= max_gap:
                gaps.append((int(y1 * thinned.shape[1] + x1),
                             int(y2 * thinned.shape[1] + x2)))
                used.add(i)
                used.add(j)
                break

    return gaps


def connect_gaps(
    thinned: np.ndarray, gaps: list[tuple[int, int]]
) -> np.ndarray:
    """Draw lines between detected gap endpoints to connect broken contours."""
    result = thinned.copy()
    _, w = result.shape
    for flat1, flat2 in gaps:
        y1, x1 = divmod(flat1, w)
        y2, x2 = divmod(flat2, w)
        cv2.line(result, (x1, y1), (x2, y2), 255, 1)
    return result


def process_edges(
    image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
    connect: bool = True,
    min_gap: int = 3,
    max_gap: int = 15,
) -> EdgeResult:
    edges = extract_edges(image, low_threshold, high_threshold)
    thinned = thin_edges(edges)
    gaps = detect_gaps(thinned, min_gap, max_gap)

    if connect and gaps:
        thinned = connect_gaps(thinned, gaps)

    return EdgeResult(
        edges=edges,
        thinned=thinned,
        gaps=gaps,
        gap_count=len(gaps),
    )
