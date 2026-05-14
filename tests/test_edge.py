from __future__ import annotations

import numpy as np

from anime_frame_qa.modules.edge import extract_edges, process_edges, thin_edges


def test_extract_edges() -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2_line_args = ((10, 50), (90, 50))
    import cv2
    cv2.line(img, *cv2_line_args, (255, 255, 255), 2)

    edges = extract_edges(img, low_threshold=50, high_threshold=80)
    assert edges.shape == (100, 100)
    assert edges.dtype == np.uint8
    assert np.any(edges > 0)


def test_thin_edges() -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    import cv2
    cv2.rectangle(img, (20, 20), (80, 80), (255, 255, 255), 3)

    edges = extract_edges(img, low_threshold=50, high_threshold=80)
    thinned = thin_edges(edges)
    assert thinned.shape == edges.shape
    assert np.sum(thinned > 0) <= np.sum(edges > 0)


def test_process_edges_returns_result() -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    import cv2
    cv2.line(img, (10, 50), (40, 50), (255, 255, 255), 2)
    cv2.line(img, (50, 50), (90, 50), (255, 255, 255), 2)

    result = process_edges(img)
    assert result.edges.shape == (100, 100)
    assert result.thinned.shape == (100, 100)
    assert result.gap_count >= 0
