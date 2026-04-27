from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def tmp_output(tmp_path: Path) -> Path:
    return tmp_path / "output"


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    path = tmp_path / "sample.png"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """Create a short test video with brightness variation to trigger flicker detection."""
    path = tmp_path / "sample.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 24.0, (640, 480))

    for i in range(30):
        brightness = 128 + 80 * (i % 2)  # alternate bright/dark for flicker
        frame = np.full((480, 640, 3), brightness, dtype=np.uint8)
        noise = np.random.randint(-20, 20, (480, 640, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)

    writer.release()
    return path
