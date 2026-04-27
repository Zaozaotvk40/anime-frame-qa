from __future__ import annotations

from pathlib import Path
from typing import Generator

import cv2
import numpy as np


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise IOError(f"Failed to write image: {path}")


def read_video_frames(
    path: Path, chunk_size: int = 16
) -> Generator[list[np.ndarray], None, None]:
    """Yield chunks of frames from a video file."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    try:
        chunk: list[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            chunk.append(frame)
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk
    finally:
        cap.release()


def get_video_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    try:
        return {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fourcc": int(cap.get(cv2.CAP_PROP_FOURCC)),
        }
    finally:
        cap.release()


class VideoWriter:
    def __init__(self, path: Path, fps: float, width: int, height: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise IOError(f"Failed to open video writer: {path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *args: object) -> None:
        self.release()
