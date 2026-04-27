from __future__ import annotations

from pathlib import Path

import cv2

from anime_frame_qa.pipeline import PipelineConfig, run


def test_process_image(sample_image: Path, tmp_output: Path) -> None:
    output = tmp_output / "result.png"
    config = PipelineConfig(denoise_enabled=True)
    run(sample_image, output, config)
    assert output.exists()
    img = cv2.imread(str(output))
    assert img is not None
    assert img.shape == (480, 640, 3)


def test_process_video_denoise(sample_video: Path, tmp_output: Path) -> None:
    output = tmp_output / "result.mp4"
    config = PipelineConfig(denoise_enabled=True)
    run(sample_video, output, config)
    assert output.exists()

    cap = cv2.VideoCapture(str(output))
    assert cap.isOpened()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert frame_count == 30


def test_process_video_deflicker(sample_video: Path, tmp_output: Path) -> None:
    output = tmp_output / "result.mp4"
    config = PipelineConfig(deflicker=True, deflicker_threshold=0.1)
    run(sample_video, output, config)
    assert output.exists()

    cap = cv2.VideoCapture(str(output))
    assert cap.isOpened()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert frame_count == 30


def test_process_video_all(sample_video: Path, tmp_output: Path) -> None:
    output = tmp_output / "result.mp4"
    config = PipelineConfig(deflicker=True, denoise_enabled=True)
    run(sample_video, output, config)
    assert output.exists()
