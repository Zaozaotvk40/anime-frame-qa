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


def test_process_directory(tmp_path: Path, tmp_output: Path) -> None:
    import numpy as np

    input_dir = tmp_path / "batch"
    input_dir.mkdir()
    for i in range(3):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(input_dir / f"frame_{i:03d}.png"), img)

    output_dir = tmp_output / "batch_out"
    config = PipelineConfig(denoise_enabled=True)
    run(input_dir, output_dir, config)

    assert (output_dir / "frame_000.png").exists()
    assert (output_dir / "frame_001.png").exists()
    assert (output_dir / "frame_002.png").exists()
