"""Inpainting using LaMa (via simple-lama-inpainting).

Requires optional dependency: pip install anime-frame-qa[cnn]
"""

from __future__ import annotations

import numpy as np


def inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Inpaint masked regions using LaMa.

    Args:
        image: BGR image (H, W, 3)
        mask: Binary mask (H, W), 255 = regions to inpaint
    """
    try:
        from simple_lama_inpainting import SimpleLama
    except ImportError:
        raise ImportError(
            "simple-lama-inpainting is required for inpainting. "
            "Install with: uv pip install anime-frame-qa[cnn]"
        )

    from PIL import Image

    pil_img = Image.fromarray(image[:, :, ::-1])  # BGR -> RGB
    pil_mask = Image.fromarray(mask).convert("L")

    lama = SimpleLama()
    result = lama(pil_img, pil_mask)
    return np.array(result)[:, :, ::-1]  # RGB -> BGR
