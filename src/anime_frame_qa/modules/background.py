"""Background removal using RMBG-2.0 (via rembg).

Requires optional dependency: pip install anime-frame-qa[cnn]
"""

from __future__ import annotations

import numpy as np


def remove_background(image: np.ndarray) -> np.ndarray:
    """Remove background, returning BGRA image with alpha channel."""
    try:
        from rembg import remove
    except ImportError:
        raise ImportError(
            "rembg is required for background removal. "
            "Install with: uv pip install anime-frame-qa[cnn]"
        )

    from rembg import new_session
    from PIL import Image

    session = new_session("isnet-anime")
    pil_img = Image.fromarray(image[:, :, ::-1])  # BGR -> RGB
    result = remove(pil_img, session=session)
    return np.array(result)[:, :, [2, 1, 0, 3]]  # RGBA -> BGRA
