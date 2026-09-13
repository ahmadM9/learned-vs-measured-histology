from __future__ import annotations

from typing import Protocol

import numpy as np

# project vocabulary; every segmenter maps its own classes onto these ids
CLASS_IDS = {
    "background": 0,
    "interstitium": 1,
    "tubule": 2,
    "glomerulus": 3,
    "tuft": 4,
    "artery": 5,
}


class Segmenter(Protocol):
    pixel_size_um: float  # the resolution the model expects its input at
    tile_px: int

    def segment_tile(self, tile: np.ndarray) -> np.ndarray:
        # RGB uint8 tile at pixel_size_um -> uint8 label map with CLASS_IDS values
        ...
