from __future__ import annotations

from typing import Protocol

import numpy as np


class Encoder(Protocol):
    pixel_size_um: float
    tile_px: int
    dim: int

    def embed(self, tiles: np.ndarray) -> np.ndarray:
        # (n, tile_px, tile_px, 3) uint8 -> (n, dim) float32
        ...
