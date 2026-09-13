from __future__ import annotations

import numpy as np


class RandomProjectionEncoder:
    # fixed random projection of the flattened tile; the random feature baseline
    # from the plan, and the stand-in encoder for tests and smoke runs
    def __init__(self, tile_px: int, pixel_size_um: float, dim: int, seed: int = 0):
        self.tile_px = int(tile_px)
        self.pixel_size_um = float(pixel_size_um)
        self.dim = int(dim)
        rng = np.random.default_rng(seed)
        n_in = self.tile_px * self.tile_px * 3
        self.w = rng.standard_normal((n_in, self.dim)).astype(np.float32) / np.sqrt(n_in)

    def embed(self, tiles: np.ndarray) -> np.ndarray:
        x = tiles.reshape(len(tiles), -1).astype(np.float32) / 255.0
        return x @ self.w
