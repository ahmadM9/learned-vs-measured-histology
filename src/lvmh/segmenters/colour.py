from __future__ import annotations

import numpy as np

from lvmh.segmenters.base import CLASS_IDS


class NearestColourSegmenter:
    # labels each pixel by the nearest reference colour; only for synthetic
    # sections and for testing the tiling loop, never for real tissue
    def __init__(
        self, colours: dict[str, tuple[int, int, int]], pixel_size_um: float, tile_px: int
    ):
        names = list(colours)
        self.ids = np.array([CLASS_IDS[n] for n in names], dtype=np.uint8)
        self.ref = np.array([colours[n] for n in names], dtype=np.float32)
        self.pixel_size_um = float(pixel_size_um)
        self.tile_px = int(tile_px)

    def segment_tile(self, tile: np.ndarray) -> np.ndarray:
        d = ((tile[:, :, None, :].astype(np.float32) - self.ref[None, None]) ** 2).sum(-1)
        return self.ids[d.argmin(-1)]
