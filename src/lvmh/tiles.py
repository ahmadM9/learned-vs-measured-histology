from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from lvmh.datasets.base import Section
from lvmh.segmenters.tiling import resize_rgb


def spot_tiles(
    section: Section, tile_px: int, pixel_size_um: float, batch_size: int = 256
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    # one tile per spot, centred on the spot, covering tile_px at the encoder's
    # pixel size (224 px at 0.5 um is 112 um, twice the spot diameter, which is
    # what the kidney benchmark and HEST use)
    scale = pixel_size_um / section.pixel_size_um  # image px per encoder px
    side = int(round(tile_px * scale))
    h, w = section.image.shape
    idx, tiles = [], []
    for i, (x, y) in enumerate(zip(section.spots.x_px, section.spots.y_px, strict=True)):
        x0, y0 = int(round(x - side / 2)), int(round(y - side / 2))
        region = np.full((side, side, 3), 255, dtype=np.uint8)
        rx0, ry0 = max(x0, 0), max(y0, 0)
        rx1, ry1 = min(x0 + side, w), min(y0 + side, h)
        if rx1 > rx0 and ry1 > ry0:
            region[ry0 - y0 : ry1 - y0, rx0 - x0 : rx1 - x0] = section.image.read_region(
                rx0, ry0, rx1 - rx0, ry1 - ry0
            )
        tiles.append(resize_rgb(region, tile_px, tile_px))
        idx.append(i)
        if len(tiles) == batch_size:
            yield np.array(idx), np.stack(tiles)
            idx, tiles = [], []
    if tiles:
        yield np.array(idx), np.stack(tiles)
