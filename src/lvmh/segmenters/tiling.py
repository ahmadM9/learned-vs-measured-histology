from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from lvmh.datasets.base import ImageReader
from lvmh.segmenters.base import CLASS_IDS, Segmenter


@dataclass
class LabelMask:
    array: np.ndarray  # uint8, CLASS_IDS values, at pixel_size_um
    pixel_size_um: float

    def to_mask_px(self, x_px: np.ndarray, image_pixel_size_um: float) -> np.ndarray:
        return x_px * (image_pixel_size_um / self.pixel_size_um)


def resize_rgb(tile: np.ndarray, width: int, height: int) -> np.ndarray:
    if tile.shape[1] == width and tile.shape[0] == height:
        return tile
    return np.asarray(Image.fromarray(tile).resize((width, height), Image.BILINEAR))


def segment_image(
    image: ImageReader,
    segmenter: Segmenter,
    overlap_px: int = 64,
    background_value: int = 220,
    background_fraction: float = 0.995,
) -> LabelMask:
    # the whole image is processed at the segmenter's resolution; tiles overlap
    # and only the centre of each prediction is kept, so tile borders leave no seam
    scale = image.pixel_size_um / segmenter.pixel_size_um  # model px per image px
    out_h = int(round(image.shape[0] * scale))
    out_w = int(round(image.shape[1] * scale))
    out = np.full((out_h, out_w), CLASS_IDS["background"], dtype=np.uint8)
    tile = segmenter.tile_px
    step = tile - 2 * overlap_px
    if step <= 0:
        raise ValueError(f"tile {tile} too small for overlap {overlap_px}")

    for y0 in range(0, out_h, step):
        for x0 in range(0, out_w, step):
            # tile window in model pixels, padded by the overlap on every side
            ty0, tx0 = max(y0 - overlap_px, 0), max(x0 - overlap_px, 0)
            ty1, tx1 = min(ty0 + tile, out_h), min(tx0 + tile, out_w)
            ty0, tx0 = max(ty1 - tile, 0), max(tx1 - tile, 0)
            # the same window in image pixels
            iy0, ix0 = int(ty0 / scale), int(tx0 / scale)
            iy1, ix1 = int(np.ceil(ty1 / scale)), int(np.ceil(tx1 / scale))
            region = image.read_region(ix0, iy0, ix1 - ix0, iy1 - iy0)
            model_in = resize_rgb(region, tx1 - tx0, ty1 - ty0)
            if model_in.shape[0] < tile or model_in.shape[1] < tile:
                pad = np.full((tile, tile, 3), 255, dtype=np.uint8)
                pad[: model_in.shape[0], : model_in.shape[1]] = model_in
                model_in = pad
            # a tile is skipped only when almost every pixel is bright; a mean
            # threshold would drop tissue edges sitting in mostly empty tiles
            bright = (model_in.min(axis=-1) > background_value).mean()
            if bright > background_fraction:
                pred = np.full((tile, tile), CLASS_IDS["background"], dtype=np.uint8)
            else:
                pred = segmenter.segment_tile(model_in)
            # keep the centre of the prediction only
            ky0, kx0 = y0, x0
            ky1, kx1 = min(y0 + step, out_h), min(x0 + step, out_w)
            out[ky0:ky1, kx0:kx1] = pred[ky0 - ty0 : ky1 - ty0, kx0 - tx0 : kx1 - tx0]
    return LabelMask(out, segmenter.pixel_size_um)
