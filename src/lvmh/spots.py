from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage

from lvmh.segmenters.base import CLASS_IDS
from lvmh.segmenters.tiling import LabelMask

CLASS_NAMES = {v: k for k, v in CLASS_IDS.items()}
# classes that form countable structures; interstitium and background are compartments
STRUCTURE_CLASSES = ("glomerulus", "tubule", "artery")


@dataclass
class Structures:
    instances: np.ndarray  # int32 instance id per pixel, 0 where no structure
    table: pd.DataFrame  # structure_id, class, area_px, cx, cy


def label_structures(
    mask: LabelMask, structure_classes=STRUCTURE_CLASSES, min_area_px: int = 0
) -> Structures:
    # tuft pixels count as part of their glomerulus, so a glomerulus is one object
    arr = mask.array
    merged = arr.copy()
    merged[arr == CLASS_IDS["tuft"]] = CLASS_IDS["glomerulus"]
    instances = np.zeros(arr.shape, dtype=np.int32)
    rows = []
    next_id = 1
    for name in structure_classes:
        lab, n = ndimage.label(merged == CLASS_IDS[name])
        if n == 0:
            continue
        ids = np.arange(1, n + 1)
        areas = ndimage.sum_labels(np.ones_like(lab), lab, ids)
        centres = ndimage.center_of_mass(np.ones_like(lab), lab, ids)
        keep = areas >= min_area_px
        remap = np.zeros(n + 1, dtype=np.int32)
        for i, (k, area, (cy, cx)) in enumerate(zip(ids, areas, centres, strict=True)):
            if not keep[i]:
                continue
            remap[k] = next_id
            rows.append((next_id, name, int(area), float(cx), float(cy)))
            next_id += 1
        instances = np.where(lab > 0, remap[lab], instances)
    table = pd.DataFrame(rows, columns=["structure_id", "class", "area_px", "cx", "cy"])
    return Structures(instances=instances, table=table)


def disc(radius_px: float) -> np.ndarray:
    r = int(np.ceil(radius_px))
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return (x * x + y * y) <= radius_px * radius_px


def assign_spots(
    spots: pd.DataFrame,
    spot_diameter_px: float,
    image_pixel_size_um: float,
    mask: LabelMask,
    structures: Structures,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # FUSION rule (Border et al. 2025): a spot contributes to a structure with weight
    # equal to the fraction of the structure's area the spot covers. Computed by
    # pixel counting on the label mask rather than polygons, same quantity.
    # Every spot also gets the class fractions under its disc, so spots on
    # interstitium are kept with a row instead of being dropped.
    scale = image_pixel_size_um / mask.pixel_size_um
    radius = spot_diameter_px * scale / 2
    d = disc(radius)
    r = d.shape[0] // 2
    disc_area = int(d.sum())
    h, w = mask.array.shape
    area_of = dict(zip(structures.table.structure_id, structures.table.area_px, strict=True))
    n_classes = len(CLASS_IDS)

    long_rows, class_rows = [], []
    for barcode, x, y in zip(spots.barcode, spots.x_px, spots.y_px, strict=True):
        cx, cy = int(round(x * scale)), int(round(y * scale))
        y0, y1, x0, x1 = cy - r, cy + r + 1, cx - r, cx + r + 1
        dy0, dx0 = max(0, -y0), max(0, -x0)
        dy1, dx1 = d.shape[0] - max(0, y1 - h), d.shape[1] - max(0, x1 - w)
        sub = d[dy0:dy1, dx0:dx1]
        yy0, yy1, xx0, xx1 = max(y0, 0), min(y1, h), max(x0, 0), min(x1, w)
        if sub.size == 0 or yy1 <= yy0 or xx1 <= xx0:
            class_rows.append((barcode, *([0.0] * n_classes), 0))
            continue
        classes = mask.array[yy0:yy1, xx0:xx1][sub]
        counts = np.bincount(classes, minlength=n_classes)[:n_classes]
        class_rows.append((barcode, *(counts / disc_area), int(sub.sum())))
        inst = structures.instances[yy0:yy1, xx0:xx1][sub]
        for sid in np.unique(inst[inst > 0]):
            overlap = int((inst == sid).sum())
            long_rows.append(
                (barcode, int(sid), overlap, overlap / area_of[int(sid)], overlap / disc_area)
            )

    spot_structure = pd.DataFrame(
        long_rows, columns=["barcode", "structure_id", "overlap_px", "weight", "spot_fraction"]
    )
    spot_classes = pd.DataFrame(
        class_rows,
        columns=["barcode", *(f"frac_{CLASS_NAMES[i]}" for i in range(n_classes)), "disc_px"],
    )
    return spot_structure, spot_classes
