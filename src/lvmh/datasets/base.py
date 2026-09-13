from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
import scipy.sparse as sp

SPOT_COLUMNS = ("barcode", "x_px", "y_px", "in_tissue", "array_row", "array_col")


@runtime_checkable
class ImageReader(Protocol):
    shape: tuple[int, int]  # height, width in full resolution pixels
    pixel_size_um: float

    def read_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        # returns RGB uint8 (height, width, 3) at full resolution
        ...


@dataclass
class Section:
    section_id: str
    participant_id: str
    image: ImageReader
    spots: pd.DataFrame  # one row per spot, columns SPOT_COLUMNS, pixel coords at full res
    counts: sp.csr_matrix  # spots x genes, rows aligned with spots
    genes: list[str]
    spot_diameter_px: float

    def __post_init__(self):
        missing = [c for c in SPOT_COLUMNS if c not in self.spots.columns]
        if missing:
            raise ValueError(f"{self.section_id}: spot table missing columns {missing}")
        if self.counts.shape != (len(self.spots), len(self.genes)):
            raise ValueError(
                f"{self.section_id}: counts {self.counts.shape} do not match "
                f"{len(self.spots)} spots x {len(self.genes)} genes"
            )

    @property
    def pixel_size_um(self) -> float:
        return self.image.pixel_size_um


class ArrayImage:
    def __init__(self, array: np.ndarray, pixel_size_um: float):
        if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
            raise ValueError("ArrayImage needs an RGB uint8 array")
        self.array = array
        self.shape = (array.shape[0], array.shape[1])
        self.pixel_size_um = float(pixel_size_um)

    def read_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        return self.array[y : y + height, x : x + width]


class Dataset(Protocol):
    def section_ids(self) -> list[str]: ...

    def load(self, section_id: str) -> Section: ...
