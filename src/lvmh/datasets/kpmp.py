from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import tifffile
import zarr

from lvmh.datasets.base import SPOT_COLUMNS, Section

# space ranger 1.x wrote tissue_positions_list.csv without a header, 2.x writes
# tissue_positions.csv with one; both carry the same six columns in this order
POSITIONS_COLUMNS = ["barcode", "in_tissue", "array_row", "array_col", "y_px", "x_px"]


class TiffRegionReader:
    def __init__(self, path: Path, pixel_size_um: float):
        self.path = Path(path)
        self.pixel_size_um = float(pixel_size_um)
        self._tif = tifffile.TiffFile(self.path)
        # kpmp images are uncompressed and contiguous, so a memmap slices the file
        # directly; anything else (compressed, tiled, pyramidal) goes through zarr,
        # which reads whole strips or tiles per region
        try:
            self._z = tifffile.memmap(self.path, mode="r")
            self.backend = "memmap"
        except ValueError:
            self._z = zarr.open(self._tif.aszarr(), mode="r")
            if isinstance(self._z, zarr.Group):
                self._z = self._z[0]  # pyramidal tif: level 0 is full resolution
            self.backend = "zarr"
        if self._z.ndim != 3 or self._z.shape[2] < 3:
            raise ValueError(f"{self.path.name}: expected an RGB image, got shape {self._z.shape}")
        self.shape = (int(self._z.shape[0]), int(self._z.shape[1]))

    def read_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        region = np.asarray(self._z[y : y + height, x : x + width, :3])
        if region.dtype != np.uint8:
            region = np.clip(region, 0, 255).astype(np.uint8)
        return region

    def close(self):
        self._tif.close()


def read_positions(spatial_dir: Path) -> pd.DataFrame:
    new = spatial_dir / "tissue_positions.csv"
    old = spatial_dir / "tissue_positions_list.csv"
    if new.exists():
        df = pd.read_csv(new)
        df = df.rename(columns={"pxl_row_in_fullres": "y_px", "pxl_col_in_fullres": "x_px"})
    elif old.exists():
        df = pd.read_csv(old, header=None, names=POSITIONS_COLUMNS)
    else:
        raise FileNotFoundError(f"no tissue positions file in {spatial_dir}")
    return df[list(SPOT_COLUMNS)]


def read_10x_h5(path: Path) -> tuple[sp.csr_matrix, list[str], list[str]]:
    with h5py.File(path, "r") as f:
        g = f["matrix"]
        shape = tuple(int(v) for v in g["shape"][()])  # genes x barcodes, column major
        m = sp.csc_matrix((g["data"][()], g["indices"][()], g["indptr"][()]), shape=shape)
        barcodes = [b.decode() for b in g["barcodes"][()]]
        genes = [n.decode() for n in g["features"]["name"][()]]
    return sp.csr_matrix(m.T, dtype=np.float32), barcodes, genes


class KPMPVisium:
    def __init__(
        self,
        root: str | Path,
        spot_diameter_um: float,
        image_glob: str = "*.tif",
        spatial_glob: str = "**/spatial",
        h5_glob: str = "**/filtered_feature_bc_matrix.h5",
    ):
        self.root = Path(root)
        self.spot_diameter_um = float(spot_diameter_um)
        self.image_glob = image_glob
        self.spatial_glob = spatial_glob
        self.h5_glob = h5_glob

    def section_dirs(self) -> dict[str, Path]:
        # one directory per section, named by participant id, as the pull kernel writes them
        return {p.name: p for p in sorted(self.root.iterdir()) if p.is_dir()}

    def section_ids(self) -> list[str]:
        return list(self.section_dirs())

    def _one(self, section_dir: Path, pattern: str, what: str) -> Path:
        hits = sorted(section_dir.glob(pattern))
        if len(hits) != 1:
            raise FileNotFoundError(
                f"{section_dir.name}: expected one {what} matching {pattern!r}, found {hits}"
            )
        return hits[0]

    def load(self, section_id: str) -> Section:
        d = self.section_dirs()[section_id]
        spatial = self._one(d, self.spatial_glob, "spatial dir")
        scale = json.loads((spatial / "scalefactors_json.json").read_text())
        spot_px = float(scale["spot_diameter_fullres"])
        # the only resolution record in a visium bundle is the spot diameter in
        # full res pixels; the physical diameter comes from the dataset config
        pixel_size_um = self.spot_diameter_um / spot_px

        counts, barcodes, genes = read_10x_h5(self._one(d, self.h5_glob, "h5 matrix"))
        positions = read_positions(spatial).set_index("barcode")
        missing = [b for b in barcodes if b not in positions.index]
        if missing:
            raise ValueError(f"{section_id}: {len(missing)} matrix barcodes have no position")
        spots = positions.loc[barcodes].reset_index()

        image = TiffRegionReader(self._one(d, self.image_glob, "image"), pixel_size_um)
        return Section(
            section_id=section_id,
            participant_id=section_id,
            image=image,
            spots=spots,
            counts=counts,
            genes=genes,
            spot_diameter_px=spot_px,
        )
