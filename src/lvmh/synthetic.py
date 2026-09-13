from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from lvmh.datasets.base import ArrayImage, Section
from lvmh.segmenters.base import CLASS_IDS

# colours are only meant to be told apart by a threshold, not to look real
COLOURS = {
    "background": (245, 245, 245),
    "interstitium": (235, 200, 215),
    "tubule": (200, 120, 160),
    "glomerulus": (150, 70, 130),
    "tuft": (110, 40, 100),
    "artery": (180, 90, 90),
}


@dataclass
class SyntheticSection:
    section: Section
    mask: np.ndarray  # uint8 label map with CLASS_IDS values, same shape as the image
    structures: pd.DataFrame  # one row per drawn structure: id, class, cx, cy, rx, ry


def _draw_ellipse(mask, cx, cy, rx, ry, value):
    h, w = mask.shape
    y, x = np.ogrid[:h, :w]
    inside = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0
    mask[inside] = value


def make_section(
    seed: int = 0,
    size_px: int = 1024,
    pixel_size_um: float = 1.0,
    n_glomeruli: int = 3,
    n_tubules: int = 20,
    spot_pitch_um: float = 100.0,
    spot_diameter_um: float = 55.0,
    n_genes: int = 30,
    section_id: str = "synthetic-0",
    participant_id: str = "synthetic",
) -> SyntheticSection:
    rng = np.random.default_rng(seed)
    mask = np.full((size_px, size_px), CLASS_IDS["background"], dtype=np.uint8)

    # a tissue blob in the middle, the rest stays background like a needle biopsy
    tissue_r = size_px * 0.42
    centre = size_px / 2
    _draw_ellipse(mask, centre, centre, tissue_r, tissue_r * 0.8, CLASS_IDS["interstitium"])

    rows = []
    sid = 0
    glom_r_px = 150.0 / pixel_size_um / 2  # a human glomerulus is roughly 150 to 200 um across
    placed: list[tuple[float, float]] = []
    while len(placed) < n_glomeruli:
        cx = rng.uniform(size_px * 0.25, size_px * 0.75)
        cy = rng.uniform(size_px * 0.3, size_px * 0.7)
        # glomeruli never touch, so each one is one connected object in the mask
        if any(np.hypot(cx - px, cy - py) < 2.3 * glom_r_px for px, py in placed):
            continue
        placed.append((cx, cy))
        _draw_ellipse(mask, cx, cy, glom_r_px, glom_r_px, CLASS_IDS["glomerulus"])
        _draw_ellipse(mask, cx, cy, glom_r_px * 0.7, glom_r_px * 0.7, CLASS_IDS["tuft"])
        rows.append((sid, "glomerulus", cx, cy, glom_r_px, glom_r_px))
        sid += 1
    tub_r_px = 40.0 / pixel_size_um / 2  # a tubule cross section is roughly 30 to 60 um
    for _ in range(n_tubules):
        cx = rng.uniform(size_px * 0.15, size_px * 0.85)
        cy = rng.uniform(size_px * 0.25, size_px * 0.75)
        if mask[int(cy), int(cx)] != CLASS_IDS["interstitium"]:
            continue
        rx, ry = tub_r_px * rng.uniform(0.8, 1.2), tub_r_px * rng.uniform(0.8, 1.2)
        _draw_ellipse(mask, cx, cy, rx, ry, CLASS_IDS["tubule"])
        rows.append((sid, "tubule", cx, cy, rx, ry))
        sid += 1
    structures = pd.DataFrame(rows, columns=["id", "class", "cx", "cy", "rx", "ry"])

    image = np.zeros((size_px, size_px, 3), dtype=np.uint8)
    for name, value in CLASS_IDS.items():
        image[mask == value] = COLOURS[name]
    image = np.clip(image.astype(int) + rng.integers(-6, 7, image.shape), 0, 255).astype(np.uint8)

    pitch = spot_pitch_um / pixel_size_um
    diameter_px = spot_diameter_um / pixel_size_um
    xs = np.arange(pitch, size_px - pitch, pitch)
    ys = np.arange(pitch, size_px - pitch, pitch * np.sqrt(3) / 2)
    spot_rows = []
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            x_off = x + (pitch / 2 if r % 2 else 0.0)
            if x_off >= size_px - pitch / 2:
                continue
            in_tissue = int(mask[int(y), int(x_off)] != CLASS_IDS["background"])
            spot_rows.append((f"SPOT{len(spot_rows):05d}-1", x_off, y, in_tissue, r, c))
    spots = pd.DataFrame(
        spot_rows, columns=["barcode", "x_px", "y_px", "in_tissue", "array_row", "array_col"]
    )

    # planted signal: gene j responds to class (j mod n_classes) under the spot,
    # so a correct spot-to-structure mapping is measurably better than a wrong one
    n_classes = len(CLASS_IDS)
    fractions = np.zeros((len(spots), n_classes))
    half = int(diameter_px / 2)
    for i, (x, y) in enumerate(zip(spots.x_px, spots.y_px, strict=True)):
        x0, y0 = max(int(x) - half, 0), max(int(y) - half, 0)
        patch = mask[y0 : int(y) + half, x0 : int(x) + half]
        fractions[i] = np.bincount(patch.ravel(), minlength=n_classes) / max(patch.size, 1)
    genes = [f"GENE{j:03d}" for j in range(n_genes)]
    mean = 2.0 + 20.0 * fractions[:, np.arange(n_genes) % n_classes]
    counts = sp.csr_matrix(rng.poisson(mean).astype(np.float32))

    section = Section(
        section_id=section_id,
        participant_id=participant_id,
        image=ArrayImage(image, pixel_size_um),
        spots=spots,
        counts=counts,
        genes=genes,
        spot_diameter_px=diameter_px,
    )
    return SyntheticSection(section=section, mask=mask, structures=structures)


class SyntheticDataset:
    def __init__(self, n_sections: int = 2, seed: int = 0, **section_kwargs):
        self.n_sections = n_sections
        self.seed = seed
        self.section_kwargs = section_kwargs

    def section_ids(self) -> list[str]:
        return [f"synthetic-{i}" for i in range(self.n_sections)]

    def load(self, section_id: str) -> Section:
        i = int(section_id.rsplit("-", 1)[1])
        return make_section(
            seed=self.seed + i,
            section_id=section_id,
            participant_id=f"synthetic-p{i}",
            **self.section_kwargs,
        ).section


def write_kpmp_layout(syn: SyntheticSection, section_dir, spot_diameter_um: float = 55.0):
    # mirrors what kaggle/pull_sections leaves on disk, so the kpmp adapter is
    # tested on the exact file layout it will meet
    import json

    import h5py
    import tifffile

    section_dir = Path(section_dir)
    spatial = section_dir / "outs" / "spatial"
    spatial.mkdir(parents=True)
    sec = syn.section
    tifffile.imwrite(section_dir / f"{sec.section_id}.tif", sec.image.array, photometric="rgb")

    scale = {
        "tissue_hires_scalef": 0.1,
        "tissue_lowres_scalef": 0.03,
        "fiducial_diameter_fullres": sec.spot_diameter_px * 1.6,
        "spot_diameter_fullres": sec.spot_diameter_px,
    }
    (spatial / "scalefactors_json.json").write_text(json.dumps(scale))
    pos = sec.spots.rename(columns={"y_px": "pxl_row_in_fullres", "x_px": "pxl_col_in_fullres"})
    cols = ["barcode", "in_tissue", "array_row", "array_col"]
    cols += ["pxl_row_in_fullres", "pxl_col_in_fullres"]
    pos[cols].to_csv(spatial / "tissue_positions.csv", index=False)

    # filtered matrix holds in-tissue spots only, genes x barcodes, csc like cell ranger
    keep = sec.spots.in_tissue.values.astype(bool)
    m = sp.csc_matrix(sec.counts[keep].T)
    with h5py.File(section_dir / "outs" / "filtered_feature_bc_matrix.h5", "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=m.data.astype(np.int32))
        g.create_dataset("indices", data=m.indices.astype(np.int64))
        g.create_dataset("indptr", data=m.indptr.astype(np.int64))
        g.create_dataset("shape", data=np.array(m.shape, dtype=np.int32))
        g.create_dataset("barcodes", data=np.array(sec.spots.barcode[keep], dtype="S"))
        feat = g.create_group("features")
        feat.create_dataset("name", data=np.array(sec.genes, dtype="S"))
        ids = [f"ENSG{j:011d}" for j in range(len(sec.genes))]
        feat.create_dataset("id", data=np.array(ids, dtype="S"))
    return section_dir
