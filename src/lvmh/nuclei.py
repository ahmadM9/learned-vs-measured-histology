from __future__ import annotations

import time
from typing import Protocol

import numpy as np
from scipy import ndimage
from skimage.measure import regionprops

from lvmh.datasets.base import Section

FEATURE_NAMES = [
    "n_nuclei",
    "density_per_1000um2",
    "area_um2_mean",
    "area_um2_sd",
    "eccentricity_mean",
    "solidity_mean",
    "nn_distance_um_mean",
]


class NucleiModel(Protocol):
    def segment(self, images: list[np.ndarray], diameter_px: float) -> list[np.ndarray]:
        # RGB uint8 crops -> int32 instance label maps, 0 is background
        ...


class ThresholdNuclei:
    # dark blob detector for synthetic sections and tests, never for real tissue
    def __init__(self, threshold: int = 60, min_area_px: int = 4):
        self.threshold = int(threshold)
        self.min_area_px = int(min_area_px)

    def segment(self, images: list[np.ndarray], diameter_px: float) -> list[np.ndarray]:
        out = []
        for img in images:
            dark = img.mean(axis=-1) < self.threshold
            lab, n = ndimage.label(dark)
            if n:
                areas = ndimage.sum_labels(np.ones_like(lab), lab, np.arange(1, n + 1))
                small = np.flatnonzero(areas < self.min_area_px) + 1
                lab[np.isin(lab, small)] = 0
            out.append(lab.astype(np.int32))
        return out


class CellposeNuclei:
    def __init__(
        self,
        pretrained_model: str = "cpsam_v2",
        batch_size: int = 32,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        gpu: bool | None = None,
    ):
        self.pretrained_model = pretrained_model
        self.batch_size = int(batch_size)
        self.flow_threshold = float(flow_threshold)
        self.cellprob_threshold = float(cellprob_threshold)
        self.gpu = gpu
        self._model = None

    def _load(self):
        from cellpose import models

        gpu = self.gpu
        if gpu is None:
            import torch

            gpu = torch.cuda.is_available()
        # weights download from the cellpose server on first use, no token needed
        self._model = models.CellposeModel(gpu=gpu, pretrained_model=self.pretrained_model)

    def segment(self, images: list[np.ndarray], diameter_px: float) -> list[np.ndarray]:
        if self._model is None:
            self._load()
        masks, _, _ = self._model.eval(
            list(images),
            diameter=diameter_px,
            batch_size=self.batch_size,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
        )
        return [np.asarray(m, dtype=np.int32) for m in masks]


def nuclear_features(label: np.ndarray, disc: np.ndarray, pixel_size_um: float) -> np.ndarray:
    # nuclei whose centroid lies inside the spot disc; sizes in physical units so
    # the two image batches (0.32 and 0.64 um per pixel) give comparable numbers
    props = [p for p in regionprops(label) if disc[int(p.centroid[0]), int(p.centroid[1])]]
    disc_area_um2 = disc.sum() * pixel_size_um**2
    n = len(props)
    if n == 0:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    areas = np.array([p.area for p in props]) * pixel_size_um**2
    ecc = np.array([p.eccentricity for p in props])
    sol = np.array([p.solidity for p in props])
    if n > 1:
        c = np.array([p.centroid for p in props]) * pixel_size_um
        d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        nn = float(d.min(axis=1).mean())
    else:
        nn = 0.0
    return np.array(
        [n, 1000.0 * n / disc_area_um2, areas.mean(), areas.std(), ecc.mean(), sol.mean(), nn],
        dtype=np.float32,
    )


class NucleiMorphometry:
    def __init__(
        self,
        model: dict,
        nucleus_diameter_um: float = 7.0,
        window_factor: float = 1.3,
        batch_size: int = 32,
    ):
        # model is a component spec dict {type, params} so one featurizer yaml can
        # name any nuclei model without a second registry lookup
        from lvmh.registry import ComponentSpec

        self.model = ComponentSpec(kind="nuclei", name="inline", **model).build()
        self.nucleus_diameter_um = float(nucleus_diameter_um)
        self.window_factor = float(window_factor)
        self.batch_size = int(batch_size)
        self.names = list(FEATURE_NAMES)

    @property
    def dim(self) -> int:
        return len(self.names)

    def featurize(self, section: Section) -> tuple[np.ndarray, float]:
        t0 = time.time()
        px = section.pixel_size_um
        radius = section.spot_diameter_px / 2
        half = int(np.ceil(radius * self.window_factor))
        side = 2 * half + 1
        yy, xx = np.ogrid[:side, :side]
        disc = (xx - half) ** 2 + (yy - half) ** 2 <= radius * radius
        h, w = section.image.shape
        feats = np.zeros((len(section.spots), self.dim), dtype=np.float32)
        crops, idx = [], []

        def flush():
            labels = self.model.segment(crops, self.nucleus_diameter_um / px)
            for i, lab in zip(idx, labels, strict=True):
                feats[i] = nuclear_features(lab, disc, px)
            crops.clear()
            idx.clear()

        for i, (x, y) in enumerate(zip(section.spots.x_px, section.spots.y_px, strict=True)):
            x0, y0 = int(round(x)) - half, int(round(y)) - half
            crop = np.full((side, side, 3), 255, dtype=np.uint8)
            rx0, ry0, rx1, ry1 = max(x0, 0), max(y0, 0), min(x0 + side, w), min(y0 + side, h)
            if rx1 > rx0 and ry1 > ry0:
                crop[ry0 - y0 : ry1 - y0, rx0 - x0 : rx1 - x0] = section.image.read_region(
                    rx0, ry0, rx1 - rx0, ry1 - ry0
                )
            crops.append(crop)
            idx.append(i)
            if len(crops) == self.batch_size:
                flush()
        if crops:
            flush()
        return feats, time.time() - t0
