from __future__ import annotations

import importlib

import numpy as np


def _resolve(value):
    # yaml cannot hold python objects, so create kwargs like mlp_layer are written
    # as "module:attr" strings, the same convention the registry uses
    if isinstance(value, str) and ":" in value and " " not in value:
        module, attr = value.split(":", 1)
        return getattr(importlib.import_module(module), attr)
    return value


class TimmHubEncoder:
    def __init__(
        self,
        hf_id: str,
        tile_px: int,
        pixel_size_um: float,
        dim: int,
        mean: list[float],
        std: list[float],
        create_kwargs: dict | None = None,
        batch_size: int = 64,
        device: str | None = None,
    ):
        self.hf_id = hf_id
        self.tile_px = int(tile_px)
        self.pixel_size_um = float(pixel_size_um)
        self.dim = int(dim)
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(1, 3, 1, 1)
        self.create_kwargs = {k: _resolve(v) for k, v in (create_kwargs or {}).items()}
        self.batch_size = int(batch_size)
        self.device = device
        self._model = None

    def _load(self):
        import timm
        import torch

        # the hub token is read from HF_TOKEN by huggingface_hub; never passed explicitly
        model = timm.create_model(f"hf-hub:{self.hf_id}", pretrained=True, **self.create_kwargs)
        model.eval()
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model.to(device)
        self.device = device

    def embed(self, tiles: np.ndarray) -> np.ndarray:
        import torch

        if self._model is None:
            self._load()
        if tiles.ndim != 4 or tiles.shape[1:] != (self.tile_px, self.tile_px, 3):
            raise ValueError(f"expected (n, {self.tile_px}, {self.tile_px}, 3), got {tiles.shape}")
        out = np.empty((len(tiles), self.dim), dtype=np.float32)
        with torch.inference_mode():
            for i in range(0, len(tiles), self.batch_size):
                x = tiles[i : i + self.batch_size].astype(np.float32) / 255.0
                x = (x.transpose(0, 3, 1, 2) - self.mean) / self.std
                feats = self._model(torch.from_numpy(np.ascontiguousarray(x)).to(self.device))
                out[i : i + len(x)] = feats.float().cpu().numpy()
        return out
