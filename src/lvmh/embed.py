"""Extract one feature vector per spot with a registered encoder.

    python -m lvmh.embed --config configs/runs/synthetic_random_smoke.yaml

Writes one npz per section under out_dir/features with the feature matrix,
the barcodes in spot table order, the section and participant ids and the
GPU or CPU seconds spent, which is the cost column of the paper.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from lvmh.config import RunConfig, load_run
from lvmh.tiles import spot_tiles


def section_list(cfg: RunConfig, dataset) -> list[str]:
    subset = cfg.get("subset")
    if not subset:
        return dataset.section_ids()
    wanted = Path(subset).read_text().split()
    have = set(dataset.section_ids())
    missing = [w for w in wanted if w not in have]
    if missing:
        raise FileNotFoundError(f"subset {subset} names sections not in the dataset: {missing}")
    return wanted


def embed_section(section, encoder, batch_size: int) -> tuple[np.ndarray, float]:
    n = len(section.spots)
    feats = np.zeros((n, encoder.dim), dtype=np.float32)
    t0 = time.time()
    for idx, tiles in spot_tiles(section, encoder.tile_px, encoder.pixel_size_um, batch_size):
        feats[idx] = encoder.embed(tiles)
    return feats, time.time() - t0


def run(cfg: RunConfig) -> list[Path]:
    dataset = cfg.build("dataset")
    encoder = cfg.build("encoder")
    out = Path(cfg["out_dir"]) / "features"
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for sid in section_list(cfg, dataset):
        path = out / f"{sid}.npz"
        if path.exists() and not cfg.get("overwrite", False):
            print(f"{sid}: exists, skipped", flush=True)
            written.append(path)
            continue
        section = dataset.load(sid)
        feats, seconds = embed_section(section, encoder, cfg.get("batch_size", 256))
        np.savez(
            path,
            features=feats,
            barcodes=np.array(section.spots.barcode, dtype=str),
            section_id=sid,
            participant_id=section.participant_id,
            seconds=seconds,
            encoder=cfg["encoder"],
        )
        print(f"{sid}: {len(feats)} spots in {seconds:.1f} s", flush=True)
        written.append(path)
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    run(load_run(args.config))


if __name__ == "__main__":
    main()
