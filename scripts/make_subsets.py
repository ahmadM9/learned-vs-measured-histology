"""Draw the committed participant lists from the KPMP manifest.

    python scripts/make_subsets.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SUBSETS = Path(__file__).resolve().parents[1] / "configs" / "subsets"


def eligible(manifest: pd.DataFrame, exclude_multi_section: bool) -> pd.DataFrame:
    img = manifest[manifest.file_kind == "image"]
    if exclude_multi_section:
        n = img.groupby("participant_id").size()
        img = img[img.participant_id.isin(n[n == 1].index)]
    return img[["participant_id", "image_size_class", "enrollment_category"]].reset_index(drop=True)


def draw(pool: pd.DataFrame, n: int, spread_over: str, rng: np.random.Generator) -> list[str]:
    # round robin over the spread column so every category is represented before any repeats
    groups = {
        k: list(rng.permutation(g.participant_id.values)) for k, g in pool.groupby(spread_over)
    }
    order = list(rng.permutation(sorted(groups)))
    picked: list[str] = []
    while len(picked) < n and any(groups.values()):
        for k in order:
            if groups[k] and len(picked) < n:
                picked.append(groups[k].pop())
    if len(picked) < n:
        raise ValueError(f"only {len(picked)} eligible participants, wanted {n}")
    return sorted(picked)


def make_lists(manifest: pd.DataFrame, spec: dict) -> dict[str, list[str]]:
    rng = np.random.default_rng(spec["seed"])
    pool = eligible(manifest, spec["exclude_multi_section"])
    out = {}
    for name, rule in spec["lists"].items():
        ids: list[str] = []
        for size_class in ("large", "small"):
            n = rule.get(size_class, 0)
            if n:
                sub = pool[pool.image_size_class == size_class]
                ids += draw(sub, n, rule["spread_over"], rng)
        out[name] = ids
    return out


def main() -> None:
    manifest = pd.read_csv(SUBSETS / "kpmp_manifest.csv")
    spec = yaml.safe_load((SUBSETS / "subsets.yaml").read_text())
    for name, ids in make_lists(manifest, spec).items():
        (SUBSETS / f"{name}.txt").write_text("\n".join(ids) + "\n")
        print(name, len(ids), ids)


if __name__ == "__main__":
    main()
