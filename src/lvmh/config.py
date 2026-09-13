from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from lvmh.registry import CONFIG_ROOT, KINDS, ComponentSpec, Registry

# run yaml keys that name a component, mapped to the registry kind they live in
COMPONENT_KEYS = {"dataset": "datasets", "segmenter": "segmenters", "encoder": "models"}


@dataclass
class RunConfig:
    path: Path
    raw: dict[str, Any]
    components: dict[str, ComponentSpec] = field(default_factory=dict)

    def __getitem__(self, key: str):
        return self.raw[key]

    def get(self, key: str, default=None):
        return self.raw.get(key, default)

    def build(self, key: str, **overrides):
        if key not in self.components:
            raise KeyError(f"{self.path.name} names no {key!r}; has {sorted(self.components)}")
        # "<key>_overrides" in the run yaml changes constructor params for this run only,
        # e.g. the data root on a kaggle machine, without touching the component yaml
        merged = {**(self.raw.get(f"{key}_overrides") or {}), **overrides}
        return self.components[key].build(**merged)


def load_run(path: str | Path, config_root: Path = CONFIG_ROOT) -> RunConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    components = {}
    for key, kind in COMPONENT_KEYS.items():
        if key in raw:
            # fail here, before any data is touched, if a config names something
            # that does not exist
            components[key] = Registry(kind, config_root).spec(raw[key])
    return RunConfig(path=path, raw=raw, components=components)


def validate_all(config_root: Path = CONFIG_ROOT) -> dict[str, list[str]]:
    found = {}
    for kind in KINDS:
        reg = Registry(kind, config_root)
        for name in reg.names():
            reg.spec(name).resolve()
        found[kind] = reg.names()
    for run in sorted((config_root / "runs").glob("*.yaml")):
        load_run(run, config_root)
    return found
