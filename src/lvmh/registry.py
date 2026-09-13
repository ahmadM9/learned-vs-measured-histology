from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
KINDS = ("datasets", "segmenters", "models")


@dataclass
class ComponentSpec:
    kind: str
    name: str
    type: str
    params: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    def resolve(self):
        # type is "package.module:Attribute"; the module is imported only here,
        # so listing or validating configs never pulls in torch or a segmenter
        if ":" not in self.type:
            raise ValueError(
                f"{self.kind}/{self.name}: type must be 'module:attr', got {self.type!r}"
            )
        module_name, attr = self.type.split(":", 1)
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise ImportError(f"{self.kind}/{self.name}: cannot import {module_name}: {e}") from e
        if not hasattr(module, attr):
            raise AttributeError(f"{self.kind}/{self.name}: {module_name} has no {attr!r}")
        return getattr(module, attr)

    def build(self, **overrides):
        return self.resolve()(**{**self.params, **overrides})


class Registry:
    def __init__(self, kind: str, root: Path = CONFIG_ROOT):
        if kind not in KINDS:
            raise ValueError(f"unknown component kind {kind!r}; known: {KINDS}")
        self.kind = kind
        self.root = Path(root) / kind

    def names(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.yaml"))

    def spec(self, name: str) -> ComponentSpec:
        path = self.root / f"{name}.yaml"
        if not path.exists():
            raise KeyError(f"unknown {self.kind} {name!r}; registered: {self.names()}")
        raw = yaml.safe_load(path.read_text()) or {}
        if "type" not in raw:
            raise ValueError(f"{path} has no 'type' key")
        return ComponentSpec(
            kind=self.kind, name=name, type=raw["type"], params=raw.get("params") or {}, path=path
        )

    def build(self, name: str, **overrides):
        return self.spec(name).build(**overrides)


def registry(kind: str, root: Path = CONFIG_ROOT) -> Registry:
    return Registry(kind, root)
