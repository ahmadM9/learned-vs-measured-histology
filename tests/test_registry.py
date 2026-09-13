from pathlib import Path

import pytest
import yaml

from lvmh.config import load_run, validate_all
from lvmh.registry import CONFIG_ROOT, Registry


def test_every_committed_config_resolves():
    found = validate_all(CONFIG_ROOT)
    assert "synthetic" in found["datasets"]


def test_unknown_name_lists_known_names():
    with pytest.raises(KeyError, match="registered"):
        Registry("datasets").spec("does-not-exist")


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        Registry("widgets")


def test_bad_type_string_fails_at_resolve(tmp_path: Path):
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "broken.yaml").write_text("type: lvmh.synthetic.SyntheticDataset\n")
    with pytest.raises(ValueError, match="module:attr"):
        Registry("datasets", tmp_path).spec("broken").resolve()


def test_missing_attribute_fails_at_resolve(tmp_path: Path):
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "broken.yaml").write_text("type: lvmh.synthetic:Nope\n")
    with pytest.raises(AttributeError, match="Nope"):
        Registry("datasets", tmp_path).spec("broken").resolve()


def test_build_passes_params_and_overrides(tmp_path: Path):
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "syn.yaml").write_text(
        yaml.safe_dump({"type": "lvmh.synthetic:SyntheticDataset", "params": {"n_sections": 3}})
    )
    ds = Registry("datasets", tmp_path).build("syn", seed=7)
    assert ds.section_ids() == ["synthetic-0", "synthetic-1", "synthetic-2"]
    assert ds.seed == 7


def test_run_config_fails_before_running_if_component_missing(tmp_path: Path):
    (tmp_path / "datasets").mkdir()
    run = tmp_path / "run.yaml"
    run.write_text("dataset: nope\n")
    with pytest.raises(KeyError, match="nope"):
        load_run(run, tmp_path)


def test_run_config_builds_named_component():
    cfg = load_run(CONFIG_ROOT / "runs" / "synthetic_smoke.yaml")
    ds = cfg.build("dataset")
    assert len(ds.section_ids()) == 2
    assert cfg["out_dir"].startswith("outputs/")
    with pytest.raises(KeyError):
        cfg.build("segmenter")
