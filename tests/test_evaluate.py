import numpy as np
import pytest

from lvmh.config import load_run
from lvmh.embed import run as embed_run
from lvmh.evaluate import evaluate, pearson_per_gene
from lvmh.evaluate import run as evaluate_run
from lvmh.registry import CONFIG_ROOT


def make_problem(seed, n_patients=6, spots=40, dim=30, genes=8, signal=True):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((dim, genes))
    features, y, participant_of = {}, {}, {}
    for p in range(n_patients):
        sid = f"s{p}"
        x = rng.standard_normal((spots, dim))
        noise = rng.standard_normal((spots, genes)) * 0.3
        y[sid] = (x @ w + noise) if signal else rng.standard_normal((spots, genes))
        features[sid] = x
        participant_of[sid] = f"p{p}"
    return features, y, participant_of, [f"g{j}" for j in range(genes)]


def test_planted_linear_signal_is_recovered():
    f, y, p, genes = make_problem(0)
    t = evaluate(f, y, p, genes, n_folds=3, n_components=30, alpha_numerator=100.0, seed=0)
    assert t["folds"].pearson_mean.mean() > 0.9
    assert len(t["genes"]) == 3 * len(genes)
    assert set(t["patients"].participant_id) == set(p.values())
    assert (t["patients"].pearson_mean > 0.8).all()


def test_no_signal_scores_near_zero():
    f, y, p, genes = make_problem(1, signal=False)
    t = evaluate(f, y, p, genes, n_folds=3, n_components=30, alpha_numerator=100.0, seed=0)
    assert abs(t["folds"].pearson_mean.mean()) < 0.15


def test_folds_are_patient_level():
    f, y, p, genes = make_problem(2, n_patients=4)
    t = evaluate(f, y, p, genes, n_folds=4, n_components=10, alpha_numerator=100.0, seed=0)
    per_fold = t["patients"].groupby("fold").participant_id.nunique()
    assert (per_fold == 1).all()


def test_constant_gene_scores_zero_not_nan():
    y = np.zeros((10, 2))
    y[:, 1] = np.arange(10)
    r = pearson_per_gene(y, y + 1e-9)
    assert r[0] == 0.0 and r[1] == pytest.approx(1.0)


def test_smoke_run_end_to_end(tmp_path):
    cfg = load_run(CONFIG_ROOT / "runs" / "synthetic_random_smoke.yaml")
    cfg.raw["out_dir"] = str(tmp_path / "run")
    paths = embed_run(cfg)
    assert len(paths) == 2 and all(p.exists() for p in paths)
    tables = evaluate_run(cfg)
    assert tables["folds"].pearson_mean.mean() > 0.3  # colour classes leak into pixels
    assert (tmp_path / "run" / "scores_patients.csv").exists()
    assert (tmp_path / "run" / "mlflow.db").exists()
