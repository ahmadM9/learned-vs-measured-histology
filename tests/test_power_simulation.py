import numpy as np
import pandas as pd
import pytest

from power_simulation import pilot_sd, simulate, summarise, tost_equivalent

CFG = {
    "n_patients": 100,
    "alpha": 0.05,
    "power_target": 0.8,
    "correlation_between_arms": [0.0, 0.8],
    "true_differences": [0.0],
    "margins": {"start": 0.01, "stop": 0.10, "step": 0.01},
    "n_sim": 400,
}


def test_tost_declares_equivalence_only_inside_margin():
    rng = np.random.default_rng(0)
    diffs = rng.normal(0.0, 0.05, size=(200, 100))
    assert tost_equivalent(diffs, margin=0.05, alpha=0.05).mean() > 0.95
    assert tost_equivalent(diffs, margin=0.002, alpha=0.05).mean() < 0.05
    shifted = diffs + 0.1
    assert tost_equivalent(shifted, margin=0.05, alpha=0.05).mean() == 0.0


def test_power_grows_with_margin_and_with_correlation():
    table = simulate(0.05, CFG, np.random.default_rng(1))
    for rho, g in table.groupby("correlation"):
        p = g.sort_values("margin").power.values
        assert np.all(np.diff(p) >= -0.02), rho
    s = summarise(table, 0.8).set_index("correlation").smallest_margin
    assert s[0.8] < s[0.0]


def test_pilot_sd_reads_per_patient_scores(tmp_path):
    df = pd.DataFrame(
        {"participant_id": ["a", "a", "b", "c"], "pearson_mean": [0.2, 0.4, 0.5, 0.6]}
    )
    df.to_csv(tmp_path / "s.csv", index=False)
    sd, n = pilot_sd(tmp_path / "s.csv")
    assert n == 3
    assert sd == pytest.approx(np.std([0.3, 0.5, 0.6], ddof=1))
