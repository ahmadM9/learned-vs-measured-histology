"""Simulate TOST equivalence power for the full cohort from pilot per patient scores.

    python scripts/power_simulation.py --config configs/power_simulation.yaml

Writes power_table.csv and power_summary.csv next to the pilot scores and prints
the smallest margin that reaches the target power for every assumed correlation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats


def tost_equivalent(diffs: np.ndarray, margin: float, alpha: float) -> np.ndarray:
    # diffs: (n_sim, n_patients) paired per patient differences; equivalence is declared
    # when both one-sided t tests reject, i.e. the 1 - 2 alpha interval lies inside the margin
    n = diffs.shape[1]
    mean = diffs.mean(axis=1)
    se = diffs.std(axis=1, ddof=1) / np.sqrt(n)
    t_low = (mean + margin) / se
    t_high = (mean - margin) / se
    crit = stats.t.ppf(1 - alpha, n - 1)
    return (t_low > crit) & (t_high < -crit)


def simulate(score_sd: float, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    m = cfg["margins"]
    margins = np.arange(m["start"], m["stop"] + m["step"] / 2, m["step"])
    rows = []
    for rho in cfg["correlation_between_arms"]:
        sd_diff = score_sd * np.sqrt(2 * (1 - rho))
        for d in cfg["true_differences"]:
            diffs = rng.normal(d, sd_diff, size=(cfg["n_sim"], cfg["n_patients"]))
            for margin in margins:
                power = tost_equivalent(diffs, margin, cfg["alpha"]).mean()
                rows.append({"correlation": rho, "true_difference": d, "margin": float(margin),
                             "sd_diff": sd_diff, "power": float(power)})
    return pd.DataFrame(rows)


def summarise(table: pd.DataFrame, power_target: float) -> pd.DataFrame:
    rows = []
    for (rho, d), g in table.groupby(["correlation", "true_difference"]):
        ok = g[g.power >= power_target]
        rows.append({"correlation": rho, "true_difference": d,
                     "smallest_margin": float(ok.margin.min()) if len(ok) else np.nan})
    return pd.DataFrame(rows)


def pilot_sd(path: Path) -> tuple[float, int]:
    scores = pd.read_csv(path)
    per_patient = scores.groupby("participant_id").pearson_mean.mean()
    return float(per_patient.std(ddof=1)), len(per_patient)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    sd, n_pilot = pilot_sd(Path(cfg["pilot_scores"]))
    rng = np.random.default_rng(cfg["seed"])
    table = simulate(sd, cfg, rng)
    summary = summarise(table, cfg["power_target"])
    out = Path(cfg["pilot_scores"]).parent
    table.to_csv(out / "power_table.csv", index=False)
    summary.to_csv(out / "power_summary.csv", index=False)
    print(f"pilot: {n_pilot} patients, per patient score sd {sd:.4f}")
    print(f"full cohort n = {cfg['n_patients']}, target power {cfg['power_target']}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
