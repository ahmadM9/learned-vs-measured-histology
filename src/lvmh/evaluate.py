"""Score spot level features against expression targets with the HEST-Bench recipe.

    python -m lvmh.evaluate --config configs/runs/synthetic_random_smoke.yaml

Reads the per section feature files written by lvmh.embed, builds the targets,
runs patient level folds with scaler, PCA and ridge, and writes per fold, per
gene and per patient Pearson tables plus an MLflow run under out_dir.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lvmh.config import RunConfig, load_run
from lvmh.targets import select_top_variance_genes, targets


def fit_predict(x_train, y_train, x_test, n_components: int, alpha_numerator: float, seed: int):
    # HEST-Bench: StandardScaler, PCA(256), Ridge(alpha=100/(n_features*n_genes), lsqr,
    # no intercept); alpha therefore depends on the PCA size and the gene count
    n_components = min(n_components, x_train.shape[0], x_train.shape[1])
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("pca", PCA(n_components=n_components, random_state=seed))]
    )
    z_train = pipe.fit_transform(x_train)
    z_test = pipe.transform(x_test)
    alpha = alpha_numerator / (z_train.shape[1] * y_train.shape[1])
    reg = Ridge(solver="lsqr", alpha=alpha, random_state=seed, fit_intercept=False, max_iter=1000)
    reg.fit(z_train, y_train)
    return reg.predict(z_test)


def pearson_per_gene(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    out = np.zeros(y_true.shape[1])
    for j in range(y_true.shape[1]):
        if y_true[:, j].std() == 0 or y_pred[:, j].std() == 0:
            out[j] = 0.0  # HEST scores a constant gene as zero rather than nan
        else:
            out[j] = pearsonr(y_true[:, j], y_pred[:, j])[0]
    return out


def evaluate(
    features: dict[str, np.ndarray],
    y: dict[str, np.ndarray],
    participant_of: dict[str, str],
    genes: list[str],
    n_folds: int,
    n_components: int,
    alpha_numerator: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    ids = sorted(features)
    x_all = np.vstack([features[s] for s in ids])
    y_all = np.vstack([y[s] for s in ids])
    section_of_row = np.concatenate([[s] * len(features[s]) for s in ids])
    groups = np.array([participant_of[s] for s in section_of_row])
    n_folds = min(n_folds, len(set(groups)))

    fold_rows, gene_rows, patient_rows = [], [], []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_folds).split(x_all, y_all, groups)):
        t0 = time.time()
        pred = fit_predict(x_all[tr], y_all[tr], x_all[te], n_components, alpha_numerator, seed)
        r = pearson_per_gene(y_all[te], pred)
        fold_rows.append({"fold": k, "n_train": len(tr), "n_test": len(te),
                          "pearson_mean": float(r.mean()), "seconds": time.time() - t0})
        for g, v in zip(genes, r, strict=True):
            gene_rows.append({"fold": k, "gene": g, "pearson": float(v)})
        for p in sorted(set(groups[te])):
            m = groups[te] == p
            rp = pearson_per_gene(y_all[te][m], pred[m])
            patient_rows.append({"fold": k, "participant_id": p, "n_spots": int(m.sum()),
                                 "pearson_mean": float(rp.mean())})
    return {
        "folds": pd.DataFrame(fold_rows),
        "genes": pd.DataFrame(gene_rows),
        "patients": pd.DataFrame(patient_rows),
    }


def load_features(feature_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    features, participant_of = {}, {}
    for f in sorted(feature_dir.glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        sid = str(z["section_id"])
        features[sid] = z["features"]
        participant_of[sid] = str(z["participant_id"])
    if not features:
        raise FileNotFoundError(f"no feature files under {feature_dir}")
    return features, participant_of


def run(cfg: RunConfig) -> dict[str, pd.DataFrame]:
    out_dir = Path(cfg["out_dir"])
    feature_dir = Path(cfg.get("feature_dir") or out_dir / "features")
    features, participant_of = load_features(feature_dir)
    dataset = cfg.build("dataset")
    sections = {sid: dataset.load(sid) for sid in features}
    ev = cfg["evaluation"]
    norm = ev.get("normalize_total")
    genes = select_top_variance_genes(list(sections.values()), ev["n_genes"], norm)
    y = {sid: targets(s, genes, norm) for sid, s in sections.items()}
    for sid in features:
        if len(features[sid]) != len(y[sid]):
            raise ValueError(f"{sid}: {len(features[sid])} feature rows, {len(y[sid])} spots")
    tables = evaluate(features, y, participant_of, genes, ev["n_folds"], ev["n_components"],
                      ev["alpha_numerator"], cfg.get("seed", 0))
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out_dir / f"scores_{name}.csv", index=False)
    (out_dir / "genes.json").write_text(json.dumps(genes))
    log_mlflow(cfg, tables, genes)
    return tables


def log_mlflow(cfg: RunConfig, tables: dict[str, pd.DataFrame], genes: list[str]) -> None:
    import mlflow

    out_dir = Path(cfg["out_dir"])
    # mlflow 3.6 refuses the file store unless an env var is set; sqlite is the
    # supported local backend now
    mlflow.set_tracking_uri(f"sqlite:///{out_dir.resolve() / 'mlflow.db'}")
    mlflow.set_experiment(cfg.get("experiment", cfg.path.stem))
    with mlflow.start_run(run_name=cfg.path.stem):
        mlflow.log_params({k: v for k, v in cfg["evaluation"].items()})
        mlflow.log_params({k: cfg[k] for k in ("dataset", "encoder") if k in cfg.raw})
        mlflow.log_param("n_genes_selected", len(genes))
        for row in tables["folds"].itertuples():
            mlflow.log_metric("pearson_mean", row.pearson_mean, step=row.fold)
        mlflow.log_metric("pearson_mean_over_folds", float(tables["folds"].pearson_mean.mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    tables = run(load_run(args.config))
    print(tables["folds"].to_string(index=False))
    print(f"pearson over folds: {tables['folds'].pearson_mean.mean():.4f}")


if __name__ == "__main__":
    main()
