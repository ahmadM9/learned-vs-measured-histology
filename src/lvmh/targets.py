from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from lvmh.datasets.base import Section


def log_expression(counts: sp.csr_matrix, normalize_total: float | None = None) -> np.ndarray:
    x = counts.astype(np.float64)
    if normalize_total:
        totals = np.asarray(x.sum(axis=1)).ravel()
        totals[totals == 0] = 1.0
        x = sp.diags(normalize_total / totals) @ x
    return np.log1p(np.asarray(x.todense()))


def select_top_variance_genes(
    sections: list[Section], n_genes: int, normalize_total: float | None = None
) -> list[str]:
    # HEST-Bench fixes its gene list as the n most variable genes of the task in
    # log space, pooled over samples; the list is then frozen for every fold
    genes = sections[0].genes
    for s in sections[1:]:
        if s.genes != genes:
            raise ValueError(f"{s.section_id}: gene list differs from {sections[0].section_id}")
    x = np.vstack([log_expression(s.counts, normalize_total) for s in sections])
    order = np.argsort(x.var(axis=0))[::-1]
    return [genes[i] for i in order[:n_genes]]


def targets(section: Section, genes: list[str], normalize_total: float | None = None) -> np.ndarray:
    index = {g: i for i, g in enumerate(section.genes)}
    cols = [index[g] for g in genes]
    return log_expression(section.counts[:, cols], normalize_total)
