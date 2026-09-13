# learned-vs-measured-histology

A kidney slide can be turned into numbers by a pathology foundation model or
by segmenting glomeruli, tubules, vessels and interstitium and measuring them.
Which representation predicts spatial gene expression better, and how much
does the answer depend on the segmenter?

This repo compares the two on the [KPMP](https://www.kpmp.org/) Visium
dataset (non-neoplastic human kidney biopsies, H&E of the capture section
plus spot level expression):

- foundation model embeddings (UNI2-h, H-optimus-1), ridge regression per
  spot following the HEST-Bench protocol
- segmentation based morphometry per structure and per compartment,
  aggregated to spots, with a degradation curve and an oracle ceiling to
  separate the representation from the segmenter
- hybrids as secondary arms: embeddings pooled inside segmented structures,
  and graphs over structures or patches
- the HEST-Bench ccRCC task as the pre-registered case where morphometry is
  expected to fail, and TCGA-KIRC for outcome

Compute cost (GPU seconds, storage, re-extraction) is reported as a column,
not a footnote.

**Status: stage 0, feasibility. No results yet.**

## Data setup

KPMP open tier files download without an account. The heavy steps run on
Kaggle, where a kernel pulls the sections straight from the KPMP repository
into a private dataset. For local development only synthetic sections are
needed:

Data is never committed to this repo.

## Development

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest          # tests that need KPMP sections auto-skip if data/kpmp is empty
ruff check .
```

Every dataset, model and segmenter is a YAML under `configs/` whose `type`
names the adapter class. Runs under `configs/runs/` refer to them by name.
Adding a component is one YAML and, if new code is needed, one adapter file.

## Citations

KPMP data: "The results here are in whole or part based upon data generated
by the Kidney Precision Medicine Project. Accessed September 13, 2026.
https://www.kpmp.org."

> B. B. Lake et al. "An atlas of healthy and injured cell states and niches
> in the human kidney." Nature 619, 585-594 (2023).
> doi:10.1038/s41586-023-05769-3

> G. Jaume, P. Doucet, A. H. Song, et al. "HEST-1k: A Dataset for Spatial
> Transcriptomics and Histology Image Analysis." NeurIPS 2024.

## License

Apache-2.0, see [LICENSE](LICENSE). KPMP data is released under CC BY 4.0
and governed by the [KPMP data policies](https://www.kpmp.org/help-docs/study-overview).
