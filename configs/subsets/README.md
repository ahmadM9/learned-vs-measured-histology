Committed participant lists. `kpmp_manifest.csv` is written by
`scripts/kpmp_manifest.py` from the KPMP repository search endpoint and every
list here is a seeded filter on it, written by `scripts/make_subsets.py`.
Participants with more than one Visium section (28-12265, cortex and medulla)
are excluded from every list because their image and spatial files cannot be
paired by name.
