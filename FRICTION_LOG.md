# Friction log

Notes on anything that was harder than it should have been while building this
project: unclear docs, surprising defaults, bugs in dependencies. Kept so that
fixable upstream issues turn into reports instead of being forgotten.

Format: date, what happened, where, whether it is worth reporting upstream.

---

**2026-09-13, MLflow 3.16 refuses the file store.** `mlflow.set_tracking_uri("file:...")`
raises "The filesystem tracking backend is in maintenance mode" unless
`MLFLOW_ALLOW_FILE_STORE=true` is set. Switched to `sqlite:///<out_dir>/mlflow.db`,
which the repository standard already allows. Documented upstream, not a bug.

**2026-09-13, tifffile 2026.9 needs zarr 3.** `TiffFile.aszarr()` imports `zarr.abc`
and fails with "zarr 2.18.7 < 3 is not supported". Pinned `zarr>=3.0`. Both
libraries document it, but the error only appears at first region read.

**2026-09-13, KPMP spatial bundles have four layouts.** `outs/spatial`,
`<slide>/outs/spatial`, `spatial/` at the top, and for 2021 sections a full
Space Ranger 1.x tree under `nfs/corenfs/.../outs` with the header-less
`tissue_positions_list.csv`. Found by unpacking five bundles on Kaggle. Handled
by glob and by supporting both positions file formats. Worth telling KPMP.
