"""Kaggle kernel that pulls KPMP Visium sections straight from the KPMP repository.

Submitted with the Kaggle CLI from the repo root:

    kaggle kernels push -p kaggle/pull_sections/

No GPU. The kernel clones the public repo, reads the committed manifest and a
subset list, downloads each participant's image, spatial bundle, matrix and
metadata into /kaggle/working/<participant_id>/, unpacks the spatial bundle,
and prints the TIFF tags of every image. The output is then saved by hand as
a private Kaggle dataset, so the images never touch a laptop. If a previous
run's output is attached as an input, files already present are not fetched
again.
"""

import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

REPO_URL = "https://github.com/ahmadM9/learned-vs-measured-histology.git"
REPO_DIR = Path("/tmp/repo")
WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")

# which committed lists under configs/subsets to pull (union); edit before pushing
SUBSET_NAMES = ("segmenter_check_5", "pilot_10")
# file kinds to fetch; cloupe files are 200 to 700 MB each and never used
FILE_KINDS = ("image", "spatial", "matrix", "metadata")


def run(cmd, **kwargs):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kwargs)


def find_previous(participant: str, file_name: str) -> Path | None:
    for prev in INPUT.glob(f"**/{participant}/{file_name}"):
        return prev
    return None


def download(url: str, dest: Path, expected_size: int, attempts: int = 3) -> None:
    for attempt in range(1, attempts + 1):
        try:
            t0 = time.time()
            with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f, length=16 * 1024 * 1024)
            size = dest.stat().st_size
            if size != expected_size:
                raise OSError(f"size mismatch: got {size}, manifest says {expected_size}")
            print(f"  {dest.name}: {size / 1e6:.0f} MB in {time.time() - t0:.0f} s", flush=True)
            return
        except Exception as e:  # retry on anything, the endpoint drops connections
            print(f"  attempt {attempt} failed: {e}", flush=True)
            if attempt == attempts:
                raise
            time.sleep(30)


def unpack(path: Path, into: Path) -> None:
    if path.suffixes[-2:] == [".tar", ".gz"] or path.suffix == ".tgz":
        with tarfile.open(path) as t:
            t.extractall(into, filter="data")
    elif path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            z.extractall(into)


def describe_tif(path: Path) -> dict:
    import tifffile

    with tifffile.TiffFile(path) as t:
        p = t.pages[0]
        tags = {tag.name: tag.value for tag in p.tags.values()}
        res = tags.get("XResolution"), tags.get("ResolutionUnit")
        return {
            "shape": p.shape,
            "dtype": str(p.dtype),
            "compression": str(p.compression),
            "tiled": p.is_tiled,
            "levels": len(t.series[0].levels) if t.series else 1,
            "resolution": res,
            "description": str(tags.get("ImageDescription", ""))[:200],
        }


def main() -> None:
    run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR])
    run([sys.executable, "-m", "pip", "install", "-q", "pandas", "pyyaml", "tifffile"])
    import pandas as pd
    import yaml

    dataset_yaml = REPO_DIR / "configs/datasets/kpmp_visium.yaml"
    repo = yaml.safe_load(dataset_yaml.read_text())["repository"]
    manifest = pd.read_csv(REPO_DIR / "configs/subsets/kpmp_manifest.csv")
    ids: set[str] = set()
    for name in SUBSET_NAMES:
        ids |= set((REPO_DIR / f"configs/subsets/{name}.txt").read_text().split())
    ids = sorted(ids)
    print(f"{SUBSET_NAMES}: {len(ids)} participants", flush=True)

    summary = []
    for pid in ids:
        rows = manifest[(manifest.participant_id == pid) & manifest.file_kind.isin(FILE_KINDS)]
        out = WORK / pid
        out.mkdir(exist_ok=True)
        print(f"== {pid}: {len(rows)} files", flush=True)
        for r in rows.itertuples():
            dest = out / r.file_name
            prev = find_previous(pid, r.file_name)
            if prev is not None:
                shutil.copy(prev, dest)
                print(f"  {r.file_name}: reused from previous output", flush=True)
            elif not dest.exists():
                url = f"{repo['download_url']}/{r.package_id}/{r.file_name}"
                download(url, dest, int(r.file_size))
            if r.file_kind == "spatial":
                unpack(dest, out)
            if r.file_kind == "image":
                info = describe_tif(dest)
                print(f"  {r.file_name}: {info}", flush=True)
                summary.append({"participant_id": pid, "file_name": r.file_name, **info})
    pd.DataFrame(summary).to_csv(WORK / "image_summary.csv", index=False)
    run(["du", "-sh", WORK])


if __name__ == "__main__":
    main()
