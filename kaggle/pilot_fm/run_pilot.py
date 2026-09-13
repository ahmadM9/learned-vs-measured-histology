"""Kaggle kernel for the stage 0b pilot: encoder features on the pilot sections, scored.

Submitted with the Kaggle CLI from the repo root:

    kaggle kernels push -p kaggle/pilot_fm/ --accelerator NvidiaTeslaT4

The kernel clones the public repo, installs it, reads the sections from the
attached output of lvmh-pull-sections, takes the Hugging Face token from the
Kaggle secret HF_TOKEN, runs lvmh.embed and lvmh.evaluate for the listed run
configs with out_dir rewritten to /kaggle/working, and prints the fold table.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_URL = "https://github.com/ahmadM9/learned-vs-measured-histology.git"
REPO_DIR = Path("/tmp/repo")
WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")

# run configs to execute in order; edit before pushing
CONFIG_NAMES = ["kpmp_random_pilot10.yaml"]


def run(cmd, **kwargs):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kwargs)


def find_data_root() -> Path:
    # the pull kernel's output mounts under /kaggle/input/<...>/ with one directory per
    # participant; the mount can lag behind kernel start, hence the retries
    for attempt in range(6):
        for summary in sorted(INPUT.glob("**/image_summary.csv")):
            return summary.parent
        print(f"attempt {attempt + 1}: no image_summary.csv yet under {INPUT}", flush=True)
        time.sleep(20)
    raise SystemExit(f"no pull_sections output found under {INPUT}")


def set_hf_token() -> None:
    try:
        from kaggle_secrets import UserSecretsClient

        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        print("HF_TOKEN set from kaggle secrets", flush=True)
    except Exception as e:  # gated models then fail at download with a clear message
        print(f"no HF_TOKEN secret: {e}", flush=True)


def main() -> None:
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR])
    run([sys.executable, "-m", "pip", "install", "-q", REPO_DIR])
    set_hf_token()
    data_root = find_data_root()
    print(f"data root: {data_root}", flush=True)

    for name in CONFIG_NAMES:
        cfg = yaml.safe_load((REPO_DIR / "configs" / "runs" / name).read_text())
        out_dir = WORK / Path(cfg["out_dir"]).name
        cfg["out_dir"] = str(out_dir)
        if cfg.get("subset"):
            cfg["subset"] = str(REPO_DIR / cfg["subset"])
        # the dataset yaml points at data/kpmp; override the root for this machine
        cfg["dataset_overrides"] = {"root": str(data_root)}
        config_path = WORK / f"config_{Path(name).stem}.yaml"
        config_path.write_text(yaml.safe_dump(cfg))
        env = {**os.environ, "LVMH_CONFIG_ROOT": str(REPO_DIR / "configs")}
        run([sys.executable, "-m", "lvmh.embed", "--config", config_path], env=env)
        run([sys.executable, "-m", "lvmh.evaluate", "--config", config_path], env=env)
        run(["ls", "-la", out_dir])


if __name__ == "__main__":
    main()
