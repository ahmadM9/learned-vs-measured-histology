"""Write the manifest of open KPMP Visium files from the repository search endpoint.

    python scripts/kpmp_manifest.py --out configs/subsets/kpmp_manifest.csv
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import pandas as pd
import yaml

DATASET_YAML = Path(__file__).resolve().parents[1] / "configs" / "datasets" / "kpmp_visium.yaml"
KEEP = [
    "redcap_id", "file_name", "package_id", "file_size", "data_format", "enrollment_category",
    "primary_adjudicated_category", "sample_type", "tissue_source", "sex", "age_binned", "dois",
]


def fetch_records(repo: dict) -> list[dict]:
    headers = {"Authorization": f"Bearer {repo['search_key']}", "Content-Type": "application/json"}
    filters = {
        "all": [
            {"experimental_strategy": repo["experimental_strategy"]},
            {"access": repo["access"]},
        ]
    }
    rows, page = [], 1
    while True:
        page_spec = {"size": repo["page_size"], "current": page}
        body = json.dumps({"query": "", "page": page_spec, "filters": filters}).encode()
        req = urllib.request.Request(repo["search_url"], data=body, headers=headers)
        with urllib.request.urlopen(req) as r:
            d = json.load(r)
        for rec in d["results"]:
            rows.append({k: v["raw"] for k, v in rec.items() if k != "_meta"})
        if page >= d["meta"]["page"]["total_pages"]:
            return rows
        page += 1


def to_manifest(rows: list[dict], repo: dict) -> pd.DataFrame:
    kind_of = {fmt: kind for kind, fmt in repo["file_kinds"].items()}
    out = []
    for r in rows:
        row = {k: r.get(k) for k in KEEP}
        for k in ("redcap_id", "enrollment_category", "primary_adjudicated_category", "sample_type",
                  "tissue_source", "sex", "age_binned"):
            v = row[k]
            row[k] = v[0] if isinstance(v, list) and v else v
        row["dois"] = ";".join(row["dois"]) if row["dois"] else ""
        row["file_kind"] = kind_of.get(r.get("data_format"), "other")
        row["image_size_class"] = ""
        if row["file_kind"] == "image":
            small = float(r["file_size"]) < repo["small_image_bytes"]
            row["image_size_class"] = "small" if small else "large"
        out.append(row)
    df = pd.DataFrame(out).rename(columns={"redcap_id": "participant_id"})
    return df.sort_values(["participant_id", "file_kind", "file_name"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    repo = yaml.safe_load(DATASET_YAML.read_text())["repository"]
    df = to_manifest(fetch_records(repo), repo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    n_img = (df.file_kind == "image").sum()
    n_p = df.participant_id.nunique()
    print(f"{len(df)} files, {n_p} participants, {n_img} images -> {args.out}")


if __name__ == "__main__":
    main()
