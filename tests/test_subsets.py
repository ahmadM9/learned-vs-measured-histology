from pathlib import Path

import pandas as pd
import yaml

from make_subsets import eligible, make_lists

SUBSETS = Path(__file__).parents[1] / "configs" / "subsets"


def test_committed_manifest_matches_stage0_inventory():
    m = pd.read_csv(SUBSETS / "kpmp_manifest.csv")
    assert m.participant_id.nunique() == 209
    assert (m.file_kind == "image").sum() == 210
    assert set(m.image_size_class.dropna()) == {"large", "small"}


def test_multi_section_participants_are_excluded():
    m = pd.read_csv(SUBSETS / "kpmp_manifest.csv")
    pool = eligible(m, exclude_multi_section=True)
    assert "28-12265" not in set(pool.participant_id)
    assert pool.participant_id.is_unique


def test_lists_are_seeded_and_match_committed_files():
    m = pd.read_csv(SUBSETS / "kpmp_manifest.csv")
    spec = yaml.safe_load((SUBSETS / "subsets.yaml").read_text())
    lists = make_lists(m, spec)
    assert lists == make_lists(m, spec)
    img = m[m.file_kind == "image"].set_index("participant_id")
    assert len(lists["segmenter_check_5"]) == 5
    assert (img.loc[lists["segmenter_check_5"], "image_size_class"] == "small").sum() == 2
    assert len(lists["pilot_10"]) == 10
    assert img.loc[lists["pilot_10"], "enrollment_category"].nunique() >= 3
    for name, ids in lists.items():
        committed = (SUBSETS / f"{name}.txt").read_text().split()
        assert committed == ids, f"{name}.txt is stale, rerun scripts/make_subsets.py"
