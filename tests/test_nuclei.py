import numpy as np
import pytest

from lvmh.config import load_run
from lvmh.embed import run as embed_run
from lvmh.evaluate import run as evaluate_run
from lvmh.nuclei import FEATURE_NAMES, ThresholdNuclei, nuclear_features
from lvmh.registry import CONFIG_ROOT, Registry
from lvmh.segmenters.base import CLASS_IDS
from lvmh.synthetic import make_section


def test_synthetic_nuclei_are_dark_and_inside_tissue():
    syn = make_section(seed=1, nuclei_per_mm2=3000)
    img = syn.section.image.array
    dark = img.mean(axis=-1) < 60
    assert 0.005 < dark.mean() < 0.2
    assert (syn.mask[dark] == CLASS_IDS["background"]).mean() < 0.01
    assert np.array_equal(syn.mask, make_section(seed=1).mask)  # nuclei never touch the mask


def test_nuclear_features_on_a_known_crop():
    lab = np.zeros((41, 41), dtype=np.int32)
    lab[5:10, 5:10] = 1  # 25 px square, centroid (7, 7)
    lab[20:24, 20:24] = 2  # 16 px square, centroid (21.5, 21.5)
    disc = np.ones_like(lab, dtype=bool)
    f = nuclear_features(lab, disc, pixel_size_um=2.0)
    assert f[0] == 2
    assert f[2] == pytest.approx((25 + 16) / 2 * 4)
    assert f[6] == pytest.approx(np.hypot(14.5, 14.5) * 2.0)
    disc[:] = False
    assert nuclear_features(lab, disc, 2.0).sum() == 0


def test_threshold_model_and_featurizer_end_to_end():
    syn = make_section(seed=2, nuclei_per_mm2=3000)
    sec = syn.section
    feat = Registry("featurizers").build("threshold_nuclei_synthetic")
    assert feat.names == FEATURE_NAMES
    x, seconds = feat.featurize(sec)
    assert x.shape == (len(sec.spots), len(FEATURE_NAMES)) and seconds >= 0
    centre = np.array(
        [syn.mask[int(y), int(xx)] for xx, y in zip(sec.spots.x_px, sec.spots.y_px, strict=True)]
    )
    r = int(sec.spot_diameter_px / 2) + 1
    fully_bg = np.array(
        [
            (syn.mask[int(y) - r : int(y) + r, int(xx) - r : int(xx) + r] == 0).all()
            for xx, y in zip(sec.spots.x_px, sec.spots.y_px, strict=True)
        ]
    )
    on_glom = x[np.isin(centre, [CLASS_IDS["glomerulus"], CLASS_IDS["tuft"]]), 1]
    on_int = x[centre == CLASS_IDS["interstitium"], 1]
    assert fully_bg.sum() > 5 and x[fully_bg, 0].max() == 0
    assert on_glom.mean() > on_int.mean()
    masks = ThresholdNuclei().segment([sec.image.array[400:600, 400:600]], diameter_px=6)
    assert masks[0].dtype == np.int32 and masks[0].max() > 0


def test_nuclei_smoke_run(tmp_path):
    cfg = load_run(CONFIG_ROOT / "runs" / "synthetic_nuclei_smoke.yaml")
    cfg.raw["out_dir"] = str(tmp_path / "run")
    paths = embed_run(cfg)
    z = np.load(paths[0])
    assert list(z["feature_names"]) == FEATURE_NAMES and str(z["source"]).startswith("threshold")
    tables = evaluate_run(cfg)
    assert tables["folds"].pearson_mean.mean() > 0.1  # density tracks the planted glomerulus gene


def test_run_must_name_one_feature_source(tmp_path):
    cfg = load_run(CONFIG_ROOT / "runs" / "synthetic_nuclei_smoke.yaml")
    cfg.components.pop("featurizer")
    with pytest.raises(ValueError, match="exactly one"):
        embed_run(cfg)
