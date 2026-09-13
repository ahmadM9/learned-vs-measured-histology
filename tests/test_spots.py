import numpy as np
import pytest

from lvmh.segmenters.base import CLASS_IDS
from lvmh.segmenters.tiling import LabelMask
from lvmh.spots import assign_spots, disc, label_structures
from lvmh.synthetic import make_section


def test_structures_match_drawn_objects():
    syn = make_section(seed=8)
    st = label_structures(LabelMask(syn.mask, 1.0))
    drawn = syn.structures["class"].value_counts()
    found = st.table["class"].value_counts()
    assert found["glomerulus"] == drawn["glomerulus"]
    # tubules can touch each other when drawn, so at most as many found as drawn
    assert 0 < found["tubule"] <= drawn["tubule"]
    assert (st.instances > 0).sum() == st.table.area_px.sum()


def test_min_area_drops_small_objects():
    syn = make_section(seed=8)
    st = label_structures(LabelMask(syn.mask, 1.0), min_area_px=10**6)
    assert st.table.empty and st.instances.max() == 0


def test_spot_inside_glomerulus_gets_fusion_weight():
    syn = make_section(seed=9)
    mask = LabelMask(syn.mask, 1.0)
    st = label_structures(mask)
    g = syn.structures[syn.structures["class"] == "glomerulus"].iloc[0]
    spots = syn.section.spots.iloc[:1].copy()
    spots["x_px"], spots["y_px"] = g.cx, g.cy  # centre the spot on the glomerulus
    long, classes = assign_spots(spots, 55.0, 1.0, mask, st)
    assert len(long) == 1
    row = long.iloc[0]
    glom_area = st.table.set_index("structure_id").loc[row.structure_id, "area_px"]
    assert row.overlap_px == disc(27.5).sum()
    assert row.weight == pytest.approx(row.overlap_px / glom_area)
    assert row.spot_fraction == pytest.approx(1.0)
    c = classes.iloc[0]
    assert c.frac_glomerulus + c.frac_tuft == pytest.approx(1.0)


def test_background_spot_keeps_a_row_with_zero_structure_weight():
    syn = make_section(seed=9)
    mask = LabelMask(syn.mask, 1.0)
    st = label_structures(mask)
    spots = syn.section.spots.iloc[:1].copy()
    spots["x_px"], spots["y_px"] = 30.0, 30.0  # corner, outside the tissue blob
    long, classes = assign_spots(spots, 55.0, 1.0, mask, st)
    assert long.empty
    assert classes.iloc[0].frac_background == pytest.approx(1.0)


def test_class_fractions_sum_to_one_and_scale_with_mask_resolution():
    syn = make_section(seed=10)
    sec = syn.section
    fine = LabelMask(syn.mask, 1.0)
    coarse = LabelMask(syn.mask[::2, ::2], 2.0)
    _, cf = assign_spots(sec.spots, sec.spot_diameter_px, 1.0, fine, label_structures(fine))
    _, cc = assign_spots(sec.spots, sec.spot_diameter_px, 1.0, coarse, label_structures(coarse))
    frac_cols = [c for c in cf.columns if c.startswith("frac_")]
    assert np.allclose(cf[frac_cols].sum(axis=1), 1.0)
    assert np.abs(cf[frac_cols].values - cc[frac_cols].values).mean() < 0.02


def test_weights_of_one_structure_sum_to_covered_fraction():
    syn = make_section(seed=11)
    sec = syn.section
    mask = LabelMask(syn.mask, 1.0)
    st = label_structures(mask)
    long, _ = assign_spots(sec.spots, sec.spot_diameter_px, 1.0, mask, st)
    per_structure = long.groupby("structure_id").weight.sum()
    assert (per_structure <= 1.0 + 1e-9).all()
    assert CLASS_IDS["glomerulus"] in set(syn.mask.ravel())
