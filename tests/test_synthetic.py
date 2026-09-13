import numpy as np

from lvmh.segmenters.base import CLASS_IDS
from lvmh.synthetic import make_section


def test_section_is_consistent():
    syn = make_section(seed=1)
    sec = syn.section
    assert sec.image.shape == syn.mask.shape == (1024, 1024)
    assert sec.counts.shape == (len(sec.spots), len(sec.genes))
    assert sec.spots.in_tissue.sum() > 10
    assert set(np.unique(syn.mask)) <= set(CLASS_IDS.values())
    assert (syn.structures["class"] == "glomerulus").sum() == 3


def test_same_seed_same_section():
    a, b = make_section(seed=3), make_section(seed=3)
    assert np.array_equal(a.mask, b.mask)
    assert (a.section.counts != b.section.counts).nnz == 0


def test_planted_signal_follows_class_under_spot():
    syn = make_section(seed=2)
    sec = syn.section
    glom_gene = list(CLASS_IDS).index("glomerulus")  # gene j responds to class j mod n
    counts = np.asarray(sec.counts[:, glom_gene].todense()).ravel()
    centre_class = np.array(
        [syn.mask[int(y), int(x)] for x, y in zip(sec.spots.x_px, sec.spots.y_px, strict=True)]
    )
    on_glom = counts[centre_class == CLASS_IDS["glomerulus"]]
    off_glom = counts[centre_class == CLASS_IDS["interstitium"]]
    assert on_glom.mean() > off_glom.mean() + 5


def test_read_region_matches_array():
    syn = make_section(seed=0)
    tile = syn.section.image.read_region(100, 120, 32, 16)
    assert tile.shape == (16, 32, 3)
    assert tile.dtype == np.uint8
