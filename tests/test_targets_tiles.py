import numpy as np
import scipy.sparse as sp

from lvmh.synthetic import make_section
from lvmh.targets import log_expression, select_top_variance_genes, targets
from lvmh.tiles import spot_tiles


def test_log_expression_is_log1p_of_raw_counts_by_default():
    counts = sp.csr_matrix(np.array([[0, 1, 3], [2, 0, 0]], dtype=np.float32))
    assert np.allclose(log_expression(counts), np.log1p([[0, 1, 3], [2, 0, 0]]))
    scaled = log_expression(counts, normalize_total=4)
    assert np.allclose(scaled[0], np.log1p(np.array([0, 1, 3]) * 4 / 4))
    assert np.allclose(scaled[1], np.log1p(np.array([2, 0, 0]) * 4 / 2))


def test_top_variance_genes_are_the_planted_ones():
    secs = [make_section(seed=s).section for s in range(2)]
    genes = select_top_variance_genes(secs, n_genes=5)
    assert len(genes) == 5 and len(set(genes)) == 5
    y = targets(secs[0], genes)
    assert y.shape == (len(secs[0].spots), 5)


def test_spot_tiles_are_centred_and_resampled():
    sec = make_section(seed=3).section
    batches = list(spot_tiles(sec, tile_px=32, pixel_size_um=2.0, batch_size=40))
    idx = np.concatenate([b[0] for b in batches])
    tiles = np.concatenate([b[1] for b in batches])
    assert len(idx) == len(sec.spots) and tiles.shape == (len(sec.spots), 32, 32, 3)
    assert [len(b[0]) for b in batches] == [40, 40, 10]
    # a 32 px tile at 2 um covers 64 image px; its centre pixel is the spot centre
    i = int(idx[0])
    x, y = int(sec.spots.x_px.iloc[i]), int(sec.spots.y_px.iloc[i])
    centre = sec.image.read_region(x - 2, y - 2, 4, 4).mean(axis=(0, 1))
    assert np.abs(tiles[0, 15:17, 15:17].mean(axis=(0, 1)) - centre).max() < 25
