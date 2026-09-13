import numpy as np
import pytest

from lvmh.registry import Registry
from lvmh.segmenters.base import CLASS_IDS
from lvmh.segmenters.tiling import segment_image
from lvmh.synthetic import make_section


@pytest.fixture
def seg():
    return Registry("segmenters").build("colour_synthetic")


def test_tiled_prediction_equals_whole_image_prediction(seg):
    syn = make_section(seed=4)
    tiled = segment_image(syn.section.image, seg, overlap_px=32, background_fraction=2.0)
    whole = seg.segment_tile(syn.section.image.array)
    assert tiled.array.shape == whole.shape
    assert np.array_equal(tiled.array, whole)


def test_tiled_prediction_recovers_synthetic_mask(seg):
    syn = make_section(seed=5)
    tiled = segment_image(syn.section.image, seg, overlap_px=32)
    agree = (tiled.array == syn.mask).mean()
    assert agree > 0.99
    assert set(np.unique(tiled.array)) >= {CLASS_IDS["glomerulus"], CLASS_IDS["tubule"]}


def test_resampling_to_model_resolution(seg):
    syn = make_section(seed=6)
    seg.pixel_size_um = 2.0  # model wants a coarser image than the section
    tiled = segment_image(syn.section.image, seg, overlap_px=32)
    assert tiled.array.shape == (512, 512)
    assert tiled.pixel_size_um == 2.0
    coarse_truth = syn.mask[::2, ::2]
    assert (tiled.array == coarse_truth).mean() > 0.97
    x = tiled.to_mask_px(np.array([100.0]), syn.section.pixel_size_um)
    assert x[0] == pytest.approx(50.0)


def test_background_tiles_skip_the_model(seg):
    calls = []
    original = seg.segment_tile

    def counting(tile):
        calls.append(1)
        return original(tile)

    seg.segment_tile = counting
    syn = make_section(seed=7)
    segment_image(syn.section.image, seg, overlap_px=32)
    n_tiles = int(np.ceil(1024 / (256 - 64))) ** 2
    assert 0 < len(calls) < n_tiles
