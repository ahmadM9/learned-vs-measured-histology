import numpy as np
import pytest

from lvmh.datasets.kpmp import KPMPVisium
from lvmh.synthetic import make_section, write_kpmp_layout


@pytest.fixture
def kpmp_root(tmp_path):
    for i in range(2):
        syn = make_section(seed=i, section_id=f"sec{i}", participant_id=f"p{i}")
        write_kpmp_layout(syn, tmp_path / f"29-1000{i}")
    return tmp_path


def test_discovers_sections_and_reads_them(kpmp_root):
    ds = KPMPVisium(root=kpmp_root, spot_diameter_um=55.0)
    assert ds.section_ids() == ["29-10000", "29-10001"]
    sec = ds.load("29-10000")
    truth = make_section(seed=0, section_id="sec0", participant_id="p0").section
    assert sec.image.shape == truth.image.shape
    assert sec.pixel_size_um == pytest.approx(truth.pixel_size_um)
    assert len(sec.spots) == int(truth.spots.in_tissue.sum())
    assert sec.counts.shape == (len(sec.spots), len(truth.genes))
    assert sec.genes == truth.genes


def test_counts_align_with_spot_table(kpmp_root):
    sec = KPMPVisium(root=kpmp_root, spot_diameter_um=55.0).load("29-10001")
    truth = make_section(seed=1, section_id="sec1", participant_id="p1").section
    keep = truth.spots.in_tissue.values.astype(bool)
    assert list(sec.spots.barcode) == list(truth.spots.barcode[keep])
    assert (sec.counts != truth.counts[keep]).nnz == 0
    assert np.allclose(sec.spots.x_px.values, truth.spots.x_px.values[keep])


def test_region_read_is_lazy_and_matches(kpmp_root):
    sec = KPMPVisium(root=kpmp_root, spot_diameter_um=55.0).load("29-10000")
    truth = make_section(seed=0, section_id="sec0", participant_id="p0").section
    x, y = int(sec.spots.x_px.iloc[0]), int(sec.spots.y_px.iloc[0])
    tile = sec.image.read_region(x - 16, y - 16, 32, 32)
    assert tile.shape == (32, 32, 3) and tile.dtype == np.uint8
    assert np.array_equal(tile, truth.image.read_region(x - 16, y - 16, 32, 32))


def test_missing_file_names_the_problem(kpmp_root):
    (kpmp_root / "29-10000" / "sec0.tif").unlink()
    with pytest.raises(FileNotFoundError, match="image"):
        KPMPVisium(root=kpmp_root, spot_diameter_um=55.0).load("29-10000")


def test_compressed_tif_falls_back_to_zarr(tmp_path):
    import tifffile

    from lvmh.datasets.kpmp import TiffRegionReader

    syn = make_section(seed=2)
    path = tmp_path / "c.tif"
    tifffile.imwrite(path, syn.section.image.array, photometric="rgb", compression="zlib")
    r = TiffRegionReader(path, 1.0)
    assert r.backend == "zarr"
    expected = syn.section.image.read_region(10, 20, 30, 40)
    assert np.array_equal(r.read_region(10, 20, 30, 40), expected)
    plain = tmp_path / "p.tif"
    tifffile.imwrite(plain, syn.section.image.array, photometric="rgb")
    assert TiffRegionReader(plain, 1.0).backend == "memmap"
