import numpy as np
import h3

from ma3.scenarios import zoning


def test_cell_index_and_sample():
    rng = np.random.default_rng(0)
    codes = np.array([0, 0, 1, 2])
    idx = zoning.build_cell_index(codes)
    assert set(idx.keys()) == {0, 1, 2}
    assert len(idx[0]) == 2

    code_index = idx
    cells_str = ["c0", "c1", "c2", "c3"]
    lon = np.array([36.8, 36.81, 36.82, 36.83])
    lat = np.array([-1.29, -1.30, -1.31, -1.32])
    area = np.ones(4)
    pick = np.array([0, 1, 2])
    olon, olat = zoning.sample_positions(
        pick, rng, code_index, lon, lat, area, cells_str, h3_resolution=8
    )
    assert olon.shape == (3,)
    assert np.all(np.isfinite(olon)) and np.all(np.isfinite(olat))


def test_empty_cell_fallback():
    rng = np.random.default_rng(1)
    A = h3.latlng_to_cell(-1.29, 36.82, 8)
    B = h3.grid_disk(A, 1)[1]
    code_index = {0: np.array([0])}
    cells_str = [A, B]
    lon = np.array([36.82, 0.0])
    lat = np.array([-1.29, 0.0])
    area = np.ones(2)
    pick = np.array([0, 1])
    olon, olat = zoning.sample_positions(
        pick, rng, code_index, lon, lat, area, cells_str, h3_resolution=8
    )
    assert np.all(np.isfinite(olon)) and np.all(np.isfinite(olat))
    assert olon[0] == 36.82
    assert olon[1] == 36.82
