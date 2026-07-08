from __future__ import annotations

import numpy as np
import h3
from shapely.geometry import Polygon, Point


def build_cell_index(building_cell_codes: np.ndarray) -> dict:
    idx = np.arange(len(building_cell_codes))
    out = {}
    for code in np.unique(building_cell_codes):
        out[int(code)] = idx[building_cell_codes == code]
    return out


def sample_positions(
    cell_codes: np.ndarray,
    rng: np.random.Generator,
    code_index: dict,
    building_lon: np.ndarray,
    building_lat: np.ndarray,
    building_area: np.ndarray,
    cells_str: list,
    h3_resolution: int,
) -> tuple:
    n = len(cell_codes)
    out_lon = np.empty(n, dtype=np.float64)
    out_lat = np.empty(n, dtype=np.float64)
    str_to_code = {s: i for i, s in enumerate(cells_str)}
    for code in np.unique(cell_codes):
        code = int(code)
        mask = cell_codes == code
        cnt = int(mask.sum())
        idxs = code_index.get(code)
        if idxs is not None and len(idxs) > 0:
            out_lon[mask], out_lat[mask] = _sample_from_cell(
                rng, idxs, building_lon, building_lat, building_area, cnt
            )
        else:
            lon, lat = _fallback_position(
                code, rng, code_index, cells_str, str_to_code, h3_resolution,
                building_lon, building_lat,
            )
            out_lon[mask] = lon
            out_lat[mask] = lat
    return out_lon, out_lat


def _sample_from_cell(rng, idxs, building_lon, building_lat, building_area, cnt):
    w = building_area[idxs].astype(np.float64)
    if w.sum() <= 0 or not np.all(np.isfinite(w)):
        chosen = idxs[rng.integers(0, len(idxs), size=cnt)]
    else:
        p = w / w.sum()
        chosen = idxs[rng.choice(len(idxs), size=cnt, p=p)]
    return building_lon[chosen], building_lat[chosen]


def _fallback_position(code, rng, code_index, cells_str, str_to_code, h3_resolution,
                        building_lon, building_lat):
    base = cells_str[code]
    for k in range(1, 5):
        for s in h3.grid_disk(base, k):
            c = str_to_code.get(s)
            if c is None:
                continue
            idxs = code_index.get(c)
            if idxs is not None and len(idxs) > 0:
                j = idxs[rng.integers(0, len(idxs))]
                return building_lon[j], building_lat[j]
    boundary = h3.cell_to_boundary(base)
    poly = Polygon([(lon, lat) for lat, lon in boundary])
    minx, miny, maxx, maxy = poly.bounds
    for _ in range(100):
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        if poly.contains(Point(x, y)):
            return x, y
    c = poly.centroid
    return c.x, c.y
