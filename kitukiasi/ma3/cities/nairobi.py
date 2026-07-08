from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import h3
from scipy.sparse import csr_matrix

from .base import CityAdapter, CityContext
from ..scenarios.config import DEFAULT_MODES, DEFAULT_PURPOSES, ThetaSpec
from ..scenarios.config import Theta

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent.parent / "data" / "nairobi"

NAIROBI_BBOX = dict(lat_min=-1.40, lat_max=-1.18, lon_min=36.70, lon_max=36.96)
H3_RES = 8


def _build_synthetic(ctx_name: str) -> CityContext:
    rng = np.random.default_rng(20240708)
    n_build = 4000
    lat = rng.uniform(NAIROBI_BBOX["lat_min"], NAIROBI_BBOX["lat_max"], n_build)
    lon = rng.uniform(NAIROBI_BBOX["lon_min"], NAIROBI_BBOX["lon_max"], n_build)
    area = rng.lognormal(mean=2.0, sigma=0.6, size=n_build)
    cells = np.array([h3.latlng_to_cell(la, lo, H3_RES) for la, lo in zip(lat, lon)])
    uniq = sorted(set(cells.tolist()))
    code_of = {c: i for i, c in enumerate(uniq)}
    cell_codes = np.array([code_of[c] for c in cells])
    n_cells = len(uniq)
    n_edges = min(2000, n_cells * n_cells)
    o = rng.integers(0, n_cells, n_edges)
    d = rng.integers(0, n_cells, n_edges)
    keep = o != d
    o, d = o[keep], d[keep]
    flows = rng.lognormal(mean=5.0, sigma=1.0, size=o.shape[0]).astype(np.float64)
    od = csr_matrix((flows, (o, d)), shape=(n_cells, n_cells))
    return CityContext(
        name=ctx_name,
        h3_resolution=H3_RES,
        abstreet_map_name="nairobi",
        buildings_lon=lon,
        buildings_lat=lat,
        buildings_area=area,
        buildings_cell_code=cell_codes,
        cells=uniq,
        od_matrix=od,
        meta={"synthetic": True},
    )


class NairobiAdapter(CityAdapter):
    def __init__(self, data_dir: str | None = None, h3_resolution: int = H3_RES):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.h3_resolution = h3_resolution

    def build(self) -> CityContext:
        buildings_csv = self.data_dir / "buildings.csv"
        buildings_parquet = self.data_dir / "buildings.parquet"
        od_npz = self.data_dir / "od.npz"
        if buildings_csv.exists() or buildings_parquet.exists():
            return self._build_from_data()
        return _build_synthetic("nairobi")

    def _build_from_data(self) -> CityContext:
        import polars as pl

        if (self.data_dir / "buildings.parquet").exists():
            df = pl.read_parquet(self.data_dir / "buildings.parquet")
        else:
            df = pl.read_csv(self.data_dir / "buildings.csv")
        lon = df["lon"].to_numpy()
        lat = df["lat"].to_numpy()
        area = df["area"].to_numpy() if "area" in df.columns else np.ones(len(lat))
        cells = np.array([h3.latlng_to_cell(la, lo, self.h3_resolution) for la, lo in zip(lat, lat)])
        uniq = sorted(set(cells.tolist()))
        code_of = {c: i for i, c in enumerate(uniq)}
        cell_codes = np.array([code_of[c] for c in cells])

        od = None
        if (self.data_dir / "od.npz").exists():
            npz = np.load(self.data_dir / "od.npz", allow_pickle=True)
            if "matrix" in npz and "cells" in npz:
                od = csr_matrix(npz["matrix"])
                uniq = list(npz["cells"].tolist())
        if od is None:
            od_csv = self.data_dir / "od.csv"
            if od_csv.exists():
                odf = pl.read_csv(od_csv)
                o_code = np.array([code_of[c] for c in odf["o_cell"].to_numpy()])
                d_code = np.array([code_of[c] for c in odf["d_cell"].to_numpy()])
                od = csr_matrix(
                    (odf["flow"].to_numpy(), (o_code, d_code)),
                    shape=(len(uniq), len(uniq)),
                )
        if od is None:
            raise FileNotFoundError(
                "OD data not found: provide od.npz (matrix+cells) or od.csv (o_cell,d_cell,flow)"
            )
        return CityContext(
            name="nairobi",
            h3_resolution=self.h3_resolution,
            abstreet_map_name="nairobi",
            buildings_lon=lon,
            buildings_lat=lat,
            buildings_area=area,
            buildings_cell_code=cell_codes,
            cells=uniq,
            od_matrix=od,
            meta={"synthetic": False},
        )
