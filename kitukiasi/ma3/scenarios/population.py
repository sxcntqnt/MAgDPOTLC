from __future__ import annotations

import numpy as np

from . import departures, od_matrix, purposes, zoning
from .config import Theta


def generate_arrays(ctx, theta: Theta, rng: np.random.Generator) -> dict:
    coo = od_matrix.scale_od(ctx.od_matrix, theta.od_log_multipliers, theta.global_scale)
    o_idx = coo.row.astype(np.int64)
    d_idx = coo.col.astype(np.int64)
    base = coo.data.astype(np.float64)
    nnz = len(base)

    purpose_probs = theta.purpose_mix
    P = theta.spec.n_purposes
    demand_ep = base[:, None] * purpose_probs[None, :]
    counts_ep = np.round(demand_ep).astype(np.int64)

    if counts_ep.sum() == 0:
        return _empty_arrays()

    ep_flat = np.repeat(np.arange(nnz * P), counts_ep.ravel())
    pers_edge = ep_flat // P
    pers_purpose = ep_flat % P
    N = len(pers_edge)

    o_code = o_idx[pers_edge]
    d_code = d_idx[pers_edge]

    code_index = zoning.build_cell_index(ctx.buildings_cell_code)

    orig_lon, orig_lat = zoning.sample_positions(
        o_code, rng, code_index, ctx.buildings_lon, ctx.buildings_lat,
        ctx.buildings_area, ctx.cells, ctx.h3_resolution,
    )
    dest_lon, dest_lat = zoning.sample_positions(
        d_code, rng, code_index, ctx.buildings_lon, ctx.buildings_lat,
        ctx.buildings_area, ctx.cells, ctx.h3_resolution,
    )

    shapes, scales = theta.departure_gamma
    departure = departures.sample_departures(
        rng, pers_purpose, shapes, scales, theta.spec.departure_unit
    )

    mode = purposes.assign_mode(rng, pers_purpose, theta.mode_probs())

    return {
        "departure": departure,
        "orig_lon": orig_lon,
        "orig_lat": orig_lat,
        "dest_lon": dest_lon,
        "dest_lat": dest_lat,
        "mode": mode.astype(np.int64),
        "purpose": pers_purpose.astype(np.int64),
        "n": N,
    }


def _empty_arrays():
    return {
        "departure": np.empty(0, np.int64),
        "orig_lon": np.empty(0),
        "orig_lat": np.empty(0),
        "dest_lon": np.empty(0),
        "dest_lat": np.empty(0),
        "mode": np.empty(0, np.int64),
        "purpose": np.empty(0, np.int64),
        "n": 0,
    }
