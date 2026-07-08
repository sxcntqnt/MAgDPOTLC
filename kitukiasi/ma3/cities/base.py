from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.sparse import csr_matrix

from ..scenarios.config import DEFAULT_MODES, DEFAULT_PURPOSES, Theta, ThetaSpec


@dataclass
class CityContext:
    name: str
    h3_resolution: int
    abstreet_map_name: str
    buildings_lon: np.ndarray
    buildings_lat: np.ndarray
    buildings_area: np.ndarray
    buildings_cell_code: np.ndarray
    cells: List[str]
    od_matrix: csr_matrix
    ground_truth: Optional[object] = None
    meta: Dict = field(default_factory=dict)


class CityAdapter(ABC):
    @abstractmethod
    def build(self) -> CityContext:
        ...

    def seed_theta(self, ctx: CityContext) -> Theta:
        spec = ThetaSpec(
            purposes=list(DEFAULT_PURPOSES),
            modes=list(DEFAULT_MODES),
            od_dim=int(ctx.od_matrix.nnz),
            departure_unit="tenths",
        )
        shape = np.full(spec.n_purposes, np.log(4.0))
        scale = np.full(spec.n_purposes, np.log(7200.0))
        return Theta.from_components(
            spec,
            departure_gamma_log=np.stack([shape, scale], axis=1).ravel(),
        )
