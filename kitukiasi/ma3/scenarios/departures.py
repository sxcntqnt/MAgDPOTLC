from __future__ import annotations

import numpy as np

from .config import to_departure_units
from . import distributions


def sample_departures(
    rng: np.random.Generator,
    purpose_codes: np.ndarray,
    shapes: np.ndarray,
    scales: np.ndarray,
    departure_unit: str,
) -> np.ndarray:
    sec = distributions.sample_gamma(
        rng, shapes[purpose_codes], scales[purpose_codes]
    )
    sec = np.clip(sec, 0.0, 24.0 * 3600.0)
    return to_departure_units(sec, departure_unit)
