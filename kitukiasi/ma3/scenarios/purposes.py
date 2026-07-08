from __future__ import annotations

import numpy as np

from . import distributions


def assign_purpose(
    rng: np.random.Generator, n: int, purpose_probs: np.ndarray
) -> np.ndarray:
    return distributions.categorical(rng, purpose_probs, size=n)


def assign_mode(
    rng: np.random.Generator, purpose_codes: np.ndarray, mode_probs: np.ndarray
) -> np.ndarray:
    out = np.empty(len(purpose_codes), dtype=np.int64)
    for p in range(mode_probs.shape[0]):
        mask = purpose_codes == p
        if not mask.any():
            continue
        out[mask] = distributions.categorical(rng, mode_probs[p], size=int(mask.sum()))
    return out
