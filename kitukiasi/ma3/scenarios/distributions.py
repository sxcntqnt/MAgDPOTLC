from __future__ import annotations

import numpy as np


def sample_gamma(rng: np.random.Generator, shape, scale, size=None):
    return rng.gamma(shape, scale, size=size)


def sample_normal(rng: np.random.Generator, mean, std, size=None):
    return rng.normal(mean, std, size=size)


def sample_poisson(rng: np.random.Generator, lam, size=None):
    return rng.poisson(lam, size=size)


def categorical(rng: np.random.Generator, probs: np.ndarray, size: int) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    p = probs / probs.sum()
    return rng.choice(len(probs), size=size, p=p)
