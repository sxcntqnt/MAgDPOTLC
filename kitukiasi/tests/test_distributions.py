import numpy as np

from ma3.scenarios import distributions


def test_gamma_moments():
    rng = np.random.default_rng(3)
    x = distributions.sample_gamma(rng, 4.0, 2.0, size=200000)
    assert abs(x.mean() - 8.0) < 0.1
    assert abs(x.var() - 16.0) < 0.5


def test_categorical():
    rng = np.random.default_rng(4)
    counts = np.bincount(distributions.categorical(rng, [0.5, 0.5], size=10000), minlength=2)
    assert abs(counts[0] / 10000 - 0.5) < 0.05
