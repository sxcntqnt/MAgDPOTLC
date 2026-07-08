import numpy as np
import pytest

from ma3.scenarios.config import Theta, ThetaSpec, to_departure_units


def test_theta_layout_and_flatten():
    spec = ThetaSpec(purposes=["Work", "School"], modes=["Walk", "Drive"], od_dim=3)
    assert spec.dim == 2 + 4 + 4 + 3 + 1
    t = Theta.zeros(spec)
    assert t.params.shape == (spec.dim,)
    assert np.allclose(t.flatten(), np.zeros(spec.dim))


def test_theta_softmax_positive():
    spec = ThetaSpec(purposes=["Work", "School"], modes=["Walk", "Drive"], od_dim=2)
    t = Theta.from_components(spec, purpose_mix_logits=[2.0, 0.0])
    pm = t.purpose_mix
    assert np.allclose(pm, np.array([np.exp(2), 1]) / (np.exp(2) + 1))
    assert np.allclose(t.mode_probs().sum(axis=1), 1.0)


def test_departure_units():
    sec = np.array([27735.0])
    assert to_departure_units(sec, "tenths")[0] == 277350000
    assert to_departure_units(sec, "seconds")[0] == 27735
    assert to_departure_units(sec, "milliseconds")[0] == 27735000
