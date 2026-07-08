from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List

DEFAULT_PURPOSES = ["Work", "School", "Shopping", "Social", "Freight", "Other"]
DEFAULT_MODES = ["Walk", "Bike", "Drive", "Transit"]


def to_departure_units(seconds: np.ndarray, unit: str) -> np.ndarray:
    if unit == "seconds":
        return np.round(seconds).astype(np.int64)
    if unit == "tenths":
        return np.round(seconds * 10000.0).astype(np.int64)
    if unit == "milliseconds":
        return np.round(seconds * 1000.0).astype(np.int64)
    raise ValueError(f"unknown departure_unit: {unit}")


@dataclass
class ThetaSpec:
    purposes: List[str] = field(default_factory=lambda: list(DEFAULT_PURPOSES))
    modes: List[str] = field(default_factory=lambda: list(DEFAULT_MODES))
    od_dim: int = 0
    departure_unit: str = "tenths"

    @property
    def n_purposes(self) -> int:
        return len(self.purposes)

    @property
    def n_modes(self) -> int:
        return len(self.modes)

    def layout(self):
        return [
            ("purpose_mix", self.n_purposes),
            ("mode_logits", self.n_purposes * self.n_modes),
            ("departure_gamma", self.n_purposes * 2),
            ("od_log_multipliers", self.od_dim),
            ("global_scale", 1),
        ]

    @property
    def dim(self) -> int:
        return sum(length for _, length in self.layout())


class Theta:
    def __init__(self, params: np.ndarray, spec: ThetaSpec):
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (spec.dim,):
            raise ValueError(f"theta params shape {params.shape} != ({spec.dim},)")
        self.params = params
        self.spec = spec
        self._slices = {}
        idx = 0
        for name, length in spec.layout():
            self._slices[name] = (idx, idx + length)
            idx += length

    def _get(self, name: str) -> np.ndarray:
        s, e = self._slices[name]
        return self.params[s:e]

    @property
    def purpose_mix(self) -> np.ndarray:
        x = self._get("purpose_mix")
        e = np.exp(x - np.max(x))
        return e / e.sum()

    @property
    def mode_logits(self) -> np.ndarray:
        return self._get("mode_logits").reshape(self.spec.n_purposes, self.spec.n_modes)

    def mode_probs(self) -> np.ndarray:
        logits = self.mode_logits
        logits = logits - np.max(logits, axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)

    @property
    def departure_gamma(self):
        g = self._get("departure_gamma").reshape(self.spec.n_purposes, 2)
        shape = np.exp(g[:, 0])
        scale = np.exp(g[:, 1])
        return shape, scale

    @property
    def od_log_multipliers(self) -> np.ndarray:
        return self._get("od_log_multipliers")

    @property
    def global_scale(self) -> float:
        return float(np.exp(self._get("global_scale")[0]))

    @staticmethod
    def zeros(spec: ThetaSpec) -> "Theta":
        return Theta(np.zeros(spec.dim, dtype=np.float64), spec)

    @staticmethod
    def from_components(
        spec: ThetaSpec,
        purpose_mix_logits=None,
        mode_logits=None,
        departure_gamma_log=None,
        od_log_multipliers=None,
        global_scale_log=None,
    ) -> "Theta":
        params = np.zeros(spec.dim, dtype=np.float64)
        t = Theta(params, spec)
        if purpose_mix_logits is not None:
            t.params[t._slices["purpose_mix"][0]:t._slices["purpose_mix"][1]] = np.asarray(purpose_mix_logits, dtype=np.float64)
        if mode_logits is not None:
            s, e = t._slices["mode_logits"]
            t.params[s:e] = np.asarray(mode_logits, dtype=np.float64).ravel()
        if departure_gamma_log is not None:
            s, e = t._slices["departure_gamma"]
            t.params[s:e] = np.asarray(departure_gamma_log, dtype=np.float64).ravel()
        if od_log_multipliers is not None:
            s, e = t._slices["od_log_multipliers"]
            t.params[s:e] = np.asarray(od_log_multipliers, dtype=np.float64).ravel()
        if global_scale_log is not None:
            s, e = t._slices["global_scale"]
            t.params[s:e] = np.asarray(global_scale_log, dtype=np.float64)
        return t

    def flatten(self) -> np.ndarray:
        return self.params.copy()
