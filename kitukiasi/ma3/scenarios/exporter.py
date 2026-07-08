from __future__ import annotations

import orjson

from .config import ThetaSpec


def _mode_name(spec: ThetaSpec, code: int) -> str:
    return spec.modes[int(code)]


def _purpose_name(spec: ThetaSpec, code: int) -> str:
    return spec.purposes[int(code)]


def iter_people(arrays: dict, spec: ThetaSpec):
    dep = arrays["departure"]
    olon, olat = arrays["orig_lon"], arrays["orig_lat"]
    dlon, dlat = arrays["dest_lon"], arrays["dest_lat"]
    mode, purp = arrays["mode"], arrays["purpose"]
    for i in range(len(dep)):
        yield {
            "trips": [
                {
                    "departure": int(dep[i]),
                    "origin": {
                        "Position": {
                            "longitude": float(olon[i]),
                            "latitude": float(olat[i]),
                        }
                    },
                    "destination": {
                        "Position": {
                            "longitude": float(dlon[i]),
                            "latitude": float(dlat[i]),
                        }
                    },
                    "mode": _mode_name(spec, mode[i]),
                    "purpose": _purpose_name(spec, purp[i]),
                }
            ]
        }


def write_scenario(path: str, arrays: dict, spec: ThetaSpec, map_name: str, scenario_name: str):
    n = int(arrays.get("n", len(arrays["departure"])))
    header = (
        b'{"scenario_name":'
        + orjson.dumps(scenario_name)
        + b',"map_name":'
        + orjson.dumps(map_name)
        + b',"people":['
    )
    with open(path, "wb") as f:
        f.write(header)
        first = True
        for person in iter_people(arrays, spec):
            chunk = orjson.dumps(person)
            if not first:
                f.write(b",")
            f.write(chunk)
            first = False
        f.write(b"]}")
    return n
