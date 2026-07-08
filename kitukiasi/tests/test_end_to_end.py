import json
import orjson

import numpy as np

from ma3.cities import get_adapter
from ma3.scenarios import exporter, population
from ma3.scenarios.config import Theta, ThetaSpec
from ma3.scenarios.types import Person, Scenario


def test_end_to_end_nairobi_synthetic():
    adapter = get_adapter("nairobi")
    ctx = adapter.build()
    theta = adapter.seed_theta(ctx)
    spec = theta.spec
    scaled = Theta.from_components(
        spec,
        purpose_mix_logits=theta._get("purpose_mix"),
        mode_logits=theta._get("mode_logits"),
        departure_gamma_log=theta._get("departure_gamma"),
        od_log_multipliers=theta.od_log_multipliers,
        global_scale_log=np.log(0.01),
    )
    rng = np.random.default_rng(7)
    arrays = population.generate_arrays(ctx, scaled, rng)
    n = exporter.write_scenario("scenario_test.json", arrays, spec, ctx.abstreet_map_name, "t")
    assert n > 0

    with open("scenario_test.json", "rb") as f:
        raw = f.read()
    obj = orjson.loads(raw)
    assert obj["scenario_name"] == "t"
    assert obj["map_name"] == "nairobi"
    people = obj["people"]
    assert len(people) == n

    for p in people[:20]:
        Person(**p)

    Scenario(scenario_name=obj["scenario_name"], map_name=obj["map_name"], people=people[:20])

    for p in people[:50]:
        trip = p["trips"][0]
        assert isinstance(trip["departure"], int)
        assert 0 < trip["departure"] <= 864000000
        pos = trip["origin"]["Position"]
        assert -90 <= pos["latitude"] <= 90
        assert -180 <= pos["longitude"] <= 180
