from __future__ import annotations

import argparse
import time

import numpy as np

from ..cities import get_adapter
from . import exporter, population
from .config import Theta


def compile_scenario(
    ctx,
    theta: Theta,
    out_path: str,
    scenario_name: str = "rl_gen_001",
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    arrays = population.generate_arrays(ctx, theta, rng)
    n = exporter.write_scenario(
        out_path, arrays, theta.spec, ctx.abstreet_map_name, scenario_name
    )
    return n, arrays


def main():
    ap = argparse.ArgumentParser(description="ma3 generative scenario compiler")
    ap.add_argument("--city", default="nairobi")
    ap.add_argument("--out", default="scenario.json")
    ap.add_argument("--scenario-name", default="rl_gen_001")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=None, help="override global demand scale")
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    adapter = get_adapter(args.city, **({"data_dir": args.data_dir} if args.data_dir else {}))
    ctx = adapter.build()
    theta = adapter.seed_theta(ctx)
    if args.scale is not None:
        spec = theta.spec
        theta = Theta.from_components(
            spec,
            purpose_mix_logits=theta._get("purpose_mix"),
            mode_logits=theta._get("mode_logits"),
            departure_gamma_log=theta._get("departure_gamma"),
            od_log_multipliers=theta.od_log_multipliers,
            global_scale_log=np.log(args.scale),
        )

    t0 = time.time()
    n, arrays = compile_scenario(ctx, theta, args.out, args.scenario_name, args.seed)
    dt = time.time() - t0
    print(f"city={ctx.name} cells={len(ctx.cells)} buildings={len(ctx.buildings_lon)}")
    print(f"generated {n} agents in {dt:.2f}s -> {args.out}")
    if n:
        print(
            f"departure range units [{int(arrays['departure'].min())}, {int(arrays['departure'].max())}]"
        )


if __name__ == "__main__":
    main()
