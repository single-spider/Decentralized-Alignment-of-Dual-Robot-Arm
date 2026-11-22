"""Saved realizable paired starts; generation independent of evaluated policy."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from dual_arm_mas.world import DualArmWorld
from dual_arm_mas.evidence import provenance


def generate(out, seed, count, purpose):
    rng = np.random.default_rng(seed)
    starts, rejected = [], []
    with DualArmWorld() as world:
        world.attach_parts()
        nominal = world.snapshot()
        targets = {n: world.pose(n) for n in ["A", "B"]}
        for trial in range(count * 5):
            world.restore(nominal)
            try:
                for n in ["A", "B"]:
                    q = np.array(nominal[n])
                    q[2] += rng.uniform(-1.2, 1.2)
                    q[0] += rng.uniform(-.5, .5)
                    world.set_joints(n, q)
                    world.set_joints(n, world.solve(n, *targets[n]))
                world.sync_parts()
                if world.robot_collisions():
                    raise ValueError("start collision")
                coarse = rng.uniform(-1, 1, 3) * [.10, .06, 0]
                world.execute({"A": coarse, "B": coarse})
                requested = rng.uniform(-1, 1, 3) * [.008, .008, .18]
                world.execute({"A": requested})
                starts.append({"id": len(starts), "q": world.snapshot(), "error": world.error().tolist(),
                               "requested_offset": requested.tolist(), "common_coarse": coarse.tolist()})
            except ValueError as exc:
                rejected.append({"trial": trial, "reason": str(exc)})
            if len(starts) == count:
                break
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix(".tmp")
    temp.write_text(json.dumps({"seed": seed, "purpose": purpose, "starts": starts, "rejected": rejected,
                               "provenance": provenance()}, indent=2))
    temp.replace(out)
    print(f"{purpose}: {len(starts)}/{count}, rejected {len(rejected)}, {out}", flush=True)
    if len(starts) != count:
        raise RuntimeError("Generation budget exhausted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--purpose", required=True)
    args = parser.parse_args()
    generate(args.out, args.seed, args.count, args.purpose)
