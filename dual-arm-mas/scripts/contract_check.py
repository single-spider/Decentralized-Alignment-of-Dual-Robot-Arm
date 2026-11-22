"""M6: realizable starts, own-only capability cost, episode isolation smoke."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from dual_arm_mas.world import DualArmWorld
from dual_arm_mas.episode import AlignmentEpisode, capability, SCALE
from dual_arm_mas.evidence import save_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    manifest, probes, failures = [], [], []
    with DualArmWorld() as world:
        env = AlignmentEpisode(world)
        nominal = world.snapshot()
        targets = {n: world.pose(n) for n in ["A", "B"]}
        # Bounded feasibility search, not an evaluation sample. Diversity is in
        # redundant IK configurations, while held part poses remain realizable.
        for trial in range(40):
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
                # Vary the realizable common coarse position as well as the
                # redundant elbow configuration. Fine errors stay sub-10 mm.
                coarse = rng.uniform(-1, 1, 3) * [.10, .06, 0]
                world.execute({"A": coarse, "B": coarse})
                world.execute({"A": rng.uniform(-1, 1, 3) * [.008, .008, .18]})
                start = world.snapshot()
                H = {n: capability(world, n) for n in ["A", "B"]}
                manifest.append({"id": len(manifest), "q": start, "error": world.error().tolist()})
                for direction in [[.002, 0, 0], [0, .002, 0], [0, 0, .04]]:
                    costs = {}
                    for n in ["A", "B"]:
                        world.restore(start)
                        q0 = world.q(n)
                        execution = world.execute({n: direction})
                        dq = world.q(n) - q0
                        unit = np.array(direction) / SCALE
                        costs[n] = {"predicted_rad2": float(unit @ H[n] @ unit),
                                    "measured_rad2": float(dq @ dq), "joint_travel_rad": float(abs(dq).sum())}
                    probes.append({"start_id": len(manifest)-1, "direction": direction, "costs": costs,
                        "cheaper": min(costs, key=lambda n: costs[n]["measured_rad2"])})
            except ValueError as exc:
                failures.append({"trial": trial, "reason": str(exc)})
            if len(manifest) >= 12:
                break
        if not manifest:
            raise RuntimeError("No realizable start")
        start = manifest[0]["q"]
        observations = env.reset(start)
        initial = env.central_state()
        _, _, terminated, truncated, clipped = env.step({"A": [.01, 0, 0], "B": [0, 0, 0]})
        clipping_pass = abs(clipped["clipped"]["A"][0] - .002) < 1e-12
        env.reset(start)
        reset_pass = (env.central_state() == initial and env.steps == 0 and env.total_bytes == 0
                      and all(not h for h in env.history.values()) and all(not np.any(a) for a in env.previous.values()))
        _, _, term_invalid, trunc_invalid, invalid = env.step({"A": [float("nan"), 0, 0], "B": [0, 0, 0]})
        invalid_pass = term_invalid and not trunc_invalid and env.central_state() == initial
        env.reset(start)
        env.config.horizon = 1
        _, _, term_timeout, trunc_timeout, timeout = env.step({"A": [0, 0, 0], "B": [0, 0, 0]})
        timeout_pass = trunc_timeout and not term_timeout
        env.reset(nominal)
        _, _, term_success, trunc_success, success = env.step({"A": [0, 0, 0], "B": [0, 0, 0]})
        success_pass = term_success and not trunc_success and success["success"]
        preferred = {p["cheaper"] for p in probes if p["direction"] == [.002, 0, 0]}
        ratios = [c["predicted_rad2"] / c["measured_rad2"] for probe in probes for c in probe["costs"].values()]
        cost_pass = all(.7 < r < 1.3 for r in ratios)
        passed = clipping_pass and reset_pass and invalid_pass and timeout_pass and success_pass and cost_pass and preferred == {"A", "B"}
        (args.out / "starts.json").write_text(json.dumps({"seed": args.seed, "purpose": "M6 development pilot; NOT held-out test", "starts": manifest}, indent=2))
        result = {"milestone": "M6", "passed": passed, "seed": args.seed, "contract": env.contract(),
            "checks": {"clipping": clipping_pass, "reset_isolation": reset_pass, "invalid_atomic": invalid_pass,
                       "timeout": timeout_pass, "success_terminal": success_pass, "cost_prediction": cost_pass, "changing_x_preference": sorted(preferred)},
            "cost_ratio_range": [min(ratios), max(ratios)], "probes": probes, "generation_failures": failures,
            "actor_keys": {n: sorted(o) for n, o in observations.items()}, "invalid_transition": invalid,
            "clipped_transition": clipped, "timeout_transition": timeout}
        print(save_result(args.out, result))
        return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
