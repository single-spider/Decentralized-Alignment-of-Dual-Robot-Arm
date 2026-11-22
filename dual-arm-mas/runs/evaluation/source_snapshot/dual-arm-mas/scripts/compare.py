"""Matched start comparison; development fitting is explicit, no test tuning."""
from pathlib import Path
import argparse
import json
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from dual_arm_mas.world import DualArmWorld
from dual_arm_mas.episode import AlignmentEpisode, SCALE, capability
from dual_arm_mas.policies import decide, FRACTIONS, Actor, pack_capability, reconstruct_teammate
from dual_arm_mas.evidence import save_result


def fit_public(world, starts):
    matrices = {"A": [], "B": []}
    errors = []
    for start in starts:
        world.restore(start["q"])
        errors.append(np.clip(world.error() / SCALE, -1, 1))
        for n in matrices:
            matrices[n].append(capability(world, n))
    priors = {n: np.mean(H, axis=0) for n, H in matrices.items()}
    scores = []
    for fraction in FRACTIONS:
        scores.append(sum((fraction * e) @ a @ (fraction * e) + ((1-fraction) * e) @ b @ ((1-fraction) * e)
                          for e, a, b in zip(errors, matrices["A"], matrices["B"], strict=True)))
    return {"priors": {n: h.tolist() for n, h in priors.items()}, "convention": FRACTIONS[np.argmin(scores)].tolist()}


def fit_predictor(episodes):
    # Training labels reconstructed from each role's local history, not copied
    # from hidden teammate commands. Laplace smoothing avoids zero probability.
    counts = {n: np.ones((3, 5)) for n in ["A", "B"]}
    for episode in episodes:
        if episode["method"] != "local_model":
            continue
        for transition in episode["trace"]:
            for n in counts:
                local = {"error_before": transition["before"]["error"], "error_after": transition["after"]["error"],
                         "own_motion": transition["executed"][n]}
                teammate = reconstruct_teammate(n, local)
                correction = np.clip(np.array(local["error_before"]) / SCALE, -1, 1) * SCALE
                for axis in range(3):
                    if abs(correction[axis]) > SCALE[axis] * .05:
                        fraction = teammate[axis] / correction[axis] * (1 if n == "A" else -1)
                        counts[n][axis, np.argmin(abs(np.array([0, .25, .5, .75, 1])-fraction))] += 1
    return {n: (4 * c / c.sum(axis=1, keepdims=True)).tolist() for n, c in counts.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--starts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--public", type=Path)
    parser.add_argument("--predictor-training", type=Path)
    parser.add_argument("--train-predictor", action="store_true")
    parser.add_argument("--methods", nargs="+", default=["a_only", "b_only", "half", "full_info", "local_convention", "local_model", "capability"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.starts.read_text())
    episodes = []
    started = time.monotonic()
    with DualArmWorld() as world:
        env = AlignmentEpisode(world)
        public = json.loads(args.public.read_text()) if args.public else fit_public(world, manifest["starts"])
        if args.predictor_training:
            public["predictor_counts"] = fit_predictor(json.loads(args.predictor_training.read_text()))
            public["predictor_training"] = str(args.predictor_training)
        (args.out / "public.json").write_text(json.dumps(public, indent=2))
        priors, convention = public["priors"], public["convention"]
        # Inspect actor boundary with an actual one-observation invocation;
        # perturb unavailable teammate state and ensure local action unchanged.
        obs = env.reset(manifest["starts"][0]["q"])
        own = obs["A"]
        actor = Actor("A", "local_model", priors, convention)
        first = actor.act(own)
        obs["B"]["own_q"][:] = 99
        obs["B"]["own_H"][:] = 99
        information_pass = bool(np.array_equal(first, actor.act(own)))
        payload_bytes = len(pack_capability(own))
        for start in manifest["starts"]:
            for method in args.methods:
                obs = env.reset(start["q"])
                trace = []
                for _ in range(env.config.horizon):
                    actions, communication = decide(method, obs, priors, convention, env.history, public.get("predictor_counts"))
                    obs, _, term, trunc, record = env.step(actions, communication)
                    trace.append(record)
                    if term or trunc:
                        break
                episodes.append({"start_id": start["id"], "method": method, "success": trace[-1]["success"],
                    "final_error": world.error().tolist(), "steps": len(trace),
                    "motion_cost_rad2": sum(t["motion_cost_rad2"] for t in trace),
                    "joint_travel_rad": sum(t["joint_travel_rad"] for t in trace),
                    "invalid_actions": sum(not t["feasible"] for t in trace),
                    "bytes": sum(t["communication"]["bytes"] for t in trace), "trace": trace})
                # Atomic per-episode saves make interrupted runs survivable.
                temp = args.out / "episodes.tmp"
                temp.write_text(json.dumps(episodes))
                temp.replace(args.out / "episodes.json")
        elapsed = time.monotonic() - started
        if args.train_predictor:
            public["predictor_counts"] = fit_predictor(episodes)
            public["predictor_training"] = str(args.out / "episodes.json")
            (args.out / "public.json").write_text(json.dumps(public, indent=2))
        summary = {method: {"episodes": len(selected := [e for e in episodes if e["method"] == method]),
            "success_rate": float(np.mean([e["success"] for e in selected])),
            "mean_motion_cost_rad2": float(np.mean([e["motion_cost_rad2"] for e in selected])),
            "mean_steps": float(np.mean([e["steps"] for e in selected])),
            "mean_bytes": float(np.mean([e["bytes"] for e in selected])),
            "invalid_actions": sum(e["invalid_actions"] for e in selected)} for method in args.methods}
        print(save_result(args.out, {"milestone": "M7-M8", "passed": information_pass and payload_bytes == 24,
            "seed": manifest["seed"], "purpose": manifest["purpose"], "public_fit": str(args.public or args.starts),
            "contract": env.contract(), "summary": summary, "elapsed_s": elapsed,
            "episodes_per_second": len(episodes) / elapsed, "actor_keys": sorted(own),
            "teammate_perturbation_invariance": information_pass, "capability_bytes_each_direction": payload_bytes,
            "limitations": "Myopic finite-grid full-information reference, not global bound; same cost family in local_model/capability; public convention separately tuned on development only"}))


if __name__ == "__main__":
    main()
