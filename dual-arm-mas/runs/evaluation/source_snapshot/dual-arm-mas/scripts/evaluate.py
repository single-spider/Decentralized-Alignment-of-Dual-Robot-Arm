"""Bounded, resumable frozen evaluation; append one measured episode at a time."""
from pathlib import Path
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from dual_arm_mas.world import DualArmWorld
from dual_arm_mas.episode import AlignmentEpisode, EpisodeConfig
from dual_arm_mas.policies import Protocol
from dual_arm_mas.evidence import provenance


def run(args):
    config = json.loads(args.config.read_text())
    manifest = json.loads(Path(config["starts"]).read_text())
    scenarios = [s for s in config["scenarios"] if not args.scenario or s["name"] == args.scenario]
    fingerprint = hashlib.sha256(args.config.read_bytes()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    lock = args.out / "writer.lock"
    with lock.open("x") as f:
        f.write(json.dumps({"pid": os.getpid(), "started": datetime.now(timezone.utc).isoformat()}))
    try:
        output = args.out / "episodes.jsonl"
        seen = set()
        if output.exists():
            for line in output.read_text().splitlines():
                row = json.loads(line)
                if row["config_sha256"] != fingerprint:
                    raise ValueError("Cannot resume with a changed configuration")
                seen.add((row["scenario"], row["fit"], row["start_id"], row["method"]))
        source = provenance()
        meta = args.out / "provenance.json"
        if meta.exists() and json.loads(meta.read_text())["sha256"] != source["sha256"]:
            raise ValueError("Source changed since this evaluation started; use a new output directory")
        if not meta.exists():
            meta.write_text(json.dumps(source, indent=2))
        started = time.monotonic()
        delivered = 0
        with DualArmWorld() as world, output.open("a", encoding="utf-8") as stream:
            for scenario in scenarios:
                env = AlignmentEpisode(world, EpisodeConfig(**scenario.get("episode", {})))
                for fit, public_path in enumerate(config["public"]):
                    if fit not in scenario.get("fits", [0]):
                        continue
                    public = json.loads(Path(public_path).read_text())
                    count = min(scenario.get("count", len(manifest["starts"])), len(manifest["starts"]))
                    for start in manifest["starts"][:count]:
                        for method in scenario["methods"]:
                            key = (scenario["name"], fit, start["id"], method)
                            if key in seen:
                                continue
                            now = datetime.now(timezone(timedelta(hours=5, minutes=30)))
                            if now >= datetime.fromisoformat(config["wrapup_at"]) or time.monotonic()-started >= args.seconds:
                                print(f"Slice stopped safely: {delivered} episodes; next {key}", flush=True)
                                return
                            noise_seed = config["seed"] + start["id"] * 101  # paired draws, reset each method
                            obs = env.reset(start["q"], seed=noise_seed)
                            protocol = Protocol(public, seed=noise_seed+1, **scenario.get("protocol", {}))
                            trace = []
                            for step in range(env.config.horizon):
                                actor_errors = {n: o["error"].tolist() for n, o in obs.items()}
                                actions, communication = protocol.actions(method, obs, env.history)
                                obs, _, term, trunc, record = env.step(actions, communication)
                                record.pop("execution")  # q/poses already in before/after, no duplicate dump
                                record["actor_error"] = actor_errors
                                trace.append(record)
                                if term or trunc:
                                    break
                            tol = np.array(env.config.tolerance)
                            reversals = sum(int(np.any((np.array(t["before"]["error"])*t["after"]["error"] < 0) &
                                (np.abs(t["before"]["error"]) > tol) & (np.abs(t["after"]["error"]) > tol))) for t in trace)
                            row = {"config_sha256": fingerprint, "scenario": scenario["name"], "fit": fit,
                                "start_id": start["id"], "method": method, "success": trace[-1]["success"],
                                "final_error": world.error().tolist(), "steps": len(trace),
                                "motion_cost_rad2": sum(t["motion_cost_rad2"] for t in trace),
                                "joint_travel_rad": sum(t["joint_travel_rad"] for t in trace),
                                "reward": sum(t["reward"] for t in trace), "overshoot_steps": reversals,
                                "same_world_direction_steps": sum(int(np.any(np.array(t["executed"]["A"])*t["executed"]["B"] > 1e-10)) for t in trace),
                                "invalid_actions": sum(not t["feasible"] for t in trace),
                                "bytes": sum(t["communication"]["bytes"] for t in trace),
                                "messages": sum(t["communication"]["messages"] for t in trace),
                                "rounds": sum(t["communication"]["rounds"] for t in trace), "trace": trace}
                            stream.write(json.dumps(row, allow_nan=False) + "\n")
                            stream.flush()
                            seen.add(key)
                            delivered += 1
                            if delivered % 100 == 0:
                                print(f"Saved {delivered} new, {len(seen)} total; {key}; {time.monotonic()-started:.1f}s", flush=True)
        # This detects accidental edits while a source-frozen slice ran.
        if source["sha256"] != provenance()["sha256"]:
            raise RuntimeError("Source changed during slice; results require provenance audit")
        print(f"Selected scenarios complete: {delivered} new episodes", flush=True)
    finally:
        lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenario")
    parser.add_argument("--seconds", type=float, default=240)
    run(parser.parse_args())
