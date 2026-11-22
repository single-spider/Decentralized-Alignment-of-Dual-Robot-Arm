"""Bounded milestone demos. Run with the existing SFN virtual environment."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dual_arm_mas.evidence import save_result, scene_image


def one_arm(out, seed):
    from sfn.panda.config import PandaConfig
    from sfn.panda.panda_scene import PandaScene
    import numpy as np

    config = PandaConfig(execution_mode="kinematic", command_steps=120)
    with PandaScene(config=config, seed=seed) as scene:
        before = scene.measure()
        result = scene.execute_cartesian_delta(0.002, 0, 0)
        after = scene.measure()
        delta = after.peg_pos_world - before.peg_pos_world
        passed = bool(result.ik_success and np.linalg.norm(delta - [0.002, 0, 0]) < 0.0002)
        scene_image(scene.p, scene.client_id, out / "scene.png", [-0.75, 0, 0.2], 1.4)
        return save_result(out, {
            "milestone": "M1", "passed": passed, "seed": seed,
            "config": config.to_dict(), "requested_delta_m": [0.002, 0, 0],
            "measured_delta_m": delta.tolist(), "before": before.to_dict(),
            "after": after.to_dict(), "execution": result.to_dict(),
            "seams": "PandaScene owns its client and execute_joint_target steps internally; two instances are not a dual-arm world.",
        })


def two_arm(out, seed):
    from dual_arm_mas.world import DualArmWorld
    import numpy as np
    with DualArmWorld() as world:
        result = world.execute({"B": [0.002, 0, 0]})
        movement = {n: (np.array(result["after"][n]["position_m"]) - result["before"][n]["position_m"]).tolist() for n in ["A", "B"]}
        passed = bool(np.linalg.norm(movement["A"]) < 1e-7 and np.linalg.norm(np.array(movement["B"]) - [.002, 0, 0]) < .0002)
        scene_image(world.p, world.client_id, out / "scene.png")
        return save_result(out, {"milestone": "M2", "passed": passed, "seed": seed,
            "robots": world.robots, "client": world.client_id, "movement_m": movement, "execution": result})


def held_parts(out, seed):
    from dual_arm_mas.world import DualArmWorld
    import numpy as np
    with DualArmWorld() as world:
        world.attach_parts()
        before = world.task_poses()
        trace = []
        for actions in [{"A": [.002, 0, .02]}, {"B": [0, -.002, -.02]}]:
            execution = world.execute(actions)
            trace.append({"actions": actions, "execution": execution, "drift": world.attachment_drift(),
                          "error": world.error().tolist()})
        after = world.task_poses()
        delta = {n: (after[n] - before[n]).tolist() for n in before}
        passed = bool(np.linalg.norm(np.array(delta["A"]) - [.002, 0, .02]) < .0003 and
                      np.linalg.norm(np.array(delta["B"]) - [0, -.002, -.02]) < .0003 and
                      all(t["drift"]["seam_position_m"] < 1e-7 for t in trace))
        scene_image(world.p, world.client_id, out / "scene.png", [0, 0, .4], 1.3)
        scene_image(world.p, world.client_id, out / "parts.png", [0, .04, .45], .52)
        return save_result(out, {"milestone": "M3", "passed": passed, "seed": seed,
            "representation": "kinematic primitive peg/compound visual fixture, collision disabled; no contact validation",
            "grasps": world.grasps, "delta_m_rad": delta, "trace": trace})


def relative_checks(out, seed):
    from dual_arm_mas.world import DualArmWorld, wrap
    import numpy as np
    checks = []
    with DualArmWorld() as world:
        world.attach_parts()
        start = world.snapshot()
        for name in ["A", "B"]:
            for axis, magnitude in [(0, .002), (1, .002), (2, .03)]:
                world.restore(start)
                before = world.error()
                action = np.zeros(3)
                action[axis] = magnitude
                execution = world.execute({name: action})
                measured = world.error() - before
                measured[2] = wrap(measured[2])
                expected = action * (1 if name == "A" else -1)
                residual = measured - expected
                checks.append({"arm": name, "axis": axis, "measured": measured.tolist(),
                    "expected": expected.tolist(), "residual": residual.tolist(),
                    "passed": bool(max(abs(residual[:2])) < .0001 and abs(residual[2]) < .002),
                    "execution": execution})
        world.restore(start)
        before = world.error()
        execution = world.execute({"A": [.002, -.001, 0], "B": [.002, -.001, 0]})
        residual = world.error() - before
        checks.append({"name": "common_translation", "residual": residual.tolist(),
            "passed": bool(max(abs(residual[:2])) < .0001 and abs(residual[2]) < .002), "execution": execution})
        return save_result(out, {"milestone": "M4", "passed": all(c["passed"] for c in checks),
            "seed": seed, "frame": "world X/Y; e=peg_tip_xy-hole_xy, wrap(peg_yaw-hole_yaw), metres/radians",
            "checks": checks})


def align(out, seed):
    from dual_arm_mas.world import DualArmWorld
    import numpy as np
    episodes = []
    frames = []
    with DualArmWorld() as world:
        world.attach_parts()
        nominal = world.snapshot()
        for offset in [[.006, -.004, .12], [-.007, .003, -.15], [.003, .007, .06]]:
            world.restore(nominal)
            world.execute({"A": offset})
            start = world.snapshot()
            for method, share in [("a_only", 1.), ("half", .5)]:
                world.restore(start)
                trace = []
                for step in range(20):
                    error = world.error()
                    if max(abs(error[:2])) <= .001 and abs(error[2]) <= np.deg2rad(1):
                        break
                    correction = np.clip(error, [-.002, -.002, -.04], [.002, .002, .04])
                    before = world.task_poses()
                    execution = world.execute({"A": -share * correction, "B": (1-share) * correction})
                    after = world.task_poses()
                    trace.append({"before_error": error.tolist(), "after_error": world.error().tolist(),
                        "actual_contributions": {n: (after[n] - before[n]).tolist() for n in before}, "execution": execution})
                    if method == "half" and offset == [.006, -.004, .12]:
                        frames.append(scene_image(world.p, world.client_id, out / "selected.png", [0, .04, .45], .65))
                final = world.error()
                episodes.append({"offset": offset, "method": method, "start_q": start, "steps": len(trace),
                    "final_error": final.tolist(), "success": bool(max(abs(final[:2])) <= .001 and abs(final[2]) <= np.deg2rad(1)), "trace": trace})
        if frames:
            frames[0].save(out / "alignment.gif", save_all=True, append_images=frames[1:], duration=500, loop=0)
        return save_result(out, {"milestone": "M5", "passed": all(e["success"] for e in episodes),
            "seed": seed, "episodes": episodes, "limits_m_rad": [.002, .002, .04],
            "limitations": "Three fixed representative offsets, smoke only; kinematic/noncontact, no statistical claim"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("milestone", choices=["m1", "m2", "m3", "m4", "m5"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result = {"m1": one_arm, "m2": two_arm, "m3": held_parts, "m4": relative_checks, "m5": align}[args.milestone](args.out, args.seed)
    print(result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
