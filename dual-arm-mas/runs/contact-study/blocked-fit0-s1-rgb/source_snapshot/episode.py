"""Execution-only local observations, atomic joint actions, measured costs."""
from dataclasses import asdict, dataclass

import numpy as np

from .world import DualArmWorld, wrap

SCALE = np.array([.002, .002, .04])  # normalized action units
TOLERANCE = np.array([.001, .001, np.deg2rad(1)])


@dataclass
class EpisodeConfig:
    horizon: int = 20
    motion_weight: float = 1000.
    byte_weight: float = .0001
    step_cost: float = .01
    success_bonus: float = 10.
    tolerance: tuple = (.001, .001, np.deg2rad(1))
    observation_noise: tuple = (0., 0., 0.)
    a_limit_factor: float = 1.
    b_limit_factor: float = 1.


def capability(world, name):
    """Own-only 7x3 full-pose constrained IK derivative; not a bare XY/yaw Jacobian.

    Coordinates are normalized by SCALE; H predicts squared joint radians.
    IK probes restore measured state and never advance the clock.
    """
    columns = []
    for axis in range(3):
        d = np.zeros(3)
        d[axis] = SCALE[axis] * .25
        columns.append((world.prepare(name, d) - world.prepare(name, -d)) / .5)
    derivative = np.array(columns).T
    return derivative.T @ derivative + np.eye(3) * 1e-12


class AlignmentEpisode:
    def __init__(self, world=None, config=None):
        self.world = world or DualArmWorld()
        self.config = config or EpisodeConfig()
        if not self.world.parts:
            self.world.attach_parts()
        self.nominal = self.world.snapshot()
        self.reset(self.nominal)

    def reset(self, start, seed=0):
        self.world.restore(start)
        self.rng = np.random.default_rng(seed)
        self.sensed_error = self.world.error() + self.rng.normal(size=3) * self.config.observation_noise
        self.steps = 0
        self.history = {"A": [], "B": []}
        self.previous = {n: np.zeros(3) for n in ["A", "B"]}
        self.total_bytes = 0
        self.done = False
        return self.observations()

    def observations(self):
        # Do not put central state, teammate joints/H/action or training labels
        # into these dictionaries. Each policy receives ONE such dictionary.
        error = self.sensed_error
        return {n: {"role": n, "error": error.copy(), "own_q": self.world.q(n).copy(),
                    "own_H": capability(self.world, n), "previous_own_motion": self.previous[n].copy()}
                for n in ["A", "B"]}

    def central_state(self):
        return {"error": self.world.error().tolist(), "q": self.world.snapshot()}

    def step(self, actions, communication=None):
        if self.done:
            raise RuntimeError("Reset required after terminal or truncated transition")
        communication = communication or {"messages": 0, "bytes": 0, "rounds": 0}
        before_state = self.central_state()
        sensed_before = self.sensed_error.copy()
        before_poses = self.world.task_poses()
        requested, clipped = {}, {}
        failure = None
        try:
            if set(actions) != {"A", "B"}:
                raise ValueError("Both actions must be supplied")
            for n in ["A", "B"]:
                action = np.asarray(actions[n], dtype=float)
                if action.shape != (3,) or not np.all(np.isfinite(action)):
                    raise ValueError("Nonfinite or malformed action")
                requested[n] = action.tolist()
                factor = self.config.a_limit_factor if n == "A" else self.config.b_limit_factor
                clipped[n] = np.clip(action, -SCALE * factor, SCALE * factor)
            execution = self.world.execute(clipped)
        except ValueError as exc:
            failure = str(exc)
            execution = None
        self.steps += 1
        after_poses = self.world.task_poses()
        contributions = {n: after_poses[n] - before_poses[n] for n in ["A", "B"]}
        for value in contributions.values():
            value[2] = wrap(value[2])
        dq = {n: self.world.q(n) - before_state["q"][n] for n in ["A", "B"]}
        motion_cost = float(sum(q @ q for q in dq.values()))
        joint_travel = float(sum(np.abs(q).sum() for q in dq.values()))
        error = self.world.error()
        self.sensed_error = error + self.rng.normal(size=3) * self.config.observation_noise
        success = bool(np.all(np.abs(error) <= self.config.tolerance) and failure is None)
        terminated = success or failure is not None
        truncated = not terminated and self.steps >= self.config.horizon
        self.done = terminated or truncated
        self.total_bytes += int(communication["bytes"])
        reward = (-float(np.sum((error / self.config.tolerance) ** 2)) - self.config.motion_weight * motion_cost
                  - self.config.byte_weight * communication["bytes"] - self.config.step_cost
                  + (self.config.success_bonus if success else 0) - (10 if failure else 0))
        record = {"step": self.steps, "before": before_state, "after": self.central_state(),
            "requested": requested, "clipped": {n: a.tolist() for n, a in clipped.items()},
            "feasible": failure is None, "invalid_reason": failure,
            "executed": {n: a.tolist() for n, a in contributions.items()},
            "motion_cost_rad2": motion_cost, "joint_travel_rad": joint_travel,
            "communication": communication, "reward": reward, "success": success,
            "terminated": terminated, "truncated": truncated, "execution": execution}
        for n in ["A", "B"]:
            # Own history includes public error transitions and own measured
            # motion only; no current or previous teammate command is stored.
            self.history[n].append({"error_before": sensed_before.tolist(), "error_after": self.sensed_error.tolist(),
                                    "own_motion": contributions[n].tolist()})
            self.previous[n] = contributions[n]
        return self.observations(), reward, terminated, truncated, record

    def contract(self):
        return {"action_scale_m_rad": SCALE.tolist(), "success_tolerance_m_rad": list(self.config.tolerance),
            "config": asdict(self.config), "mode": "kinematic", "gravity": self.world.config.gravity,
            "cost": "sum_i ||q_after-q_before||^2 rad^2; joint travel is not energy",
            "constraint_limits": "joint limits, full-pose IK, sampled inter-arm/self/floor collisions; adjacent URDF overlap excluded; visual part collision/contact excluded"}
