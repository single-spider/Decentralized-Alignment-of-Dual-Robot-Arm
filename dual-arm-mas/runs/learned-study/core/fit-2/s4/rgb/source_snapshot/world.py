"""One clock, two handles. Explicitly kinematic, not motor/contact validation."""
from dataclasses import dataclass, asdict
import math

import numpy as np
import pybullet as p
import pybullet_data
from sfn.panda.kinematics import joint_limit_arrays


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass
class WorldConfig:
    time_step: float = 1 / 240
    command_steps: int = 24
    execution_mode: str = "kinematic"
    gravity: tuple = (0., 0., 0.)
    position_tolerance_m: float = 0.0001
    orientation_tolerance_rad: float = 0.002


class DualArmWorld:
    """No handle may step. All target planning restores state before execution."""
    joints = tuple(range(7))
    ee_link = 11

    def __init__(self, config=None):
        self.config = config or WorldConfig()
        if self.config.execution_mode != "kinematic" or self.config.command_steps < 1:
            raise ValueError("Only explicit kinematic mode is currently implemented")
        self.p = p
        self.client_id = p.connect(p.DIRECT)
        self.tick = 0
        cid = self.client_id
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
        p.setTimeStep(self.config.time_step, physicsClientId=cid)
        p.setGravity(*self.config.gravity, physicsClientId=cid)
        self.robots = {}
        self.parts = {}
        self.adjacent_links = {}
        try:
            self.floor = p.loadURDF("plane.urdf", physicsClientId=cid)
            for name, x, y, yaw in [("A", -.55, -.2, 0), ("B", .55, .3, math.pi)]:
                body = p.loadURDF("franka_panda/panda.urdf", [x, y, 0], p.getQuaternionFromEuler([0, 0, yaw]),
                                  useFixedBase=True, physicsClientId=cid)
                self.robots[name] = body
                info = {j: p.getJointInfo(body, j, physicsClientId=cid) for j in range(p.getNumJoints(body, physicsClientId=cid))}
                excluded = set()
                for child in info:
                    ancestor, moving = child, 0
                    while ancestor != -1:
                        moving += int(info[ancestor][2] != p.JOINT_FIXED)
                        ancestor = info[ancestor][16]
                        if moving <= 1:
                            excluded.add((min(child, ancestor), max(child, ancestor)))
                self.adjacent_links[name] = excluded
                self.set_joints(name, [0, -.4, 0, -2.2, 0, 1.8, .8])
                for j in [9, 10]:
                    p.resetJointState(body, j, .01, physicsClientId=cid)
            targets = {"A": ([0, 0, .50], p.getQuaternionFromEuler([math.pi, 0, 0])),
                       "B": ([0, .28, .42], p.getQuaternionFromEuler([math.pi, 0, math.pi]))}
            prepared = {n: self.solve(n, *pose) for n, pose in targets.items()}
            for n, q in prepared.items():
                self.set_joints(n, q)
            if self.robot_collisions():
                raise ValueError("Initial robot collision")
        except Exception:
            self.close()
            raise

    def close(self):
        if p.isConnected(self.client_id):
            p.disconnect(self.client_id)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def q(self, name):
        return np.array([p.getJointState(self.robots[name], j, physicsClientId=self.client_id)[0] for j in self.joints])

    def set_joints(self, name, q):
        for j, value in zip(self.joints, q, strict=True):
            p.resetJointState(self.robots[name], j, float(value), physicsClientId=self.client_id)

    def pose(self, name):
        state = p.getLinkState(self.robots[name], self.ee_link, computeForwardKinematics=True, physicsClientId=self.client_id)
        return np.array(state[4]), np.array(state[5])

    def solve(self, name, pos, quat):
        """Full-pose IK with measured residual and joint limits; no world step."""
        saved = self.q(name)
        lower, upper, ranges = joint_limit_arrays(p, self.robots[name], self.joints, self.client_id)
        try:
            for _ in range(12):
                raw = p.calculateInverseKinematics(self.robots[name], self.ee_link, list(pos), list(quat),
                    lowerLimits=lower, upperLimits=upper, jointRanges=ranges, restPoses=saved.tolist(),
                    maxNumIterations=200, residualThreshold=1e-10, physicsClientId=self.client_id)
                candidate = np.array(raw[:7])
                self.set_joints(name, candidate)
                actual_pos, actual_quat = self.pose(name)
                pos_err = np.linalg.norm(actual_pos - pos)
                rot_err = 2 * np.arccos(np.clip(abs(np.dot(actual_quat, quat)), 0, 1))
                if pos_err < self.config.position_tolerance_m and rot_err < self.config.orientation_tolerance_rad:
                    break
            if np.any(candidate < np.array(lower)) or np.any(candidate > np.array(upper)):
                raise ValueError(f"{name}: IK joint limits")
            if pos_err >= self.config.position_tolerance_m or rot_err >= self.config.orientation_tolerance_rad:
                raise ValueError(f"{name}: IK residual {pos_err} m, {rot_err} rad")
            return candidate
        finally:
            self.set_joints(name, saved)

    def measure(self):
        return {n: {"q": self.q(n).tolist(), "position_m": self.pose(n)[0].tolist(),
                    "quaternion_xyzw": self.pose(n)[1].tolist()} for n in self.robots}

    def attach_parts(self):
        """Kinematic visual compound fixture; no collision/contact claim."""
        if self.parts:
            raise RuntimeError("Parts already attached")
        self.grasps = {
            "A": ([0, 0, .03], p.getQuaternionFromEuler([math.pi, 0, 0])),
            "B": ([0, -.28, 0], p.getQuaternionFromEuler([math.pi, 0, math.pi])),
        }
        peg = p.createVisualShape(p.GEOM_BOX, halfExtents=[.005, .005, .03], rgbaColor=[.1, .3, .9, 1], physicsClientId=self.client_id)
        # Square ring opening is 14 mm across, plus a handle toward B's jaws.
        boxes = [[.009, .025, .007], [.009, .025, .007], [.007, .009, .007], [.007, .009, .007], [.009, .128, .007]]
        positions = [[-.016, 0, 0], [.016, 0, 0], [0, -.016, 0], [0, .016, 0], [0, .153, 0]]
        fixture = p.createVisualShapeArray([p.GEOM_BOX] * 5, halfExtents=boxes,
            visualFramePositions=positions, rgbaColors=[[.2, .65, .4, 1]] * 5, physicsClientId=self.client_id)
        for n, visual in [("A", peg), ("B", fixture)]:
            self.parts[n] = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual, baseCollisionShapeIndex=-1, physicsClientId=self.client_id)
        seam = p.createVisualShapeArray([p.GEOM_BOX] * 4,
            halfExtents=[[.0006, .0076, .0002]] * 2 + [[.0076, .0006, .0002]] * 2,
            visualFramePositions=[[-.0076, 0, .0072], [.0076, 0, .0072], [0, -.0076, .0072], [0, .0076, .0072]],
            rgbaColors=[[1, .8, .1, 1]] * 4, physicsClientId=self.client_id)
        self.seam = p.createMultiBody(baseMass=0, baseVisualShapeIndex=seam, baseCollisionShapeIndex=-1, physicsClientId=self.client_id)
        self.sync_parts()

    def expected_part_pose(self, name):
        return p.multiplyTransforms(*self.pose(name), *self.grasps[name])

    def sync_parts(self):
        for n, body in self.parts.items():
            p.resetBasePositionAndOrientation(body, *self.expected_part_pose(n), physicsClientId=self.client_id)
        if self.parts:
            p.resetBasePositionAndOrientation(self.seam, *self.expected_part_pose("B"), physicsClientId=self.client_id)

    def part_pose(self, name):
        pos, quat = p.getBasePositionAndOrientation(self.parts[name], physicsClientId=self.client_id)
        return np.array(pos), np.array(quat)

    def task_poses(self):
        peg_pos, peg_quat = self.part_pose("A")
        tip, _ = p.multiplyTransforms(peg_pos, peg_quat, [0, 0, -.03], [0, 0, 0, 1])
        hole, hole_quat = self.part_pose("B")
        return {"A": np.array([tip[0], tip[1], p.getEulerFromQuaternion(peg_quat)[2]]),
                "B": np.array([hole[0], hole[1], p.getEulerFromQuaternion(hole_quat)[2]])}

    def error(self):
        poses = self.task_poses()
        diff = poses["A"] - poses["B"]
        diff[2] = wrap(diff[2])
        return diff

    def attachment_drift(self):
        residuals = {}
        for n in self.parts:
            expected = self.expected_part_pose(n)
            actual = self.part_pose(n)
            residuals[n] = {"position_m": float(np.linalg.norm(actual[0] - expected[0])),
                            "quaternion_error": float(max(0, 1 - abs(np.dot(actual[1], expected[1]))))}
        seam_pos, _ = p.getBasePositionAndOrientation(self.seam, physicsClientId=self.client_id)
        residuals["seam_position_m"] = float(np.linalg.norm(np.array(seam_pos) - self.part_pose("B")[0]))
        return residuals

    def snapshot(self):
        return {n: self.q(n).tolist() for n in self.robots}

    def restore(self, snapshot):
        for n in self.robots:
            self.set_joints(n, snapshot[n])
        self.sync_parts()
        self.tick = 0

    def inter_arm_penetration(self):
        contacts = p.getClosestPoints(self.robots["A"], self.robots["B"], 0, physicsClientId=self.client_id)
        return [{"link_A": c[3], "link_B": c[4], "distance_m": c[8]} for c in contacts if c[8] < -1e-6]

    def robot_collisions(self):
        found = self.inter_arm_penetration()
        for name, body in self.robots.items():
            for c in p.getClosestPoints(body, body, 0, physicsClientId=self.client_id):
                a, b = c[3], c[4]
                if a < b and c[8] < -1e-6 and (a, b) not in self.adjacent_links[name]:
                    found.append({"robot": name, "self_links": [a, b], "distance_m": c[8]})
            for c in p.getClosestPoints(body, self.floor, 0, physicsClientId=self.client_id):
                if c[3] >= 0 and c[8] < -1e-6:
                    found.append({"robot": name, "floor_link": c[3], "distance_m": c[8]})
        return found

    def prepare(self, name, delta):
        pos, quat = self.part_pose(name) if self.parts else self.pose(name)
        rpy = p.getEulerFromQuaternion(quat)
        quat_target = p.getQuaternionFromEuler([rpy[0], rpy[1], rpy[2] + delta[2]])
        if not np.any(delta):
            return self.q(name)
        target = (pos + [delta[0], delta[1], 0], quat_target)
        if self.parts:
            target = p.multiplyTransforms(*target, *p.invertTransform(*self.grasps[name]))
        return self.solve(name, *target)

    def execute(self, actions):
        """Both actions are world XYZ/yaw increments prepared from one snapshot."""
        before = self.measure()
        prepared = {}
        for n in self.robots:
            delta = np.asarray(actions.get(n, [0, 0, 0]), dtype=float)
            if delta.shape != (3,) or not np.all(np.isfinite(delta)):
                raise ValueError("Actions must be finite [dx_m,dy_m,dyaw_rad]")
            prepared[n] = self.prepare(n, delta)
        # One owner advances the common clock; kinematic joint interpolation is
        # applied to BOTH robots each tick. Motors/contact are not being tested.
        # Validate the same discrete path BEFORE executing either arm. This is
        # sampled collision checking, not continuous collision certification.
        try:
            for step in range(1, self.config.command_steps + 1):
                f = step / self.config.command_steps
                for n in self.robots:
                    q0 = np.array(before[n]["q"])
                    self.set_joints(n, q0 + f * (prepared[n] - q0))
                if self.robot_collisions():
                    raise ValueError("Robot collision on sampled command path")
        finally:
            for n in self.robots:
                self.set_joints(n, before[n]["q"])
        for step in range(1, self.config.command_steps + 1):
            f = step / self.config.command_steps
            for n in self.robots:
                q0 = np.array(before[n]["q"])
                self.set_joints(n, q0 + f * (prepared[n] - q0))
            p.stepSimulation(physicsClientId=self.client_id)
            # Cancel motor integration in this explicitly reset-based mode.
            for n in self.robots:
                q0 = np.array(before[n]["q"])
                self.set_joints(n, q0 + f * (prepared[n] - q0))
            self.sync_parts()
            self.tick += 1
        return {"before": before, "after": self.measure(), "tick": self.tick,
                "prepared_q": {n: q.tolist() for n, q in prepared.items()}, "config": asdict(self.config)}
