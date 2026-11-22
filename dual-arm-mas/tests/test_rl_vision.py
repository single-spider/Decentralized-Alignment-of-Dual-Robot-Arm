"""Narrow risks: actor boundary, action likelihoods, GAE ends, RGB-only sensing."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import json
import unittest
import numpy as np
import torch
from dual_arm_mas.rl import CooperativePPO,local_features,actions_from_indices
from dual_arm_mas.world import DualArmWorld
from dual_arm_mas.vision import RGBPoseSensor
from sfn.training.rl_utils import compute_gae


class RLAndVisionBoundaries(unittest.TestCase):
    def test_actor_inputs_and_probability(self):
        torch.set_num_threads(1)
        own={"role":"A","error":np.array([.004,-.003,.06]),"own_q":np.zeros(7),
             "own_H":np.eye(3)*.001,"previous_own_motion":np.zeros(3)}
        # Extra hidden/global fields are not features even if a caller adds them.
        first=local_features(own,"s1",{})
        own["central_state"]={"error":[999,999,999]};own["teammate_action"]=np.ones(3)*999
        np.testing.assert_array_equal(first,local_features(own,"s1",{}))
        model=CooperativePPO();features=np.stack([first,first]);indices,lp=model.choose(features)
        with torch.no_grad():recomputed=model.distribution(torch.tensor(features)).log_prob(torch.tensor(indices)).sum(-1)
        np.testing.assert_allclose(lp,recomputed.numpy(),atol=1e-6)
        actions=actions_from_indices(np.full((2,3),4),{"A":own,"B":{**own,"role":"B"}})
        np.testing.assert_allclose(actions["A"],[-.002,.002,-.04]);np.testing.assert_allclose(actions["B"],-actions["A"])

    def test_gae_timeout_bootstrap_not_reset_leak(self):
        values=[torch.tensor(2.),torch.tensor(99.)]
        adv=compute_gae([1.,100.],values,[torch.tensor(3.),torch.tensor(0.)],[False,True],[True,False],.9,.95)
        self.assertAlmostEqual(float(adv[0]),1+.9*3-2,places=5)

    def test_rgb_only_and_blank_refusal(self):
        with DualArmWorld() as world:
            world.attach_parts();sensor=RGBPoseSensor(size=384)
            # Neither true error nor held-part pose accessor can supply sensing.
            def forbidden(*args):raise AssertionError("Privileged sensing path")
            world.error=forbidden;world.part_pose=forbidden
            rgb,out=sensor.observe(world)
            self.assertGreater(int((out["mask"]==1).sum()),10)
            self.assertGreater(int((out["mask"]==2).sum()),10)
            self.assertTrue(out["valid"])
            invalid=sensor.estimate(np.zeros_like(rgb))
            self.assertFalse(invalid["valid"]);self.assertIsNone(invalid["error"])


if __name__=="__main__":unittest.main()
