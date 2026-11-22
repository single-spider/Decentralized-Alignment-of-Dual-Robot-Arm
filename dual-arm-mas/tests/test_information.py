"""Focused causal-boundary checks; no simulator needed."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from dual_arm_mas.policies import decide, reconstruct_teammate, reconcile_intent, prediction_action, pack_capability, unpack_capability
from dual_arm_mas.episode import SCALE


class InformationTests(unittest.TestCase):
    def setUp(self):
        self.priors = {"A": np.diag([.01, .02, .03]), "B": np.diag([.03, .02, .01])}
        self.obs = {n: {"role": n, "error": np.array([.003, -.004, .08]), "own_q": np.zeros(7),
            "own_H": H.copy(), "previous_own_motion": np.zeros(3)} for n, H in self.priors.items()}

    def test_reconstruction_accounts_for_own_motion(self):
        a, b = np.array([.001, -.002, .03]), np.array([-.002, .001, -.02])
        for n, own, other in [("A", a, b), ("B", b, a)]:
            local = {"error_before": [0, 0, 0], "error_after": a-b, "own_motion": own}
            np.testing.assert_allclose(reconstruct_teammate(n, local), other, atol=1e-12)

    def test_capability_wire_and_complementary_intent(self):
        for n in self.obs:
            payload = pack_capability(self.obs[n])
            self.assertEqual(len(payload), 24)
            np.testing.assert_allclose(unpack_capability(payload), self.obs[n]["own_H"], rtol=1e-6)
        actions, communication = decide("intent", self.obs, self.priors, [.5]*3)
        np.testing.assert_allclose(actions["B"] - actions["A"], np.clip(self.obs["A"]["error"] / SCALE, -1, 1)*SCALE)
        self.assertEqual(communication["bytes"], 24)
        self.assertEqual(communication["rounds"], 1)

    def test_prediction_uses_only_local_history(self):
        counts = np.ones((3, 5))
        own = self.obs["A"]
        history = [{"error_before": own["error"].copy(), "error_after": own["error"]*.75,
                    "own_motion": [-.0002, .0002, -.01]}]
        first = prediction_action(own, counts, history)
        self.obs["B"]["own_q"][:] = 100
        self.obs["B"]["own_H"][:] = 100
        np.testing.assert_array_equal(first, prediction_action(own, counts, history))
        np.testing.assert_array_equal(counts, np.ones((3, 5)))


if __name__ == "__main__":
    unittest.main()
