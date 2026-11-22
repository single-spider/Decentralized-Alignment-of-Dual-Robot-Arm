"""Only new noise-history, routing-reset and model-reload risk checks."""
from pathlib import Path
import json
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from dual_arm_mas.world import DualArmWorld
from dual_arm_mas.episode import AlignmentEpisode, EpisodeConfig
from dual_arm_mas.policies import Protocol


class EpisodeTests(unittest.TestCase):
    def test_noise_history_and_protocol_reset(self):
        with DualArmWorld() as world:
            env = AlignmentEpisode(world, EpisodeConfig(observation_noise=(.0002, .0002, .003)))
            world.execute({"A": [.004, 0, .08]})
            start = world.snapshot()
            before = env.reset(start, seed=55)
            after, _, _, _, record = env.step({"A": [0, 0, 0], "B": [0, 0, 0]})
            for n in ["A", "B"]:
                np.testing.assert_array_equal(env.history[n][0]["error_before"], before[n]["error"])
                np.testing.assert_array_equal(env.history[n][0]["error_after"], after[n]["error"])
            self.assertFalse(np.array_equal(before["A"]["error"], record["before"]["error"]))
            public = {"priors": {n: before[n]["own_H"].tolist() for n in before}, "convention": [.5]*3}
            public = json.loads(json.dumps(public))  # save/reload precision and schema
            protocol = Protocol(public, delay=1)
            actions, comm = protocol.actions("intent", before, env.history)
            self.assertEqual(comm["received_messages"], 0)
            _, comm = protocol.actions("intent", before, env.history)
            self.assertEqual(comm["received_messages"], 2)
            reset_protocol = Protocol(public, delay=1)
            again, comm = reset_protocol.actions("intent", before, {"A": [], "B": []})
            self.assertEqual(comm["received_messages"], 0)
            for n in actions:
                np.testing.assert_array_equal(actions[n], again[n])


if __name__ == "__main__":
    unittest.main()
