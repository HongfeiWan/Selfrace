import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "simulator"))

from simulator import TeraflowSimulator


class _Planner:
    def __init__(self, candidates):
        self.routable_quad_ids = torch.tensor(candidates, dtype=torch.int32)


class RouteSamplingTests(unittest.TestCase):
    def make_simulator(self):
        simulator = TeraflowSimulator.__new__(TeraflowSimulator)
        simulator.device = torch.device("cpu")
        simulator.road_network = SimpleNamespace(num_quads=10)
        simulator.path_planner = _Planner([2, 4, 6, 8])
        return simulator

    def test_first_goal_is_independent_and_uniform_over_routable_map(self):
        simulator = self.make_simulator()
        torch.manual_seed(7)
        valid = torch.ones((4000, 1), dtype=torch.bool)

        sampled = simulator._sample_uniform_route_quads(valid).flatten()

        self.assertEqual(set(sampled.tolist()), {2, 4, 6, 8})
        counts = torch.stack([(sampled == value).sum() for value in (2, 4, 6, 8)])
        self.assertTrue(bool(((counts > 850) & (counts < 1150)).all()))

    def test_inactive_vehicle_has_no_goal(self):
        simulator = self.make_simulator()
        valid = torch.tensor([[True, False], [False, True]])

        sampled = simulator._sample_uniform_route_quads(valid)

        self.assertTrue(bool((sampled[~valid] == -1).all()))
        self.assertTrue(bool((sampled[valid] >= 0).all()))


if __name__ == "__main__":
    unittest.main()
