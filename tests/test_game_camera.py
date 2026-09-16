import math
import unittest
from types import SimpleNamespace

import torch

from game.game import InferenceGame


def make_game(states: torch.Tensor, *, follows_heading: bool = False) -> InferenceGame:
    game = InferenceGame.__new__(InferenceGame)
    game.simulator = SimpleNamespace(agents_state=states)
    game.current_world = 0
    game.selected_agent = 0
    game.camera_follows_heading = follows_heading
    game._smoothed_camera_pose = None
    game._camera_track_key = None
    return game


class CameraSmoothingTests(unittest.TestCase):
    def test_north_up_camera_smooths_position_without_rotating_scene(self):
        states = torch.tensor([[[10.0, 20.0, 0.8, 0.0, 0.0, 0.0, 1.0]]])
        game = make_game(states)

        self.assertEqual(game._camera_pose(), (10.0, 20.0, 0.0))
        states[0, 0, 0] = 14.0
        states[0, 0, 1] = 16.0
        states[0, 0, 2] = -1.2

        x, y, yaw = game._camera_pose()
        self.assertAlmostEqual(x, 10.0 + 4.0 * game.CAMERA_POSITION_ALPHA)
        self.assertAlmostEqual(y, 20.0 - 4.0 * game.CAMERA_POSITION_ALPHA)
        self.assertEqual(yaw, 0.0)

    def test_heading_follow_uses_shortest_angle_and_smooths_rotation(self):
        states = torch.tensor(
            [[[0.0, 0.0, math.radians(179.0), 0.0, 0.0, 0.0, 1.0]]]
        )
        game = make_game(states, follows_heading=True)
        initial_yaw = game._camera_pose()[2]
        states[0, 0, 2] = math.radians(-179.0)

        smoothed_yaw = game._camera_pose()[2]
        moved = math.atan2(
            math.sin(smoothed_yaw - initial_yaw),
            math.cos(smoothed_yaw - initial_yaw),
        )
        self.assertGreater(moved, 0.0)
        self.assertLess(moved, math.radians(1.0))

    def test_switching_observed_car_snaps_to_new_target(self):
        states = torch.tensor(
            [[
                [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [80.0, 90.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            ]]
        )
        game = make_game(states)
        game._camera_pose()
        game.selected_agent = 1

        self.assertEqual(game._camera_pose(), (80.0, 90.0, 0.0))


if __name__ == "__main__":
    unittest.main()
