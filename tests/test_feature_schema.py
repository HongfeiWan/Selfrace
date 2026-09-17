import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from feature_schema import FEATURE_PAD_VALUE, FeatureSchema


def namespace(**values):
    return SimpleNamespace(**values)


def make_config(*, boundary_dim=160, neighbor_element_dim=10, goal_dim=8, active_channel=-1):
    schema = namespace(
        simple=namespace(state=13, goal=goal_dim, reward=12, vehicle_style=4),
        sets=namespace(
            road_boundary=namespace(flat_dim=boundary_dim, element_dim=2),
            lane_points=namespace(flat_dim=560, element_dim=7),
            stop_lines=namespace(flat_dim=20, element_dim=2),
            other_agents=namespace(
                flat_dim=200,
                element_dim=neighbor_element_dim,
                active_channel=active_channel,
            ),
        ),
    )
    observation = namespace(
        local_state_dim=13,
        num_w_boundaries=80,
        boundary_feature_dim=2,
        num_w_lanes=80,
        waypoint_feature_dim=5,
        num_neighbors=20,
        neighbor_feature_dim=10,
    )
    return namespace(
        simulator=namespace(
            observation=observation,
            traffic=namespace(stop_line_observation_count=5),
        ),
        training=namespace(network=namespace(feature_schema=schema)),
    )


class FeatureSchemaTests(unittest.TestCase):
    def test_named_schema_has_stable_offsets(self):
        schema = FeatureSchema.from_config(make_config())

        self.assertEqual(schema.total_input_dim, 977)
        self.assertEqual(schema.flat_slice("state"), slice(0, 13))
        self.assertEqual(schema.flat_slice("goal"), slice(13, 21))
        self.assertEqual(schema.flat_slice("reward"), slice(21, 33))
        self.assertEqual(schema.flat_slice("vehicle_style"), slice(33, 37))
        self.assertEqual(schema.flat_slice("road_boundary"), slice(37, 197))
        self.assertEqual(schema.flat_slice("lane_points"), slice(197, 757))
        self.assertEqual(schema.flat_slice("stop_lines"), slice(757, 777))
        self.assertEqual(schema.flat_slice("other_agents"), slice(777, 977))
        self.assertIs(schema.flat_slice("other_agents"), schema.flat_slice("other_agents"))
        self.assertIs(schema.group("other_agents"), schema.group("other_agents"))
        self.assertEqual(schema.group("other_agents").resolved_active_channel, 9)
        self.assertEqual(FEATURE_PAD_VALUE, -2.0)

    def test_schema_rejects_simulator_dimension_drift(self):
        with self.assertRaisesRegex(ValueError, "road_boundary"):
            FeatureSchema.from_config(make_config(boundary_dim=158))

        with self.assertRaisesRegex(ValueError, "other_agents.element_dim"):
            FeatureSchema.from_config(make_config(neighbor_element_dim=5))

        with self.assertRaisesRegex(ValueError, "goal"):
            FeatureSchema.from_config(make_config(goal_dim=6))

        with self.assertRaisesRegex(ValueError, "active_channel"):
            FeatureSchema.from_config(make_config(active_channel=8))

    def test_missing_named_schema_is_rejected(self):
        config = make_config()
        config.training.network = namespace(
            simple_feature_dims=[13, 8, 12, 4],
            permutation_feature_dims=[160, 560, 20, 200],
        )
        with self.assertRaisesRegex(ValueError, "feature_schema is required"):
            FeatureSchema.from_config(config)


if __name__ == "__main__":
    unittest.main()
