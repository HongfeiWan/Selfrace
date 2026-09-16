import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "training"))


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class NetworkSchemaTests(unittest.TestCase):
    def test_set_encoder_handles_an_all_invalid_mask_without_host_branch(self):
        import torch
        from network import PermutationInvariantEncoder

        encoder = PermutationInvariantEncoder(6, output_dim=4, element_dim=2).eval()
        values = torch.randn(2, 3, 3, 2)
        mask = torch.zeros(2, 3, 3, dtype=torch.bool)

        with torch.inference_mode():
            encoded = encoder(values, mask)

        self.assertEqual(tuple(encoded.shape), (2, 3, 4))
        self.assertTrue(torch.equal(encoded, torch.zeros_like(encoded)))

    def test_independent_network_uses_schema_without_changing_module_keys(self):
        import torch
        from network import create_network

        ns = SimpleNamespace
        config = ns(
            simulator=ns(
                observation=ns(
                    local_state_dim=13,
                    num_w_boundaries=2,
                    boundary_feature_dim=2,
                    num_w_lanes=2,
                    waypoint_feature_dim=5,
                    num_neighbors=2,
                    neighbor_feature_dim=10,
                ),
                traffic=ns(stop_line_observation_count=1),
            ),
            training=ns(
                weight_init=ns(type="orthogonal", gain=1.0, bias_zero=True),
                network=ns(
                    encoder_dim=8,
                    network_dim=16,
                    num_actions=12,
                    feature_schema=ns(
                        simple=ns(state=13, goal=8, reward=12, vehicle_style=4),
                        sets=ns(
                            road_boundary=ns(flat_dim=4, element_dim=2),
                            lane_points=ns(flat_dim=14, element_dim=7),
                            stop_lines=ns(flat_dim=4, element_dim=2),
                            other_agents=ns(flat_dim=20, element_dim=10, active_channel=-1),
                        ),
                    ),
                ),
            ),
        )
        torch.manual_seed(7)
        model = create_network(config, network_type="independent").eval()
        width = model.policy_feature_encoder.total_input_dim
        features = torch.zeros(2, 3, width)

        with torch.inference_mode():
            logits, values = model(features, mode="both")

        self.assertEqual(tuple(logits.shape), (2, 3, 12))
        self.assertEqual(tuple(values.shape), (2, 3))
        keys = tuple(model.state_dict())
        self.assertIn("policy_feature_encoder.simple_encoders.0.mlp.0.weight", keys)
        self.assertIn("value_feature_encoder.permutation_encoders.3.element_encoder.2.bias", keys)


if __name__ == "__main__":
    unittest.main()
