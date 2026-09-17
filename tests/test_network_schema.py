import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "training"))


def make_config(*, compile_enabled=False):
    ns = SimpleNamespace
    return ns(
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
                compile=ns(
                    enabled=compile_enabled,
                    mode="default",
                    dynamic=True,
                    fullgraph=False,
                    min_agents=1,
                ),
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

        config = make_config()
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

    def test_forward_both_prepares_features_and_set_masks_once(self):
        import torch
        from network import create_network

        model = create_network(make_config()).eval()
        width = model.policy_feature_encoder.total_input_dim
        features = torch.zeros(4, 1, width)
        policy_prepare = model.policy_feature_encoder.prepare
        value_prepare = model.value_feature_encoder.prepare

        with (
            mock.patch.object(
                model.policy_feature_encoder, "prepare", wraps=policy_prepare
            ) as prepare_policy,
            mock.patch.object(
                model.value_feature_encoder, "prepare", wraps=value_prepare
            ) as prepare_value,
            torch.inference_mode(),
        ):
            logits, values = model.forward_both(features)

        self.assertEqual(prepare_policy.call_count, 1)
        self.assertEqual(prepare_value.call_count, 0)
        self.assertEqual(tuple(logits.shape), (4, 1, 12))
        self.assertEqual(tuple(values.shape), (4, 1))

    def test_reused_set_compaction_matches_dense_masked_encoding(self):
        import torch
        from feature_schema import FEATURE_PAD_VALUE
        from network import create_network

        model = create_network(make_config()).eval()
        encoder = model.policy_feature_encoder
        features = torch.randn(3, 2, encoder.total_input_dim)
        for set_index, (feature_slice, element_dim, active_channel) in enumerate(
            zip(
                encoder.set_slices,
                encoder.set_element_dims,
                encoder.set_active_channels,
            )
        ):
            elements = features[:, :, feature_slice].view(3, 2, -1, element_dim)
            positions = torch.arange(elements.shape[2]).view(1, 1, -1)
            mask = (
                torch.zeros_like(positions, dtype=torch.bool)
                if set_index == 2
                else (positions + set_index) % 2 == 0
            ).expand(3, 2, -1)
            if active_channel is None:
                elements.masked_fill_(~mask.unsqueeze(-1), FEATURE_PAD_VALUE)
            else:
                elements[..., active_channel] = mask.to(elements.dtype)

        prepared = encoder.prepare(features)
        with torch.inference_mode():
            compact = encoder.forward_prepared(
                prepared.features,
                prepared.set_masks,
                prepared.set_valid_indices,
                prepared.set_group_indices,
                prepared.set_nonempty,
            )
            dense_parts = [
                simple_encoder(prepared.features[:, :, feature_slice])
                for feature_slice, simple_encoder in zip(
                    encoder.simple_slices, encoder.simple_encoders
                )
            ]
            dense_parts.extend(
                set_encoder(
                    prepared.features[:, :, feature_slice], mask=set_mask
                )
                for feature_slice, set_mask, set_encoder in zip(
                    encoder.set_slices,
                    prepared.set_masks,
                    encoder.permutation_encoders,
                )
            )
            dense = torch.cat(dense_parts, dim=-1)

        torch.testing.assert_close(compact, dense)
        for mask, valid, groups in zip(
            prepared.set_masks,
            prepared.set_valid_indices,
            prepared.set_group_indices,
        ):
            expected_valid = torch.nonzero(
                mask.reshape(-1), as_tuple=False
            ).squeeze(-1)
            self.assertTrue(torch.equal(valid, expected_valid))
            self.assertEqual(groups.shape, valid.shape)

    def test_actor_and_critic_are_disjoint_and_sequential_gradients_match(self):
        import torch
        from network import create_network

        torch.manual_seed(19)
        combined = create_network(make_config()).train()
        sequential = copy.deepcopy(combined).train()
        width = combined.policy_feature_encoder.total_input_dim
        features = torch.randn(5, 1, width)

        combined_logits, combined_values = combined.forward_both(features)
        (combined_logits.square().mean() + 0.5 * combined_values.square().mean()).backward()

        prepared = sequential.prepare_features(features)
        sequential.forward_prepared_policy(prepared).square().mean().backward()
        (0.5 * sequential.forward_prepared_value(prepared).square().mean()).backward()

        policy_ids = {id(parameter) for parameter in sequential.policy_parameters()}
        value_ids = {id(parameter) for parameter in sequential.value_parameters()}
        self.assertTrue(policy_ids)
        self.assertTrue(value_ids)
        self.assertTrue(policy_ids.isdisjoint(value_ids))
        for (name_a, parameter_a), (name_b, parameter_b) in zip(
            combined.named_parameters(), sequential.named_parameters()
        ):
            self.assertEqual(name_a, name_b)
            torch.testing.assert_close(parameter_a.grad, parameter_b.grad)


if __name__ == "__main__":
    unittest.main()
