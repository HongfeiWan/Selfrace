import math
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from adaptive_batch import AdaptiveBatchSizer
from ddppo import (
    RolloutTensorBuffer,
    adapt_num_envs_to_memory,
    average_gradients_across_ranks,
    load_checkpoint,
    reconcile_lr_scheduler_horizon,
    save_checkpoint,
)


class TrainingRuntimeTests(unittest.TestCase):
    def test_checkpoint_round_trip_restores_update_runtime(self):
        torch.manual_seed(17)
        random.seed(17)
        model = torch.nn.Linear(3, 2)
        policy_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        value_optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
        policy_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(policy_optimizer, T_max=10)
        value_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(value_optimizer, T_max=10)
        loss = model(torch.ones(2, 3)).sum()
        loss.backward()
        policy_optimizer.step()
        value_optimizer.step()
        policy_scheduler.step()
        value_scheduler.step()
        sizers = {
            "ppo": AdaptiveBatchSizer(32, minimum=2, initial=8),
            "feature": AdaptiveBatchSizer(32, minimum=2, initial=16),
            "rollout": AdaptiveBatchSizer(64, minimum=4, initial=32),
        }
        sizers["ppo"].backoff(8)
        saved_weight = model.weight.detach().clone()

        with tempfile.TemporaryDirectory() as directory:
            path = save_checkpoint(
                model,
                policy_optimizer,
                value_optimizer,
                policy_scheduler,
                value_scheduler,
                None,
                sizers,
                {
                    "update_step": 7,
                    "environment_steps": 896,
                    "rank0_completed_world_episodes": 4,
                },
                torch.tensor(2.5),
                directory,
                torch.device("cpu"),
            )
            expected_torch_random = torch.rand(4)
            expected_python_random = random.random()
            with torch.no_grad():
                model.weight.zero_()
            sizers["ppo"].current = 32
            torch.manual_seed(99)
            random.seed(99)

            progress = load_checkpoint(
                model,
                policy_optimizer,
                value_optimizer,
                policy_scheduler,
                value_scheduler,
                None,
                sizers,
                path,
                torch.device("cpu"),
            )

            self.assertTrue(torch.equal(model.weight, saved_weight))
            self.assertEqual(progress["update_step"], 7)
            self.assertEqual(float(progress["a_max_ewma"]), 2.5)
            self.assertEqual(sizers["ppo"].current, 4)
            self.assertTrue(torch.equal(torch.rand(4), expected_torch_random))
            self.assertEqual(random.random(), expected_python_random)
            self.assertTrue((Path(directory) / "latest.pt").is_file())

    def test_resume_reconciles_cosine_scheduler_to_new_final_update(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.Adam([parameter], lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        for _ in range(4):
            optimizer.step()
            scheduler.step()

        reconcile_lr_scheduler_horizon(scheduler, 20)

        expected = 1e-3 * (1.0 + math.cos(math.pi * 4 / 20)) / 2.0
        self.assertEqual(scheduler.T_max, 20)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], expected)
        self.assertAlmostEqual(scheduler.get_last_lr()[0], expected)

    def test_rollout_buffer_keeps_per_world_time_indices(self):
        buffer = RolloutTensorBuffer(2)
        state = torch.zeros(3, 2, 10)
        route = {
            "route_quad_ids": torch.zeros(3, 2, 4, dtype=torch.int32),
            "target_count": torch.ones(3, 2, dtype=torch.long),
            "current_idx": torch.zeros(3, 2, dtype=torch.long),
        }
        buffer.write_pre_step(state, route, time_index=torch.tensor([0, 8, 16]))
        zeros = torch.zeros(3, 2)
        buffer.write_post_step(zeros, zeros.bool(), zeros, zeros, zeros.long())

        self.assertEqual(tuple(buffer.time_indices.shape), (2, 3))
        self.assertEqual(buffer.time_indices[0].tolist(), [0, 8, 16])

    def test_memory_budget_reduces_worlds_without_touching_cpu_config(self):
        config = {
            "simulator": {"num_envs": 2600, "max_agents_num": 150},
            "training": {
                "rollout_length": 128,
                "memory_adaptation": {
                    "enabled": True,
                    "target_fraction": 0.8,
                    "reserve_mb": 512,
                    "min_num_envs": 8,
                },
            },
        }
        with mock.patch("torch.cuda.mem_get_info", return_value=(2 * 1024**3, 4 * 1024**3)):
            effective = adapt_num_envs_to_memory(config, torch.device("cuda"))

        self.assertGreaterEqual(effective, 8)
        self.assertLess(effective, 2600)
        self.assertEqual(config["simulator"]["num_envs"], effective)

    def test_gradient_reduction_uses_buckets_and_preserves_average(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        calls = []

        def fake_all_reduce(tensor, op=None):
            calls.append(tensor.numel())
            tensor.mul_(2)

        with (
            mock.patch.object(torch.distributed, "is_available", return_value=True),
            mock.patch.object(torch.distributed, "is_initialized", return_value=True),
            mock.patch.object(torch.distributed, "all_reduce", side_effect=fake_all_reduce),
        ):
            bucket_count = average_gradients_across_ranks(
                model, local_samples=16, device=torch.device("cpu"), bucket_cap_mb=0.00005
            )

        self.assertEqual(len(calls), bucket_count + 1)
        self.assertLess(bucket_count, sum(1 for _ in model.parameters()))
        for parameter in model.parameters():
            self.assertTrue(torch.equal(parameter.grad, torch.ones_like(parameter.grad)))


if __name__ == "__main__":
    unittest.main()
