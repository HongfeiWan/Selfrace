import ast
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "simulator"))


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class CollisionStreamingTests(unittest.TestCase):
    @staticmethod
    def make_checker(pair_budget):
        from collision import CollisionChecker

        spatial_hash = SimpleNamespace(device="cpu", cell_size=10.0)
        return CollisionChecker(
            {
                "simulator": {
                    "collision_stream_pair_budget": pair_budget,
                    "collision_stream_compile": False,
                }
            },
            spatial_hash,
        )

    @staticmethod
    def make_states():
        import torch

        generator = torch.Generator().manual_seed(31)
        states_t0 = torch.zeros(4, 7, 7)
        states_t0[..., :2] = torch.randn(4, 7, 2, generator=generator) * 20.0
        states_t0[..., 2] = torch.randn(4, 7, generator=generator) * 0.2
        states_t0[..., 4] = 4.5
        states_t0[..., 5] = 2.0
        states_t0[..., 6] = 1.0
        states_t1 = states_t0.clone()
        states_t1[..., :2] += torch.randn(4, 7, 2, generator=generator) * 0.5

        # One certain active collision and one overlapping inactive vehicle.
        states_t0[0, 0, :2] = torch.tensor([0.0, 0.0])
        states_t0[0, 1, :2] = torch.tensor([1.0, 0.0])
        states_t0[0, 2, :2] = torch.tensor([0.5, 0.0])
        states_t1[0, :3, :2] = states_t0[0, :3, :2]
        states_t1[0, 2, 6] = 0.0
        return states_t0, states_t1

    def test_small_pair_budget_matches_single_block_and_is_hard_bounded(self):
        import torch

        states_t0, states_t1 = self.make_states()
        reference = self.make_checker(pair_budget=10_000)
        streamed = self.make_checker(pair_budget=5)
        block_sizes = []
        run_chunk = streamed._run_stream_chunk

        def recording_run_chunk(*args):
            block_sizes.append(args[0].shape[0] * args[5].numel())
            return run_chunk(*args)

        streamed._run_stream_chunk = recording_run_chunk
        expected = reference.check(states_t0, states_t1)
        actual = streamed.check(states_t0, states_t1)

        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(actual[0, 0])
        self.assertTrue(actual[0, 1])
        self.assertFalse(actual[0, 2])
        self.assertTrue(block_sizes)
        self.assertLessEqual(max(block_sizes), streamed.stream_pair_budget)

    def test_sparse_and_sync_free_dense_chunk_agree(self):
        import torch
        from collision import (
            _bounded_collision_world_chunk,
            _bounded_collision_world_chunk_dense,
        )

        checker = self.make_checker(pair_budget=100)
        states_t0, states_t1 = self.make_states()
        active = states_t1[..., 6] > 0.5
        verts_t0 = checker._get_world_vertices(states_t0)
        verts_t1 = checker._get_world_vertices(states_t1)
        pair_i, pair_j = checker._pair_indices(states_t0.shape[1])

        sparse = _bounded_collision_world_chunk(
            states_t0, states_t1, active, verts_t0, verts_t1, pair_i, pair_j
        )
        dense = _bounded_collision_world_chunk_dense(
            states_t0, states_t1, active, verts_t0, verts_t1, pair_i, pair_j
        )
        self.assertTrue(torch.equal(sparse, dense))

    def test_python_collision_hot_path_has_no_tensor_scalar_reads(self):
        tree = ast.parse(
            (PROJECT_ROOT / "simulator" / "collision.py").read_text(encoding="utf-8")
        )
        checker = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "CollisionChecker"
        )
        hot_methods = {
            "check",
            "_streaming_dynamic_collisions",
            "_run_stream_chunk",
        }
        calls = [
            node
            for method in checker.body
            if isinstance(method, ast.FunctionDef) and method.name in hot_methods
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "item"
        ]
        self.assertEqual(calls, [])

    def test_stationary_crossing_rectangles_without_inside_corners(self):
        import torch
        checker = self.make_checker(64)
        states = torch.tensor([[[0., 0., 0., 0., 6., 1., 1.],
                                [0., 0., torch.pi / 2, 0., 6., 1., 1.]]])
        self.assertTrue(checker.check(states, states).all())

    def test_static_rectangles_have_real_dimensions_and_bounded_tiles(self):
        import torch
        checker = self.make_checker(2)
        states = torch.tensor([[[0., 0., 0., 0., 4., 2., 1.],
                                [20., 0., 0., 0., 4., 2., 1.]]])
        obstacle_states = torch.tensor([[[0., 0., torch.pi / 4, 0., 4., 2., 1.],
                                         [40., 0., 0., 0., 4., 2., 1.]]])
        obstacles = checker._get_world_vertices(obstacle_states)[0]
        self.assertEqual(checker.check(states, states, static_obstacles=obstacles).tolist(), [[True, False]])

    def test_cuda_dynamic_cells_match_reference_with_overflow_and_crowding(self):
        import torch
        if not torch.cuda.is_available() or not importlib.util.find_spec('triton'):
            self.skipTest('CUDA and Triton are required')
        from collision_cuda import cell_collisions
        from collision import _bounded_collision_world_chunk_dense
        checker = self.make_checker(128)
        torch.manual_seed(927)
        start = torch.zeros(12, 31, 7, device='cuda')
        start[..., :2] = torch.randn(12, 31, 2, device='cuda') * 30
        start[..., 2] = torch.randn(12, 31, device='cuda') * 3
        start[..., 4:6] = torch.rand(12, 31, 2, device='cuda') * 6 + 1
        start[..., 6] = 1
        # Dense overflow, large vehicle, negative/out-of-encoding coordinates.
        start[0, :, :2] = 0
        start[1, 0, 4] = 90
        start[2, :, :2] += 30_000_000
        end = start.clone()
        end[..., :2] += torch.randn_like(end[..., :2]) * 4
        end[3, 0, 0] += 150
        end[4, 0, 6] = 0
        active = end[..., 6] > .5
        v0, v1 = checker._get_world_vertices(start), checker._get_world_vertices(end)
        pairs = torch.triu_indices(31, 31, 1, device='cuda')
        expected = _bounded_collision_world_chunk_dense(start, end, active, v0, v1, *pairs)
        for cell_size in (5., 16., 32.):
            actual = cell_collisions(start, end, active, v0, v1, cell_size)
            self.assertTrue(torch.equal(expected, actual), f'cell_size={cell_size}, mismatches={(expected != actual).nonzero().tolist()}')


if __name__ == "__main__":
    unittest.main()
