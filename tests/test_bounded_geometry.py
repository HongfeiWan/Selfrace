import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'simulator'), str(ROOT / 'utils')]
from offroad import OffroadChecker
from spatial_hash import SpatialHash
from tensor_budget import nearest_indices
from dynamics import KinematicBicycleModel
from road import RoadNetwork
from observation import ObservationGenerator


def make_road():
    yaw = math.pi / 4
    rotation = torch.tensor([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    vertices = torch.tensor([[-20., -1.5], [20., -1.5], [20., 1.5], [-20., 1.5]]) @ rotation.T
    road = SimpleNamespace(device=torch.device('cpu'), quads_vertices=vertices[None])
    spatial = SpatialHash(5, vertices.amin(0) - 1, vertices.amax(0) + 1, road.device)
    return road, spatial


def test_rotated_car_fits_rotated_road_and_chunking_agrees():
    road, spatial = make_road()
    checker = OffroadChecker(road, spatial, workspace_mb=0.005)
    states = torch.tensor([[0., 0., math.pi / 4, 4., 2.], [0., 4., math.pi / 4, 4., 2.]])
    actual = checker.check_on_road(states)
    assert actual.tolist() == [True, False]
    points = checker._get_discretized_bounding_boxes(states[:1])
    assert torch.allclose(points[0, -1], torch.zeros(2))
    checker.offroad_point_chunk_size = 1
    assert torch.equal(actual, checker.check_on_road(states))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_offroad_box_oom_shrinks_and_remembers_vehicle_tile():
    road, _ = make_road()
    road.device = torch.device('cuda')
    road.quads_vertices = road.quads_vertices.cuda()
    spatial = SpatialHash(5, road.quads_vertices.amin((0, 1)) - 1,
                          road.quads_vertices.amax((0, 1)) + 1, road.device)
    checker = OffroadChecker(road, spatial, workspace_mb=.02)
    states = torch.tensor([[0., 0., math.pi / 4, 4., 2.]], device='cuda').repeat(20, 1)
    original = checker._get_discretized_bounding_boxes
    calls = []

    def limited_boxes(chunk):
        calls.append(chunk.shape[0])
        if chunk.shape[0] > 2:
            raise torch.cuda.OutOfMemoryError('injected box workspace pressure')
        return original(chunk)

    with mock.patch.object(checker, '_get_discretized_bounding_boxes', side_effect=limited_boxes):
        assert checker.check_on_road(states).all()
        assert max(calls) > 2
        calls.clear()
        assert checker.check_on_road(states).all()
        assert max(calls) <= 2


def test_csr_candidates_match_bounding_boxes_and_tiles():
    road, spatial = make_road()
    OffroadChecker(road, spatial)
    queries = torch.tensor([[0., 0.], [10., 10.]])
    ids, mask = spatial.query_points_padded(queries)
    tiled, valid = spatial.query_points_padded(queries, 0, 1)
    assert torch.equal(ids, tiled)
    assert torch.equal(mask, valid)
    assert spatial.static_cell_items.numel() == 0


def test_tiled_knn_is_exact_and_never_builds_full_matrix():
    torch.manual_seed(12)
    queries, reference = torch.randn(57, 2), torch.randn(2300, 2)
    expected = torch.cdist(queries, reference, compute_mode='donot_use_mm_for_euclid_dist').topk(7, largest=False).indices
    sizes = []
    cdist = torch.cdist
    def record(a, b, **kwargs):
        sizes.append((a.shape[0], b.shape[0]))
        return cdist(a, b, **kwargs)
    with mock.patch('torch.cdist', side_effect=record):
        actual = nearest_indices(queries, reference, 7, budget_bytes=32768)
    assert torch.equal(actual, expected)
    assert max(n * m for n, m in sizes) <= 2048
    assert nearest_indices(queries, reference[:3], 7)[:, 3:].eq(-1).all()


def test_sparse_dynamics_reuses_original_slot_history():
    torch.manual_seed(51)
    dense = KinematicBicycleModel({}, torch.device('cpu'))
    packed = KinematicBicycleModel({}, torch.device('cpu'))
    states = torch.zeros(2, 5, 7)
    states[..., 3:5] = 4
    style = torch.ones(2, 5, 4)
    for alive in (torch.tensor([0, 2, 5, 6, 8]), torch.tensor([2, 6]), torch.tensor([2])):
        actions = torch.randint(0, 12, (2, 5))
        mask = torch.zeros(10, dtype=torch.bool)
        mask[alive] = True
        expected = dense.step(states[..., :4].reshape(-1, 4), actions.flatten(), .3, style.reshape(-1, 4), mask, states[..., 4].flatten())
        proposal, controls = packed.propose_selected(states, actions, .3, alive, style)
        assert torch.allclose(proposal, expected[alive], atol=1e-6)
        packed.commit_selected(alive, controls)
        for name in ('current_along', 'current_alat', 'current_steering_angle'):
            assert torch.allclose(getattr(dense, name), getattr(packed, name))
        states.reshape(-1, 7)[alive, :4] = proposal


def test_selected_neighbors_read_shared_worlds_without_world_copy():
    generator = ObservationGenerator.__new__(ObservationGenerator)
    generator.num_neighbors, generator.horizon = 3, 200.
    generator.device, generator.workspace_bytes = torch.device('cpu'), 512
    torch.manual_seed(44)
    states = torch.randn(4, 7, 7)
    states[..., 6] = 1
    b, m = torch.tensor([3, 0, 3, 1]), torch.tensor([2, 0, 4, 5])
    ego = states[b, m, None]
    actual = generator._get_selected_neighbors(states, m, ego, b)
    expected = generator._get_nearest_neighbors(states)[b, m, None]
    assert torch.allclose(actual, expected)


def test_shared_frenet_sign_contract():
    road = RoadNetwork.__new__(RoadNetwork)
    road.quad_directions = torch.tensor([[1., 0.]])
    road.quad_centerlines = torch.tensor([[[0., 0.], [10., 0.]]])
    d, theta = road.frenet_from_indices(torch.tensor([[0., 2.]]), torch.tensor([.5]), torch.tensor([0]))
    assert torch.allclose(d, torch.tensor([-2.]))
    assert torch.allclose(theta, torch.tensor([-.5]))


def test_stop_line_normal_is_perpendicular():
    from simulator import TeraflowSimulator
    sim = TeraflowSimulator.__new__(TeraflowSimulator)
    sim.device = torch.device('cpu')
    sim.stop_line_width = 3.5
    sim.road_network = SimpleNamespace(
        stop_line_centers=torch.zeros(1, 2), stop_line_yaws=torch.tensor([math.pi / 4]),
        stop_line_control_indices=torch.tensor([0]), traffic_light_locations=torch.zeros(1, 2),
    )
    sim._prepare_traffic_controls()
    tangent = torch.tensor([math.cos(math.pi / 4), math.sin(math.pi / 4)])
    assert abs(float((sim.stop_line_segments[0, 1] - sim.stop_line_segments[0, 0]) @ tangent)) < 1e-6
