"""Real-map regression of rollout reconstruction, sparse step, PPO and resets."""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'training'), str(ROOT / 'simulator'), str(ROOT / 'utils')]
from simulator import TeraflowSimulator
from ddppo import (FeatureBuildWorkspace, RolloutTensorBuffer, DistributedContext,
                   build_features_for_selected_agents, current_control_state,
                   gather_current_route_state_selected, gather_condition_state_selected,
                   snapshot_condition_state, preallocate_rollout_buffer, run_training_loop)
from network import create_network
from adaptive_batch import AdaptiveBatchSizer


def namespace(value):
    return SimpleNamespace(**{k: namespace(v) for k, v in value.items()}) if isinstance(value, dict) else value


def small_config():
    config = yaml.safe_load((ROOT / 'configs/default_config.yaml').read_text(encoding='utf-8'))
    sim, train = config['simulator'], config['training']
    sim.update(num_envs=2, max_agents_num=6, num_npc_vehicles=6,
               map_path=str(ROOT / 'maps/processed_map_Town01_stitched.json'),
               init_candidate_pool_size=128, init_candidate_pool_refill_threshold=16,
               init_candidate_pool_refill_batch=128, init_distribution_correction=False,
               geometry_workspace_mb=2)
    sim['execution']['compile'] = False
    train.update(rollout_length=2, max_episode_length=4, batch_size=8,
                 batch_size_per_gpu=8, min_ppo_samples=1, ppo_epochs=1,
                 precision='32-bit', total_updates=1, log_interval=2)
    train['network'].update(encoder_dim=8, network_dim=32)
    train['network']['compile']['enabled'] = False
    return config


@pytest.fixture(scope='module')
def simulator():
    torch.manual_seed(121)
    sim = TeraflowSimulator(small_config(), torch.device('cpu'))
    sim.reset(return_observation=False)
    return sim


def test_shared_world_features_match_materialized_world_reference(simulator):
    sim = simulator
    config = namespace(sim.config)
    _, b, m, alive = sim.active_selection()
    b, m = b[:4], m[:4]
    assert b.numel() > 0
    controls = current_control_state(sim)
    conditions = gather_condition_state_selected(snapshot_condition_state(sim), b, m)
    routes = gather_current_route_state_selected(sim.get_route_state(clone=False), b, m)
    reference_worlds = torch.cat((sim.agents_state[b], controls[b]), dim=-1)
    expected = build_features_for_selected_agents(reference_worlds, sim, config, routes, conditions, m,
        env_indices=b, time_indices=3, workspace=FeatureBuildWorkspace(config)).clone()
    actual = build_features_for_selected_agents(sim.agents_state, sim, config, routes, conditions, m,
        env_indices=b, time_indices=3, workspace=FeatureBuildWorkspace(config),
        world_indices=b, control_state=controls, alive_mask=alive)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_step_geometry_reward_and_masked_reset(simulator):
    sim = simulator
    for _ in range(3):
        _, b, m, alive = sim.active_selection()
        actions = torch.randint(0, 12, sim.agents_state.shape[:2])
        old = sim.agents_state.clone()
        intermediate = sim._current_target_is_intermediate().clone()
        goals = sim.goal_positions.clone()
        reward, done = sim.step(actions, return_observation=False)
        assert torch.equal(sim.agents_state[~alive], old[~alive])
        d, theta = sim.road_network.calculate_frenet_coordinates(sim.agents_state[..., :2], sim.agents_state[..., 2], sim.spatial_hash)
        assert torch.allclose(sim._geometry[1][b, m], d[b, m])
        assert torch.allclose(sim._geometry[2][b, m], theta[b, m])
        control = sim._current_control_state()
        # Newly done slots' controls are masked in observations, but reward uses
        # the just-proposed controls (as did the dense implementation).
        along = sim.dynamics_model.current_along.reshape(2, 6)
        alat = sim.dynamics_model.current_alat.reshape(2, 6)
        jerk = sim.dynamics_model.discrete_action_space.get_action(actions)
        extended = torch.cat((sim.agents_state[..., :4], along[..., None], alat[..., None], jerk,
                              theta[..., None], d[..., None]), -1)
        info = sim.last_step_train_info
        expected, _ = sim.reward_calculator.calculate(extended, info['collision_mask'], info['offroad_mask'],
            sim.dt, goals, intermediate, sim.stop_line_violation)
        assert torch.allclose(reward, expected * alive, atol=1e-6)
        assert torch.isfinite(reward).all()
    untouched = sim.agents_state[1].clone()
    sim.reset_worlds(torch.tensor([True, False]))
    assert torch.equal(sim.agents_state[1], untouched)
    assert sim._geometry is None and sim._active_selection is None


def test_preallocated_buffer_reused_through_real_ppo(simulator, tmp_path):
    _check_training_update(simulator, tmp_path)


def test_route_id_storage_uses_map_capacity_not_first_random_targets(simulator):
    sim = copy.copy(simulator)
    sim.road_network = copy.copy(sim.road_network)
    sim.road_network.num_quads = 40001
    buffer = RolloutTensorBuffer(2)
    alive = sim.agents_state[..., 6].bool()
    first = torch.zeros((*alive.shape, 4), dtype=torch.int32)
    buffer.write_pre_step_from_simulator(sim, alive, {'route_quad_ids': first})
    assert buffer.route_state['route_quad_ids'].dtype == torch.int32
    # A later legal target must not silently wrap into negative padding IDs.
    buffer.route_state['route_quad_ids'][1].fill_(40000)
    assert buffer.route_state['route_quad_ids'][1].eq(40000).all()


def _check_training_update(simulator, tmp_path):
    sim = simulator
    sim.reset(return_observation=False)
    config_dict = copy.deepcopy(sim.config)
    config_dict['training']['checkpoint_dir'] = str(tmp_path)
    config = namespace(config_dict)
    model = create_network(config).to(sim.device)
    policy = torch.optim.Adam(model.policy_parameters(), lr=1e-3)
    value = torch.optim.Adam(model.value_parameters(), lr=1e-3)
    policy_sched = torch.optim.lr_scheduler.CosineAnnealingLR(policy, 2)
    value_sched = torch.optim.lr_scheduler.CosineAnnealingLR(value, 2)
    sizers = {key: AdaptiveBatchSizer(8, minimum=1) for key in ('ppo', 'rollout', 'feature')}
    buffer = preallocate_rollout_buffer(sim, 2)
    pointer = buffer.states.data_ptr()
    old_weight = model.policy_network[0].weight.detach().clone()
    result = run_training_loop(DistributedContext(0, 1, sim.device), model, sim, config,
        policy, value, policy_sched, value_sched, {'update_step': 0}, 1, None, sizers, buffer)
    assert buffer.states.data_ptr() == pointer
    assert len(buffer) == 2  # a truthiness replacement would leave this at zero
    assert result['update_step'] == 1
    assert not torch.equal(old_weight, model.policy_network[0].weight)
    assert (tmp_path / 'latest.pt').is_file()
    return model


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_compiled_graph_pipeline_completes_real_ppo(tmp_path):
    config = small_config()
    config['simulator']['execution'].update(compile=True, chunk_agents=4)
    config['simulator']['execution']['cuda_graph']['enabled'] = True
    config['training']['network']['compile'].update(enabled=True, min_agents=1)
    config['training']['network']['cuda_graph'].update(enabled=True, bucket_agents=2)
    sim = TeraflowSimulator(config, torch.device('cuda'))
    model = _check_training_update(sim, tmp_path)
    assert model._inference_graph.graph is not None, model._inference_graph.disabled_reason
    assert not sim.dynamics_model._selected_stage.function.failed
    assert not sim.offroad_checker._inside_stage.failed
    assert not sim.collision_checker._cell_failed


def test_failed_geometric_proposal_does_not_advance_world_or_controls(simulator):
    from unittest import mock
    sim = simulator
    sim.reset(return_observation=False)
    sim.dynamics_model._ensure_control_storage(sim.agents_state.shape[0] * sim.agents_state.shape[1], torch.float32)
    sim.dynamics_model.current_along.fill_(.75)
    states = sim.agents_state.clone()
    history = sim.dynamics_model.current_along.clone()
    with mock.patch.object(sim.offroad_checker, 'check_on_road', side_effect=RuntimeError('test geometry failure')):
        with pytest.raises(RuntimeError, match='test geometry failure'):
            sim.step(torch.zeros(sim.agents_state.shape[:2], dtype=torch.long), return_observation=False)
    assert torch.equal(states, sim.agents_state)
    assert torch.equal(history, sim.dynamics_model.current_along)


def test_traffic_phase_snapshot_and_optional_inputs(simulator):
    from feature_schema import FeatureGroup
    sim = copy.copy(simulator)
    sim.road_network = copy.copy(sim.road_network)
    sim.stop_line_segments = torch.tensor([[[4., -2.], [4., 2.]]])
    sim.stop_line_control_indices = torch.tensor([0])
    sim.road_network.crosswalks = torch.tensor([[8., 0., 0., 4., 2.]])
    sim.traffic_light_states = torch.tensor([[1], [2]], dtype=torch.int8)
    sim.last_done = None
    sim.agents_state = torch.tensor([[[0., 0., 0., 0., 4., 2., 1.]],
                                    [[0., 0., 0., 0., 4., 2., 1.]]])
    groups = (FeatureGroup('traffic_lights', 16, 8, -1), FeatureGroup('crosswalks', 14, 7, -1))
    features = sim.traffic_input_sets(sim.agents_state, sim.traffic_light_states, groups)
    assert features['traffic_lights'][0, 0, 0, 4:8].tolist() == [1., 0., 0., 1.]
    assert features['traffic_lights'][1, 0, 0, 4:8].tolist() == [0., 1., 0., 1.]
    assert features['traffic_lights'][:, :, 1].eq(0).all()
    assert features['crosswalks'][0, 0, 0, -1] == 1
    stop_lines = sim._compute_stop_line_observation(sim.agents_state, sim.traffic_light_states)
    assert torch.equal(stop_lines[0, 0, :4], torch.tensor([4., -2., 4., 2.]))
    assert stop_lines[1].eq(0).all()  # yellow is valid traffic input, not a red stop line
    buffer = RolloutTensorBuffer(2)
    buffer.write_pre_step_from_simulator(sim, sim.agents_state[..., 6].bool(), {})
    sim.traffic_light_states.fill_(3)
    assert buffer.traffic_light_states[0].flatten().tolist() == [1, 2]
