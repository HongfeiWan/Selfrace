import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'utils'), str(ROOT / 'training')]
from execution import CompiledStage, SingleCudaGraph
from test_network_schema import make_config
from network import create_network


def test_cpu_execution_does_not_capture_or_compile():
    stage = CompiledStage(lambda x: x * 3, enabled=True)
    graph = SingleCudaGraph(stage, enabled=True, capacity=2)
    assert torch.equal(graph(torch.ones(2)), torch.full((2,), 3.))
    assert graph.graph is None and stage.compiled is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_graph_replay_owns_outputs_and_reads_updated_weights():
    config = make_config(compile_enabled=False)
    config.training.network.cuda_graph = SimpleNamespace(enabled=True, bucket_agents=8, max_private_mb=64)
    model = create_network(config).cuda().eval()
    reference = copy.deepcopy(model)
    reference._inference_graph.enabled = False
    features = torch.randn(8, 1, model.policy_feature_encoder.total_input_dim, device='cuda')
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        first = model.forward_both(features)
        expected = reference.forward_both(features)
        assert model._inference_graph.graph is not None, model._inference_graph.disabled_reason
        assert all(torch.allclose(a, b, atol=2e-3, rtol=2e-3) for a, b in zip(first, expected))
        saved = tuple(x.clone() for x in first)
        for net in (model, reference):
            net.policy_network[-1].bias.add_(.75)
            net.value_network[-1].bias.sub_(.5)
        second = model.forward_both(features * .5)
        # Disable eager AMP weight caching for the comparison as well.
        with torch.autocast('cuda', cache_enabled=False):
            expected = reference.forward_both(features * .5)
        assert all(torch.equal(a, b) for a, b in zip(first, saved))
        assert all(torch.allclose(a, b, atol=3e-3, rtol=3e-3) for a, b in zip(second, expected))
        graph_id = id(model._inference_graph.graph)
        model.forward_both(features[:3])
        assert id(model._inference_graph.graph) == graph_id


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_compile_and_graph_pure_stage_parity_and_budget_fallback():
    def compute(x, y):
        return x.sin() + y.square(), x * y
    x, y = torch.rand(64, 4, device='cuda'), torch.rand(64, 4, device='cuda')
    compiled = CompiledStage(compute, enabled=True)
    graph = SingleCudaGraph(compiled, enabled=True, capacity=64, max_private_mb=64)
    with torch.no_grad():
        expected = compute(x, y)
        actual = graph(x, y)
        assert not compiled.failed
        assert graph.graph is not None, graph.disabled_reason
        assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(actual, expected))
        limited = SingleCudaGraph(compute, enabled=True, capacity=64, max_private_mb=.001)
        assert torch.equal(limited(x, y)[0], expected[0])
        assert limited.graph is None
        assert limited.disabled_reason == 'capture memory budget'
