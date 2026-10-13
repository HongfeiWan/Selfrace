"""Short, deterministic CUDA checks; no training, checkpoint writes or logging."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'simulator'), str(ROOT / 'training'), str(ROOT / 'utils')]
from collision import CollisionChecker
from network import create_network


def namespace(value):
    return SimpleNamespace(**{k: namespace(v) for k, v in value.items()}) if isinstance(value, dict) else value


def measure(function, repeats):
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall = time.perf_counter()
    begin.record()
    for _ in range(repeats):
        function()
    end.record()
    torch.cuda.synchronize()
    return dict(cuda_ms=begin.elapsed_time(end) / repeats,
                wall_ms=(time.perf_counter() - wall) * 1000 / repeats,
                allocated_peak_mb=torch.cuda.max_memory_allocated() / 1024**2,
                reserved_peak_mb=torch.cuda.max_memory_reserved() / 1024**2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worlds', type=int, default=128)
    parser.add_argument('--agents', type=int, default=150)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--network', action='store_true', help='Also benchmark a full-size independent actor/critic')
    args = parser.parse_args()
    if min(args.worlds, args.agents, args.repeats) < 1 or not torch.cuda.is_available():
        parser.error('positive sizes and a CUDA device are required')
    torch.manual_seed(87)
    config = yaml.safe_load((ROOT / 'configs/default_config.yaml').read_text(encoding='utf-8'))
    checker = CollisionChecker(config, SimpleNamespace(device='cuda', cell_size=5.))
    reference_config = copy.deepcopy(config)
    reference_config['simulator']['collision_dynamic_cells'] = False
    reference = CollisionChecker(reference_config, SimpleNamespace(device='cuda', cell_size=5.))
    s0 = torch.zeros(args.worlds, args.agents, 7, device='cuda')
    s0[..., :2] = torch.rand_like(s0[..., :2]) * 300
    s0[..., 2] = torch.rand_like(s0[..., 2]) * 6.28
    s0[..., 4], s0[..., 5] = 4.5, 2.
    s0[..., 6] = (torch.rand_like(s0[..., 6]) > .4).float()
    s1 = s0.clone()
    s1[..., :2] += torch.randn_like(s0[..., :2])
    with torch.no_grad():
        assert torch.equal(checker.check(s0, s1), reference.check(s0, s1))
        report = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, worlds=args.worlds,
                      agents=args.agents, repeats=args.repeats,
                      collision_reference=measure(lambda: reference.check(s0, s1), args.repeats),
                      collision_dynamic_cells=measure(lambda: checker.check(s0, s1), args.repeats),
                      dynamic_cells_active=not checker._cell_failed,
                      reference_compiled=reference._compiled_stream_chunk is not None and not reference._stream_compile_failed)
    if args.network:
        config['training']['network']['cuda_graph']['enabled'] = False
        model = create_network(namespace(config)).cuda().eval()
        schema = model.policy_feature_encoder.feature_schema
        features = torch.randn(2048, 1, schema.total_input_dim, device='cuda').clamp(-1, 1)
        for group in schema.set_groups:
            elements = features[..., schema.flat_slice(group.name)].view(2048, 1, group.element_count, group.element_dim)
            valid = torch.rand(2048, 1, group.element_count, device='cuda') > .4
            elements.masked_fill_(~valid[..., None], -2.)
            if group.active_channel is not None:
                elements[..., group.resolved_active_channel] = valid
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            expected = tuple(v.clone() for v in model.forward_both(features))
            report['network_compiled'] = measure(lambda: model.forward_both(features), args.repeats)
            model._inference_graph.enabled = True
            actual = model.forward_both(features)
            assert all(torch.allclose(a, b, atol=.003, rtol=.003) for a, b in zip(actual, expected))
            report['network_graph'] = measure(lambda: model.forward_both(features), args.repeats)
            report['graph_captured'] = model._inference_graph.graph is not None
            report['graph_disabled_reason'] = model._inference_graph.disabled_reason
            report['graph_reserved_mb'] = model._inference_graph.private_reserved_bytes / 1024**2
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
