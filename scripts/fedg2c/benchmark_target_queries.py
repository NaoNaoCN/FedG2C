"""Compare full/root target queries on identical synthetic inputs, without training.

Includes expert forward and backward to trainable input/adaptation modules.
This is a microbenchmark, not a measured dataset/pipeline completion time.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch

from fedg2c.batching import GraphBatch
from fedg2c.context import GraphView, structural_relation_rows
from fedg2c.edge_config import EdgePredictionConfig
from fedg2c.model import SharedBackbone
from fedg2c.target_model import TargetConfig, TargetModel


def full_queries(model, batch, roots):
    """Original node-target path, retained only as a benchmark reference."""
    x = model.node_adapter(batch.x)
    h = model.shared.encode_experts(x, batch.edge_index)
    if model.prompt is not None:
        h = h + model.prompt
    z = model.shared.compose_batch(h, batch)
    if model.adapter is not None:
        z = model.adapter(z)
    return model.head(z[batch.offsets + roots])


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--nodes', type=int, default=128)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--experts', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--adaptation', choices=['none','lora','adapter','prompt'], default='lora')
    args = parser.parse_args(argv)
    if min(args.batch_size, args.nodes, args.hidden_dim, args.experts, args.repeats) < 1:
        parser.error('Use positive counts')
    device = torch.device(args.device)
    if device.type not in ('cpu','cuda') or (device.type == 'cuda' and not torch.cuda.is_available()):
        parser.error('Requested CPU/CUDA device is unavailable')
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    config = EdgePredictionConfig(edge_mask_rate=.2, feature_mask_rate=.2,
                                  hidden_dim=args.hidden_dim, num_experts=args.experts,
                                  num_heads=4, predictor_dim=args.hidden_dim, predictor_heads=4)
    target = TargetConfig(adaptation=args.adaptation)
    model = TargetModel(SharedBackbone(config), config, {'input_dim':64},
                        {'level':'node','objective':'multiclass','num_outputs':7}, target).to(device).train()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'lora_B' in name or name.startswith('adapter.up') or name == 'prompt':
                parameter.normal_(0, .02)
    views, descriptors = [], []
    roots = torch.tensor([i % args.nodes for i in range(args.batch_size)], device=device)
    for i in range(args.batch_size):
        u = torch.arange(args.nodes)
        edges = torch.stack((u, (u+1) % args.nodes))
        edges = torch.cat((edges, edges.flip(0)), dim=1)
        views.append(GraphView(torch.randn(args.nodes, 64), edges, None, u))
        descriptors.append(structural_relation_rows(edges, args.nodes, torch.tensor([i % args.nodes]),
                                                    dtype=torch.bool)[0])
    # Both references receive the same packed graph, precision and batch size.
    # Rebuild GraphBatch per call just as predict() does; descriptors are cached.
    relations = torch.stack(descriptors).to(device)
    probe = torch.randn(args.batch_size, 7, device=device)

    reference, optimized = deepcopy(model), deepcopy(model)
    left = full_queries(reference, GraphBatch.collate(views).to(device), roots)
    right = optimized.forward_batch(GraphBatch.collate(views).to(device), roots, relations)
    (left * probe).sum().backward()
    (right * probe).sum().backward()
    logit_error = float((left-right).abs().max().detach().cpu())
    gradient_error = 0.
    for (name, p), (_, q) in zip(reference.named_parameters(), optimized.named_parameters()):
        if (p.grad is None) != (q.grad is None):
            raise AssertionError('Gradient reachability differs: '+name)
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=3e-5, rtol=3e-4, msg=name)
            gradient_error = max(gradient_error, float((p.grad-q.grad).abs().max().cpu()))
    torch.testing.assert_close(left, right, atol=5e-6, rtol=3e-5)
    del reference, optimized, left, right

    measurements = {}
    for label in ('full_queries', 'root_queries'):
        times, peaks = [], []
        for repetition in range(args.repeats + 1):
            model.zero_grad(set_to_none=True)
            synchronize(device)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            batch = GraphBatch.collate(views).to(device)
            result = (full_queries(model, batch, roots) if label == 'full_queries' else
                      model.forward_batch(batch, roots, relations))
            (result * probe).sum().backward()
            synchronize(device)
            elapsed = time.perf_counter()-started
            if repetition:  # First call warms this path, including kernels.
                times.append(elapsed)
                if device.type == 'cuda':
                    peaks.append(torch.cuda.max_memory_allocated(device)/1024**2)
            del batch, result
        measurements[label] = {'seconds':times, 'median_seconds':statistics.median(times),
                               'peak_cuda_allocated_mib':max(peaks) if peaks else None}
    print(json.dumps({'note':'synthetic target forward/backward, not a dataset/pipeline runtime',
                      'device':str(device), 'torch':str(torch.__version__),
                      'batch_size':args.batch_size, 'nodes':args.nodes,
                      'hidden_dim':args.hidden_dim, 'experts':args.experts,
                      'target':asdict(target), 'precision':'float32; no AMP',
                      'logit_max_abs_error':logit_error, 'gradient_max_abs_error':gradient_error,
                      'measurements':measurements,
                      'speedup':measurements['full_queries']['median_seconds'] /
                                measurements['root_queries']['median_seconds']}, indent=2))


if __name__ == '__main__':
    main()
