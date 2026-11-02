"""Bounded, isolated scalar/physical-batch benchmark; writes no training outputs.

Supply a prepared source manifest to measure the actual source client. Without
it, use synthetic graphs to check execution locally; this is not a Cora result.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch

from fedg2c.context import GraphView
from fedg2c.data import read_manifest
from fedg2c.edge_client import EdgePredictionClient
from fedg2c.edge_config import EdgePredictionConfig
from fedg2c.gefra_loo_matrix import target_config
from fedg2c.model import SharedBackbone
from fedg2c.tensor_shards import IndexedSourceGraphs


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=REPO/'config/fedg2c_gefra_loo_v1_cora_cuda.json')
    parser.add_argument('--target', default='cora')
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--client', default='pubmed')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--physical-batch-size', type=int, default=32)
    parser.add_argument('--views', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args(argv)
    if min(args.views, args.repeats, args.threads, args.physical_batch_size) < 1:
        parser.error('Use positive counts')
    raw = json.loads(args.config.read_text(encoding='utf-8'))
    cfg = target_config(raw, args.target)[0] if 'matrix' in raw else raw
    config = EdgePredictionConfig(**cfg['stage1']['module_a'])
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda') or (device.type == 'cuda' and not torch.cuda.is_available()):
        parser.error('Requested CPU/CUDA device is unavailable')
    torch.set_num_threads(args.threads)
    seed = cfg['run']['seed']
    torch.manual_seed(seed)
    if args.source_manifest:
        manifest = read_manifest(args.source_manifest)
        records = [r for r in manifest['clients'] if r['id'] == args.client]
        if len(records) != 1:
            parser.error('Choose a client in the source manifest')
        record = records[0]
        graphs = IndexedSourceGraphs(record['graphs'], record.get('graph_counts'))
        schema, client_id = record['feature_schema'], record['id']
        sampling = cfg['stage1']['training'].get('sampling_unit', 'graph')
        note = 'real permitted source views; bounded timing run, not formal training'
    else:
        gen = torch.Generator().manual_seed(seed)
        schema, client_id, sampling, graphs = {'input_dim':64}, 'synthetic', 'graph', []
        for i in range(args.views):
            n = min(config.max_nodes, 8 + i % 25)
            u = torch.arange(n)
            edges = torch.stack((u, (u + 1) % n))
            graphs.append(GraphView(torch.randn(n, 64, generator=gen),
                                    torch.cat((edges, edges.flip(0)), dim=1), None, u))
        note = 'synthetic execution benchmark; not a Cora dataset/result'
    initial = EdgePredictionClient(client_id, config, SharedBackbone(config), schema, device)
    factory = lambda: iter(graphs)
    # Each measured run starts from identical weights/RNG and makes exactly one
    # optimizer/EMA update after args.views views. Formal config is untouched.
    timings, reports = {}, {}
    for physical in dict.fromkeys((1, args.physical_batch_size)):
        warmup = deepcopy(initial)
        warmup.pretrain(factory, seed=seed, epochs=args.views, max_steps=1,
                        batch_size=min(8, args.views), physical_batch_size=physical,
                        sampling_unit=sampling)
        synchronize(device)
        del warmup
        seconds, peaks = [], []
        for _ in range(args.repeats):
            client = deepcopy(initial)
            synchronize(device)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            result = client.pretrain(factory, seed=seed, epochs=args.views, max_steps=1,
                                     batch_size=args.views, physical_batch_size=physical,
                                     sampling_unit=sampling)
            synchronize(device)
            elapsed = time.perf_counter() - started
            if result['total_views'] != args.views or len(result['steps']) != 1:
                raise RuntimeError('Source pass ended before the requested benchmark budget')
            seconds.append(elapsed)
            if device.type == 'cuda':
                peaks.append(torch.cuda.max_memory_allocated(device)/1024**2)
            reports[physical] = result['steps'][-1]['loss']
            del client
        timings[physical] = {'seconds':seconds, 'median_seconds':statistics.median(seconds),
                             'views_per_second':args.views/statistics.median(seconds),
                             'peak_cuda_allocated_mib':max(peaks) if peaks else None}
    summary = {'device':str(device), 'torch':str(torch.__version__), 'note':note,
               'client':client_id, 'views':args.views, 'threads':args.threads,
               'timings':timings, 'speedup':timings[1]['median_seconds'] /
               timings[args.physical_batch_size]['median_seconds'],
               'loss_absolute_difference':abs(reports[1]-reports[args.physical_batch_size]),
               'precision':'float32; no AMP; no change to formal config or checkpoints'}
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
