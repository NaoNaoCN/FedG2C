"""Bounded tensor-only shard storage for full graph datasets.

Packed shards store a few contiguous tensors and offsets rather than hundreds
of tensor objects. Random graph access uses mmap and an LRU; it never retains
all PCBA GraphViews in Python memory. Labels are only requested by target code.
"""
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path

import torch

from .context import GraphView, validate_graph


def pack_records(records):
    if not records:
        raise ValueError('Cannot pack an empty shard')
    nodes = [int(r['x'].shape[0]) for r in records]
    edges = [int(r['edge_index'].shape[1]) for r in records]
    result = {'format': 'packed_graphs_v1', 'count': len(records),
              'x': torch.cat([r['x'] for r in records]),
              # Edge indices stay LOCAL to each graph, so no batch relabel is needed.
              'edge_index': torch.cat([r['edge_index'] for r in records], dim=1),
              'node_ptr': torch.tensor([0] + nodes).cumsum(0),
              'edge_ptr': torch.tensor([0] + edges).cumsum(0),
              'graph_ids': [r.get('graph_id', str(i)) for i, r in enumerate(records)]}
    for key in ('edge_attr', 'y'):
        present = [key in r for r in records]
        if any(present) and not all(present):
            raise ValueError(f'Inconsistent {key} in shard')
        if all(present):
            result[key] = (torch.stack([r[key].reshape(-1) for r in records]) if key == 'y'
                           else torch.cat([r[key] for r in records]))
    return result


def record_count(payload):
    if isinstance(payload, list):
        return len(payload)
    return int(payload['count']) if payload.get('format') == 'packed_graphs_v1' else 1


def record_at(payload, index=0):
    count = record_count(payload)
    if type(index) is not int or not 0 <= index < count:
        raise ValueError('Shard record index out of range')
    if isinstance(payload, list):
        return payload[index]
    if payload.get('format') != 'packed_graphs_v1':
        return payload
    ns, ne = (int(v) for v in payload['node_ptr'][index:index+2])
    es, ee = (int(v) for v in payload['edge_ptr'][index:index+2])
    result = {'x': payload['x'][ns:ne], 'edge_index': payload['edge_index'][:, es:ee],
              'graph_id': payload['graph_ids'][index]}
    if 'edge_attr' in payload:
        result['edge_attr'] = payload['edge_attr'][es:ee]
    if 'y' in payload:
        result['y'] = payload['y'][index]
    return result


class TensorShardCache:
    def __init__(self, capacity=64):
        self.capacity, self.cache = capacity, OrderedDict()

    def get(self, path):
        path = str(Path(path))
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        value = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        self.cache[path] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return value


def graph_view(record, validate=True):
    x, edges = record['x'], record['edge_index']
    view = GraphView(x, edges, record.get('edge_attr'),
                     record.get('original_node_id', torch.arange(len(x))))
    if validate:
        validate_graph(view)
    return view


class IndexedSourceGraphs:
    def __init__(self, paths, counts=None):
        self.paths, self.cache = list(paths), TensorShardCache()
        if counts is None:
            counts = [record_count(self.cache.get(p)) for p in self.paths]
        if len(counts) != len(self.paths) or any(type(n) is not int or n < 1 for n in counts):
            raise ValueError('One positive graph count per shard required')
        self.offsets = [0]
        for count in counts:
            self.offsets.append(self.offsets[-1] + count)

    def __len__(self):
        return self.offsets[-1]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard = bisect_right(self.offsets, index) - 1
        payload = self.cache.get(self.paths[shard])
        if record_count(payload) != self.offsets[shard+1] - self.offsets[shard]:
            raise ValueError('Source shard count differs from manifest')
        # Do not expose y, masks, or graph IDs to a source training client.
        return graph_view(record_at(payload, index-self.offsets[shard]))
