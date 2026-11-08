# ===== FedG2C ADD BEGIN C: explicit labels/splits and lazy target views =====
"""Target manifests with explicit split identities and bounded lazy loading."""
from collections.abc import Sequence
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import json
import torch

from .context import GraphView, structural_relation_rows
from .edge_views import NeighborhoodEdgeSampler
from .graph_index import GraphNeighborhoodIndex
from .tensor_shards import TensorShardCache, graph_view, record_at


@dataclass
class TargetSample:
    view: GraphView
    label: torch.Tensor
    root: int | None
    sample_id: str
    root_relations: torch.Tensor | None = None


class TargetDataCache:
    """Shared across seeds/adaptations; cache topology only, never embeddings.

    Crops keep node IDs, remapped edges, original arc rows, the local root and
    its three binary structural-relation channels. No feature/embedding cache.
    Their LRU byte budget excludes the existing shard and adjacency-index cache.
    Labels and support choices are always read through the requested split.
    """
    def __init__(self, crop_cache_mb=128):
        self.shards, self.node_indices = TensorShardCache(), {}
        self.max_bytes = crop_cache_mb * 1024 * 1024
        self.crops, self.crop_bytes = OrderedDict(), 0
        self.hits, self.misses = 0, 0
        self.evictions = 0

    @staticmethod
    def crop_size(crop):
        return sum(t.numel() * t.element_size() for t in crop if isinstance(t, torch.Tensor))

    def stats(self):
        return {'entries':len(self.crops), 'tensor_bytes':self.crop_bytes,
                'budget_bytes':self.max_bytes, 'hits':self.hits, 'misses':self.misses,
                'evictions':self.evictions}

    def get_crop(self, key, index, root, hops, cap):
        if key in self.crops:
            self.hits += 1
            self.crops.move_to_end(key)
            return self.crops[key][0]
        self.misses += 1
        nodes = NeighborhoodEdgeSampler.neighborhood(index.neighbors, root, hops, cap)
        edges, arcs = index.induced_edges(nodes)
        local_root = int((nodes == root).nonzero().item())
        relations = structural_relation_rows(edges, len(nodes), torch.tensor([local_root]),
                                             dtype=torch.bool)[0]
        crop = (nodes, edges, arcs, local_root, relations)
        size = self.crop_size(crop)
        if size <= self.max_bytes:
            while self.crop_bytes + size > self.max_bytes:
                _, (_, old_size) = self.crops.popitem(last=False)
                self.crop_bytes -= old_size
                self.evictions += 1
            self.crops[key] = (crop, size)
            self.crop_bytes += size
        return crop


def checked_label(label, task):
    label = label.reshape(-1)
    if task['objective'] == 'multiclass':
        if (label.numel() != 1 or not torch.isfinite(label).all()
                or label.item() != int(label.item())
                or not 0 <= label.item() < task['num_outputs']):
            raise ValueError('Each multiclass sample requires one class index')
        return label.long()
    if (label.numel() != task['num_outputs'] or torch.isinf(label).any()
            or not ((label == 0) | (label == 1) | torch.isnan(label)).all()):
        raise ValueError('Binary/multilabel labels must be 0, 1 or NaN')
    return label.float()


class TargetSamples(Sequence):
    """References stay small; graph tensors are opened on sample access."""
    def __init__(self, refs, task, config, cache=None, node_indices=None, data_cache=None):
        self.refs, self.task, self.config = list(refs), task, config
        self.cache = cache or TensorShardCache()
        self.node_indices = node_indices if node_indices is not None else {}
        self.data_cache = data_cache if data_cache is not None else TargetDataCache(config.crop_cache_mb)

    def __len__(self):
        return len(self.refs)

    def _record(self, index):
        path, offset, root = self.refs[index]
        return path, offset, root, record_at(self.cache.get(path), offset)

    def label(self, index):
        _, _, root, record = self._record(index)
        if root is not None and (record['y'].ndim < 1 or len(record['y']) != len(record['x'])):
            raise ValueError('Node labels must match the full graph')
        label = record['y'] if root is None else record['y'][root]
        return checked_label(label, self.task)

    def _node_crop(self, path, offset, root, record):
        key = (path, offset)
        if key not in self.node_indices:
            self.node_indices[key] = GraphNeighborhoodIndex(graph_view(record))
        graph_index = self.node_indices[key]
        if root < 0 or root >= graph_index.graph.num_nodes:
            raise ValueError('Invalid node root')
        crop_key = (path, offset, root, self.config.node_hops, self.config.max_nodes)
        crop = self.data_cache.get_crop(crop_key, graph_index, root,
                                        self.config.node_hops, self.config.max_nodes)
        return graph_index, crop

    def topology(self, index):
        """Prepare fixed topology without materializing features or reading labels."""
        path, offset, root, record = self._record(index)
        if root is None:
            raise ValueError('Only node targets have a rooted topology cache')
        return self._node_crop(path, offset, root, record)[1]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        path, offset, root, record = self._record(index)
        key = (path, offset)
        view = (self.node_indices[key].graph if root is not None and key in self.node_indices
                else graph_view(record))
        if root is None:
            if self.config.max_nodes and view.num_nodes > self.config.max_nodes:
                raise ValueError('Graph exceeds max_nodes; do not silently crop labels')
            cropped, local_root, relations = view, None, None
        else:
            graph_index, (nodes, edges, arcs, local_root, relations) = self._node_crop(
                path, offset, root, record)
            cropped = graph_index.induce(nodes, topology=(edges, arcs))
        gid = str(record.get('graph_id', path))
        original = None if root is None else int(view.original_node_id[root])
        return TargetSample(cropped, self.label(index), local_root, str((gid, original)), relations)


def precompute_topology(splits):
    """Warm only the selected training support and validation roots, never test.

    The configured byte limit remains hard. Report the working set so callers
    can enlarge the budget if LRU eviction would otherwise rebuild crops.
    """
    train = splits['train']
    if (train.task['level'] != 'node' or not train.config.precompute_topology
            or not train.data_cache.max_bytes):
        return {'enabled':False}
    required_bytes, counts = 0, {}
    for name in ('train', 'valid'):
        samples = splits[name]
        counts[name] = len(samples)
        for i in range(len(samples)):
            required_bytes += train.data_cache.crop_size(samples.topology(i))
    return {'enabled':True, 'split_sizes':counts, 'working_set_tensor_bytes':required_bytes,
            'working_set_fits':required_bytes <= train.data_cache.max_bytes,
            **train.data_cache.stats()}


def load_target_manifest(path, config, include_test=True, data_cache=None):
    """Validate split identity and paths without opening test tensors in tuning."""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get('schema_version') != 1 or not manifest.get('protocol_id'):
        raise ValueError('Require target schema_version=1 and protocol_id')
    task = manifest['task']
    if task['level'] not in ('node', 'graph') or task['objective'] not in ('multiclass', 'binary', 'multilabel'):
        raise ValueError('Unsupported target task')
    outputs = task['num_outputs']
    if type(outputs) is not int or outputs < 1 or (task['objective'] == 'multiclass' and outputs < 2) or (task['objective'] == 'binary' and outputs != 1):
        raise ValueError('Invalid task output dimension')
    visibility = manifest.get('visibility')
    if visibility not in ('transductive', 'split_graphs') or (task['level'] == 'graph' and visibility != 'split_graphs'):
        raise ValueError('Explicit visibility required')
    if task['level'] == 'node' and config.max_nodes == 0:
        raise ValueError('Node targets require a positive bounded max_nodes')
    data_cache = data_cache if data_cache is not None else TargetDataCache(config.crop_cache_mb)
    cache, node_indices, splits, seen = data_cache.shards, data_cache.node_indices, {}, set()
    for split in ('train', 'valid', 'test'):
        refs = []
        for entry in manifest['splits'][split]:
            relative = Path(entry['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Tensor paths must stay inside manifest folder')
            source = (path.parent / relative).resolve()
            if not source.is_relative_to(path.parent):
                raise ValueError('Target tensor path leaves manifest folder')
            offset = entry.get('index', 0)
            if type(offset) is not int or offset < 0:
                raise ValueError('Invalid shard index')
            roots = [None] if task['level'] == 'graph' else entry['indices']
            for root in roots:
                if root is not None and (type(root) is not int or root < 0):
                    raise ValueError('Invalid root index')
                key = (source, offset, root)
                if key in seen:
                    raise ValueError('Duplicate sample or train/valid/test leakage')
                seen.add(key)
                refs.append(key)
        if split == 'test' and not include_test:
            continue
        if not refs:
            raise ValueError(f'{split} has no samples')
        for source in {ref[0] for ref in refs}:
            if not source.is_file():
                raise FileNotFoundError(source)
        splits[split] = TargetSamples(refs, task, config, cache, node_indices, data_cache)
    return manifest, splits


def select_support(samples, task, k_shot):
    """Select the first k sorted train IDs per class/task-class, Gefra style."""
    if k_shot == -1:
        return samples, list(range(len(samples)))
    if type(k_shot) is not int or k_shot < 1:
        raise ValueError('k_shot must be -1 or a positive integer')
    selected, counts = set(), {}
    for i in range(len(samples)):
        label = samples.label(i)
        if task['objective'] == 'multiclass':
            keys = [(0, int(label.item()))]
        else:
            keys = [(j, int(value)) for j, value in enumerate(label) if torch.isfinite(value)]
        for key in keys:
            if counts.get(key, 0) < k_shot:
                selected.add(i)
                counts[key] = counts.get(key, 0) + 1
    ids = sorted(selected)
    if not ids:
        raise ValueError('No labelled few-shot support')
    return TargetSamples([samples.refs[i] for i in ids], task, samples.config,
                         samples.cache, samples.node_indices, samples.data_cache), ids
# ===== FedG2C ADD END C =====
