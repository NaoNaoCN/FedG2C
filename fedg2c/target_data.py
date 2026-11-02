# ===== FedG2C ADD BEGIN C: explicit labels/splits and lazy target views =====
"""Target manifests with explicit split identities and bounded lazy loading."""
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import json
import torch

from .context import GraphView
from .edge_views import NeighborhoodEdgeSampler
from .graph_index import GraphNeighborhoodIndex
from .tensor_shards import TensorShardCache, graph_view, record_at


@dataclass
class TargetSample:
    view: GraphView
    label: torch.Tensor
    root: int | None
    sample_id: str


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
    def __init__(self, refs, task, config, cache=None, node_indices=None):
        self.refs, self.task, self.config = list(refs), task, config
        self.cache = cache or TensorShardCache()
        self.node_indices = node_indices if node_indices is not None else {}

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
            cropped, local_root = view, None
        else:
            if key not in self.node_indices:
                self.node_indices[key] = GraphNeighborhoodIndex(view)
            graph_index = self.node_indices[key]
            if root < 0 or root >= view.num_nodes:
                raise ValueError('Invalid node root')
            nodes = NeighborhoodEdgeSampler.neighborhood(graph_index.neighbors, root,
                                                          self.config.node_hops, self.config.max_nodes)
            cropped, local_root = graph_index.induce(nodes), int((nodes == root).nonzero().item())
        gid = str(record.get('graph_id', path))
        original = None if root is None else int(view.original_node_id[root])
        return TargetSample(cropped, self.label(index), local_root, str((gid, original)))


def load_target_manifest(path, config, include_test=True):
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
    cache, node_indices, splits, seen = TensorShardCache(), {}, {}, set()
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
        splits[split] = TargetSamples(refs, task, config, cache, node_indices)
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
                         samples.cache, samples.node_indices), ids
# ===== FedG2C ADD END C =====
