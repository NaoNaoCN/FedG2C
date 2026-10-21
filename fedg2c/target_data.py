# ===== FedG2C ADD BEGIN C：显式标签划分与节点局部读出 =====
from dataclasses import dataclass
from pathlib import Path
import json
import torch
from .context import GraphView, induced_view
from .edge_views import NeighborhoodEdgeSampler


@dataclass
class TargetSample:
    view: GraphView
    label: torch.Tensor
    root: int | None
    sample_id: str


def load_target_manifest(path, config):
    """Tensor-only files, graph IDs and disjoint node IDs; never infer a split."""
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get('schema_version') != 1 or not manifest.get('protocol_id'):
        raise ValueError('Require target schema_version=1 and protocol_id')
    task = manifest['task']
    if task['level'] not in ('node', 'graph') or task['objective'] not in ('multiclass', 'binary', 'multilabel'):
        raise ValueError('Unsupported target task')
    outputs = task['num_outputs']
    if type(outputs) is not int or outputs < 1 or (task['objective']=='multiclass' and outputs<2) or (task['objective']=='binary' and outputs!=1):
        raise ValueError('Invalid task output dimension')
    visibility = manifest.get('visibility')
    if visibility not in ('transductive', 'split_graphs') or (task['level']=='graph' and visibility!='split_graphs'):
        raise ValueError('Explicit visibility: transductive node graph or split_graphs required')
    cache, seen_samples, graph_files, graph_splits = {}, set(), {}, {}
    splits = {}
    for split in ('train', 'valid', 'test'):
        samples = []
        for entry in manifest['splits'][split]:
            relative = Path(entry['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Tensor file paths must be portable and inside the manifest folder')
            source = (path.parent / relative).resolve()
            if source not in cache:
                record = torch.load(source, map_location='cpu', weights_only=True)
                x, edges = record['x'], record['edge_index']
                if x.ndim!=2 or len(x)<1 or not torch.isfinite(x).all():
                    raise ValueError('Require finite nonempty [N,F] features')
                if edges.dtype!=torch.long or edges.ndim!=2 or edges.shape[0]!=2 or (edges.numel() and (edges.min()<0 or edges.max()>=len(x))):
                    raise ValueError('Invalid edge_index')
                edge = record.get('edge_attr')
                if edge is not None and (edge.ndim!=2 or edge.shape[0]!=edges.shape[1] or not torch.isfinite(edge).all()):
                    raise ValueError('Invalid edge attributes')
                ids = record.get('original_node_id', torch.arange(len(x)))
                if ids.dtype!=torch.long or ids.shape!=(len(x),) or len(ids.unique())!=len(x):
                    raise ValueError('Invalid original node IDs')
                view = GraphView(x, edges, edge, ids)
                neighbors = [set() for _ in range(len(x))]
                for u,v in edges.t().tolist():
                    neighbors[u].add(v); neighbors[v].add(u)
                cache[source] = record, view, neighbors
            record, view, neighbors = cache[source]
            gid = str(record['graph_id'])
            if gid in graph_files and graph_files[gid]!=source:
                raise ValueError('A graph ID must identify exactly one tensor file')
            graph_files[gid] = source
            if visibility=='split_graphs' and gid in graph_splits and graph_splits[gid]!=split:
                raise ValueError('A graph cannot cross split_graphs partitions')
            graph_splits[gid] = split
            y = record['y']
            roots = [None] if task['level']=='graph' else entry['indices']
            if task['level']=='node' and (y.ndim<1 or len(y)!=view.num_nodes):
                raise ValueError('Node labels must have N rows')
            for root in roots:
                if root is not None and (type(root) is not int or not 0<=root<view.num_nodes):
                    raise ValueError('Invalid root index')
                key = (gid, None if root is None else int(view.original_node_id[root]))
                if key in seen_samples:
                    raise ValueError('Duplicate sample or train/valid/test leakage')
                seen_samples.add(key)
                label = (y if root is None else y[root]).reshape(-1)
                if task['objective']=='multiclass':
                    if label.numel()!=1 or not torch.isfinite(label).all() or label.item()!=int(label.item()) or not 0<=label.item()<outputs:
                        raise ValueError('Each multiclass sample requires one class index')
                    label = label.long()
                else:
                    if label.numel()!=outputs or torch.isinf(label).any() or not ((label==0)|(label==1)|torch.isnan(label)).all():
                        raise ValueError('Binary/multilabel labels must be 0,1 or NaN (missing)')
                    label = label.float()
                if root is None:
                    if view.num_nodes>config.max_nodes:
                        raise ValueError('Graph exceeds max_nodes; do not silently crop graph labels')
                    cropped, local_root = view, None
                else:
                    nodes = NeighborhoodEdgeSampler.neighborhood(neighbors, root, config.node_hops, config.max_nodes)
                    cropped = induced_view(view, nodes)
                    local_root = int((nodes==root).nonzero().item())
                samples.append(TargetSample(cropped, label, local_root, str(key)))
        if not samples or not any(torch.isfinite(s.label.float()).any() for s in samples):
            raise ValueError(f'{split} has no observed labels')
        splits[split] = samples
    return manifest, splits
# ===== FedG2C ADD END C =====
