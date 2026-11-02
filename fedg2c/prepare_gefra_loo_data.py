"""Prepare the Gefra LODO data roles for FedG2C.

This exporter copies Gefra's dataset membership and split rules, then emits
portable tensor manifests. It does not export labels to source clients and it
does not import Gefra's geometric model or aggregation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, Planetoid, Reddit, TUDataset, WikiCS
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import from_scipy_sparse_matrix

from .context import GraphView, validate_graph
from .prepare_molecular_data import normalized_degree, molecular_schema, add_ogb_safe_globals
from .gefra_protocol import PROTOCOL_VERSION, sources_for
from .tensor_shards import pack_records
from utils.split_protocol import dataset_seed, make_paper_node_split

GRAPH_DATASETS = ('chemhiv', 'chempcba', 'proteins', 'imdb-binary')
NODE_DATASETS = ('cora', 'citeseer', 'pubmed', 'photo', 'computers', 'wikics',
                 'arxiv', 'reddit', 'airport', 'disease')
GRAPH_SHARD_SIZE = 8192


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def node_record(data, name, labels=None):
    # Keep the actual orientation (including Arxiv). Region selection alone
    # uses an undirected adjacency in FedG2C, as in its original sampler.
    edge = data.edge_index.detach().cpu().long()
    x = data.x.detach().cpu().float().contiguous()
    original = torch.arange(x.shape[0], dtype=torch.long)
    result = {'x': x, 'edge_index': edge.contiguous(),
              'original_node_id': original, 'graph_id': str(name)}
    if labels is not None:
        result['y'] = labels.detach().cpu().reshape(-1).long()
    validate_graph(GraphView(x, edge, None, original))
    return result


def load_domain_node(cache, name):
    import scipy.sparse as sp
    if name == 'airport':
        import networkx as nx
        with (cache / 'airport/airport.p').open('rb') as stream:
            graph = pickle.load(stream)
        adjacency = nx.adjacency_matrix(graph)
        features = np.asarray([graph.nodes[u]['feat'] for u in graph.nodes()])
        raw_labels = features[:, 4]
        labels = np.digitize(raw_labels, [1.0, 8.0 / 7, 9.0 / 7])
        labels = labels - labels.min()
        features = features[:, :4]
    else:
        root = cache / 'disease_nc'
        objects, edges = {}, []
        with (root / 'disease_nc.edges.csv').open(encoding='utf-8') as stream:
            for line in stream:
                left, right = line.rstrip().split(',')
                for node in (left, right):
                    if node not in objects:
                        objects[node] = len(objects)
                edges.append((objects[left], objects[right]))
        rows = [u for u, _ in edges] + [v for _, v in edges]
        cols = [v for _, v in edges] + [u for u, _ in edges]
        adjacency = sp.csr_matrix((np.ones(len(rows)), (rows, cols)),
                                  shape=(len(objects), len(objects)))
        features = sp.load_npz(root / 'disease_nc.feats.npz')
        labels = np.load(root / 'disease_nc.labels.npy')
    edge_index, _ = from_scipy_sparse_matrix(adjacency)
    if hasattr(features, 'todense'):
        features = np.asarray(features.todense())
    labels = np.asarray(labels)
    if labels.ndim > 1 and labels.shape[1] > 1:
        labels = labels.argmax(axis=1)
    return Data(x=torch.tensor(features, dtype=torch.float32), edge_index=edge_index,
                y=torch.tensor(labels, dtype=torch.long))


def load_node(cache, name):
    if name in ('cora', 'citeseer', 'pubmed'):
        data = Planetoid(str(cache), name.capitalize(), transform=NormalizeFeatures())[0]
        policy = 'NormalizeFeatures'
    elif name in ('photo', 'computers'):
        data = Amazon(str(cache), name.capitalize(), transform=NormalizeFeatures())[0]
        policy = 'NormalizeFeatures'
    elif name == 'wikics':
        data = WikiCS(root=str(cache / 'wikics'), is_undirected=True)[0]
        policy = 'as_provided'
    elif name == 'reddit':
        data = Reddit(root=str(cache / 'reddit'))[0]
        policy = 'as_provided'
    elif name == 'arxiv':
        from ogb.nodeproppred import PygNodePropPredDataset
        from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
        from torch_geometric.data.storage import EdgeStorage, GlobalStorage, NodeStorage
        torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr,
                                               GlobalStorage, NodeStorage, EdgeStorage])
        data = PygNodePropPredDataset(name='ogbn-arxiv', root=str(cache / 'arxiv'))[0]
        policy = 'as_provided'
    elif name in ('airport', 'disease'):
        data = load_domain_node(cache, name)
        policy = 'Gefra_domain_preprocessing'
    else:
        raise ValueError(f'Unsupported node dataset: {name}')
    labels = data.y.detach().cpu().reshape(-1).long()
    record = node_record(data, name, labels)
    audit = {'dataset': name, 'nodes': int(record['x'].shape[0]),
             'edge_arcs': int(record['edge_index'].shape[1]),
             'input_dim': int(record['x'].shape[1]),
             'num_classes': int(labels.max()) + 1,
             'feature_preprocessing': policy}
    return record, labels, audit


def load_graph(cache, name, feature_policy='native'):
    if name in ('chemhiv', 'chempcba'):
        add_ogb_safe_globals()
        from ogb.graphproppred import PygGraphPropPredDataset
        dataset = PygGraphPropPredDataset(
            name='ogbg-molhiv' if name == 'chemhiv' else 'ogbg-molpcba', root=str(cache))
        outputs = int(dataset.num_tasks)
        objective = 'multilabel' if outputs > 1 else 'binary'
        schema = molecular_schema() if feature_policy == 'native' else {'input_dim': 9, 'edge_dim': 0}
    else:
        import torch_geometric.transforms as transforms
        transform = (transforms.OneHotDegree(max_degree=135)
                     if name == 'imdb-binary' and feature_policy == 'gefra_compat' else None)
        dataset = TUDataset(root=str(cache), name='PROTEINS' if name == 'proteins' else 'IMDB-BINARY',
                            transform=transform)
        if name == 'imdb-binary' and feature_policy == 'native':
            input_dim = 1
        elif name == 'imdb-binary' and feature_policy == 'gefra_compat':
            input_dim = 136
        else:
            input_dim = int(dataset.num_node_features)
        # All four Gefra graph tasks use sigmoid/BCE and ROC-AUC, including TU.
        schema, objective, outputs = {'input_dim': input_dim, 'edge_dim': 0}, 'binary', 1
    return dataset, schema, {'objective': objective, 'num_outputs': outputs}


def export_graph(graph, name, index, feature_policy, label=False):
    # Target singleton/edgeless graphs remain in the target dataset. Source
    # eligibility is audited separately, since A requires a non-self edge.
    x = graph.x
    if x is None and name == 'imdb-binary':
        x = normalized_degree(graph.edge_index, int(graph.num_nodes))
    if x is None or x.ndim != 2 or len(x) < 1 or not torch.isfinite(x).all():
        raise ValueError(f'Invalid graph features: {name}:{index}')
    result = {'x': x.detach().cpu().contiguous().clone(),
              'edge_index': graph.edge_index.detach().cpu().long().contiguous().clone(),
              'graph_id': f'{name}:{index}'}
    if feature_policy == 'gefra_compat':
        result['x'] = result['x'].float()
    elif graph.edge_attr is not None:
        result['edge_attr'] = graph.edge_attr.detach().cpu().contiguous().clone()
    if label:
        y = graph.y.detach().cpu().reshape(-1).float().clone()
        y[y == -1] = float('nan')
        if not ((y == 0) | (y == 1) | torch.isnan(y)).all():
            raise ValueError(f'Expected 0/1/NaN graph labels: {name}:{index}')
        result['y'] = y
    return result


def save_shard(output, prefix, records, number):
    path = output / f'{prefix}_{number:05d}.pt'
    torch.save(pack_records(records), path)
    return {'path': path.name, 'count': len(records), 'sha256': file_digest(path)}


def graph_split_indices(count, target, seed):
    # Gefra's graph_set_partition replaces the masks initially assigned by its
    # raw loader, including OGB's official scaffold masks. Match the FINAL masks
    # of client 0 in its one-client-per-dataset LODO path.
    order = np.arange(count)
    np.random.default_rng(dataset_seed(seed, target, 0)).shuffle(order)
    boundaries = (int(.8 * count), int(.9 * count))
    # Masks in Gefra are converted to sorted indices before few-shot selection.
    return {'train': sorted(order[:boundaries[0]].tolist()),
            'valid': sorted(order[boundaries[0]:boundaries[1]].tolist()),
            'test': sorted(order[boundaries[1]:].tolist())}


def prepare_gefra_loo_data(cache_root, output, seed=2026, target='chemhiv',
                          feature_policy='native', source_ineligible_policy='exclude'):
    cache, output = Path(cache_root).resolve(), Path(output).resolve()
    if not cache.is_dir():
        raise FileNotFoundError(cache)
    if output.exists():
        raise FileExistsError('Use a new Gefra LOO output directory')
    if output.is_relative_to(cache):
        raise ValueError('Prepared output must be outside the shared cache')
    if feature_policy not in ('native', 'gefra_compat'):
        raise ValueError('Unknown feature_policy')
    if source_ineligible_policy not in ('error', 'exclude'):
        raise ValueError('Unknown source_ineligible_policy')
    level, sources = sources_for(target)
    output.mkdir(parents=True)
    identity = f'{PROTOCOL_VERSION}-{target}-{feature_policy}-{source_ineligible_policy}-seed{seed}'
    clients, audits, shards = [], [], []
    for name in sources:
        if level == 'node':
            record, _, audit = load_node(cache, name)
            record.pop('y', None)
            eligible = torch.unique(record['edge_index'][:, record['edge_index'][0] != record['edge_index'][1]])
            eligible_mask = torch.zeros(len(record['x']), dtype=torch.bool)
            eligible_mask[eligible] = True
            excluded_roots = (~eligible_mask).nonzero().flatten().tolist()
            audit['eligible_roots'] = len(eligible)
            audit['isolated_roots_excluded'] = len(excluded_roots)
            audit['excluded_no_nonself_edge_root_ids'] = excluded_roots
            if excluded_roots and source_ineligible_policy == 'error':
                write_json(output / 'eligibility_report.json', audit)
                raise ValueError(f'{name}: {len(excluded_roots)} source roots have no non-self edge; '
                                 'review eligibility_report.json and choose an exclusion policy')
            if len(eligible) == 0:
                raise ValueError(f'{name}: no eligible source roots')
            path = output / f'source_{name}.pt'
            torch.save(record, path)
            shard = {'path': path.name, 'count': 1, 'sha256': file_digest(path)}
            schema = {'input_dim': int(record['x'].shape[1]), 'edge_dim': 0}
            current = [shard]
        else:
            dataset, schema, _ = load_graph(cache, name, feature_policy)
            current, buffer, excluded = [], [], []
            for index in range(len(dataset)):
                graph = dataset[index]
                eligible = int(graph.num_nodes) >= 2 and bool((graph.edge_index[0] != graph.edge_index[1]).any())
                if not eligible:
                    excluded.append(index)
                    continue
                buffer.append(export_graph(graph, name, index, feature_policy))
                if len(buffer) == GRAPH_SHARD_SIZE:
                    current.append(save_shard(output, f'source_{name}', buffer, len(current)))
                    buffer = []
            if buffer:
                current.append(save_shard(output, f'source_{name}', buffer, len(current)))
            audit = {'dataset': name, 'raw_graphs': len(dataset),
                     'eligible_graphs': sum(s['count'] for s in current),
                     'excluded_no_nonself_edge_ids': excluded}
            if excluded and source_ineligible_policy == 'error':
                write_json(output / 'eligibility_report.json', audit)
                raise ValueError(f'{name}: {len(excluded)} source graphs cannot form an edge view; '
                                 'review eligibility_report.json and explicitly choose an exclusion policy')
            if not current:
                raise ValueError(f'{name}: no eligible source graphs')
            del dataset
        clients.append({'id': name, 'feature_schema': schema, 'graphs': [s['path'] for s in current],
                        'graph_counts': [s['count'] for s in current]})
        audits.append(audit)
        shards.extend(current)
        print(json.dumps({'prepared_source': name, **{key: value for key, value in audit.items()
                          if not key.endswith('_ids')}}), flush=True)
    source = {'schema_version': 1, 'protocol_id': identity, 'target_dataset': target,
              'level': level, 'clients': clients}
    if level == 'node':
        record, labels, audit = load_node(cache, target)
        path = output / f'target_{target}.pt'
        torch.save(record, path)
        train, valid, test = make_paper_node_split(labels, target, seed)
        splits = {split: [{'path': path.name, 'indices': mask.nonzero().flatten().tolist()}]
                  for split, mask in (('train', train), ('valid', valid), ('test', test))}
        target_manifest = {'task': {'level': 'node', 'objective': 'multiclass',
                                    'num_outputs': int(labels.max()) + 1},
                           'visibility': 'transductive',
                           'feature_schema': {'input_dim': int(record['x'].shape[1]), 'edge_dim': 0},
                           'splits': splits}
        target_audit = dict(audit, split_sizes={name: len(entries[0]['indices']) for name, entries in splits.items()},
                            tensor_sha256=file_digest(path))
    else:
        dataset, schema, task = load_graph(cache, target, feature_policy)
        indices = graph_split_indices(len(dataset), target, seed)
        entries, max_nodes, target_shards = {}, 0, []
        for split, ids in indices.items():
            entries[split], buffer, shard_id = [], [], 0
            for index in ids:
                record = export_graph(dataset[index], target, index, feature_policy, label=True)
                max_nodes = max(max_nodes, len(record['x']))
                entries[split].append({'path': f'target_{split}_{shard_id:05d}.pt', 'index': len(buffer)})
                buffer.append(record)
                if len(buffer) == GRAPH_SHARD_SIZE:
                    target_shards.append(save_shard(output, f'target_{split}', buffer, shard_id))
                    buffer, shard_id = [], shard_id + 1
            if buffer:
                target_shards.append(save_shard(output, f'target_{split}', buffer, shard_id))
        target_manifest = {'task': {'level': 'graph', **task}, 'visibility': 'split_graphs',
                           'feature_schema': schema, 'splits': entries}
        target_audit = {'dataset': target, 'raw_graphs': len(dataset), 'max_nodes': max_nodes,
                        'split_sizes': {k: len(v) for k, v in indices.items()},
                        'selected_indices': indices, 'shards': target_shards,
                        'split': 'Gefra graph_set_partition random 80/10/10; overrides official OGB split'}
    target_manifest.update(schema_version=1, protocol_id=identity, dataset=target)
    write_json(output / 'source_manifest.json', source)
    write_json(output / 'target_manifest.json', target_manifest)
    write_json(output / 'preparation_report.json', {
        'kind': PROTOCOL_VERSION, 'protocol_id': identity, 'seed': seed,
        'target_dataset': target, 'level': level, 'source_datasets': sources,
        'feature_policy': feature_policy, 'source_ineligible_policy': source_ineligible_policy,
        'source_labels_exported': False, 'target_in_source_pretraining': False,
        'sources': audits, 'source_shards': shards, 'target': target_audit,
        'few_shot': 'sorted train IDs; first k per class/task-class; union for PCBA',
        'node_epoch': 'each non-isolated source root once; bounded independent local views',
        'smoke_only': False})
    return output / 'source_manifest.json', output / 'target_manifest.json'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root', type=Path, default=Path('cache_data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--target', choices=GRAPH_DATASETS + NODE_DATASETS, required=True)
    parser.add_argument('--feature-policy', choices=('native', 'gefra_compat'), default='native')
    parser.add_argument('--source-ineligible-policy', choices=('error', 'exclude'), default='exclude')
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    prepare_gefra_loo_data(args.cache_root, args.output, args.seed, args.target,
                          args.feature_policy, args.source_ineligible_policy)


if __name__ == '__main__':
    main()
