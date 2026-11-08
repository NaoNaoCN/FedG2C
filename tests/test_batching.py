"""Compare physical batches with independent scalar forwards and updates."""
from copy import deepcopy
from dataclasses import replace
import os
from unittest.mock import patch

import pytest
import torch

from fedg2c.batching import EdgePredictionBatch, GraphBatch
from fedg2c.composer import RelationComposer
from fedg2c.context import GraphView, structural_relations
from fedg2c.edge_client import EdgePredictionClient
from fedg2c.edge_config import EdgePredictionConfig
from fedg2c.edge_model import NodewisePredictionModel
from fedg2c.edge_views import EdgePredictionPair, NeighborhoodEdgeSampler
from fedg2c.model import SharedBackbone
from fedg2c.structural_encoding import random_walk_encoding
from fedg2c.target_data import TargetDataCache, TargetSample, load_target_manifest, select_support
from fedg2c.target_model import TargetConfig, TargetModel
from fedg2c.target_smoke import write_smoke_manifest
from fedg2c.target_train import classification_loss, predict

DEVICE = os.environ.get('FEDG2C_TEST_DEVICE', 'cpu')


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.set_num_threads(1)
    torch.manual_seed(31)


def config(**overrides):
    values = dict(edge_mask_rate=0.25, feature_mask_rate=0.2, hidden_dim=8,
                  num_experts=2, num_heads=2, predictor_dim=8, predictor_heads=2,
                  predictor_depth=1, max_nodes=4, walk_steps=(1, 2, 3), undirected=False)
    values.update(overrides)
    return EdgePredictionConfig(**values)


def views(categorical=False):
    result = []
    for n in (1, 3, 7):
        # Include duplicate arcs, directed edges, self edges and an isolate.
        edges = (torch.empty((2, 0), dtype=torch.long) if n == 1 else
                 torch.tensor([[0, 0, 1, 2, 2], [1, 1, 0, 1, 2]]))
        x = torch.randint(0, 3, (n, 4)) if categorical else torch.randn(n, 4)
        attrs = torch.randint(0, 2, (edges.shape[1], 2)) if categorical else None
        result.append(GraphView(x, edges, attrs, torch.arange(n) + 100))
    return result


def pairs(categorical=False):
    result = []
    for i, reference in enumerate(views(categorical)):
        visible = GraphView(reference.x, reference.edge_index[:, :2],
                            None if reference.edge_attr is None else reference.edge_attr[:2],
                            reference.original_node_id)
        q = torch.arange((1, 2, 4)[i])
        channels = 8 if categorical else 4
        mask = (torch.arange(channels) + i) % 3 != 0
        result.append(EdgePredictionPair(reference, visible, q, mask, 0,
                                         reference.edge_index.shape[1] - visible.edge_index.shape[1]))
    return result


def assert_gradients(a, b, atol=3e-5, rtol=3e-4):
    for (name, p), (other, q) in zip(a.named_parameters(), b.named_parameters()):
        assert name == other
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=atol, rtol=rtol, msg=name)


def test_topology_walks_and_packed_edges_preserve_independent_graphs():
    graphs = views()
    batch = GraphBatch.collate(graphs).to(DEVICE)
    relation = batch.relations(torch.float32)
    walks = batch.walk_encoding((1, 2, 3, 8), torch.float32)
    for i, graph in enumerate(graphs):
        g = graph.to(DEVICE)
        torch.testing.assert_close(relation[i, :g.num_nodes, :g.num_nodes],
                                   structural_relations(g.edge_index, g.num_nodes))
        start = sum(batch.sizes[:i])
        torch.testing.assert_close(walks[start:start + g.num_nodes],
                                   random_walk_encoding(g.edge_index, g.num_nodes, (1, 2, 3, 8)))
    u, v = batch.edge_index
    assert torch.equal(batch.node_batch[u], batch.node_batch[v])
    assert batch.edge_index.shape[1] == sum(g.edge_index.shape[1] for g in graphs)


@pytest.mark.parametrize('mode', ['hypernetwork_relu', 'hypernetwork_linear', 'standard_attention', 'linear_gate'])
def test_composer_padding_outputs_gradients_and_graph_isolation(mode):
    module = RelationComposer(8, 2, 2, mode, mode != 'linear_gate').to(DEVICE)
    batched = deepcopy(module)
    batch = GraphBatch.collate(views()).to(DEVICE)
    h = torch.randn(batch.num_nodes, 2, 8, device=DEVICE, requires_grad=True)
    hp = h.detach().clone().requires_grad_()
    relations = batch.relations(h.dtype)
    gate = torch.rand(3, batch.max_nodes, batch.max_nodes, device=DEVICE)
    expected = []
    for i, n in enumerate(batch.sizes):
        start = sum(batch.sizes[:i])
        expected.append(module(h[start:start+n], relations[i, :n, :n], gate[i, :n, :n]))
    expected = torch.cat(expected)
    result = batched.forward_batch(batch.pad(hp), relations, batch.valid_nodes, gate)
    actual = batch.unpad(result)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=1e-5)
    assert torch.count_nonzero(result[~batch.valid_nodes]) == 0
    expected.square().sum().backward()
    actual.square().sum().backward()
    assert_gradients(module, batched)
    torch.testing.assert_close(h.grad, hp.grad, atol=3e-5, rtol=3e-4)
    # Changing another graph or arbitrary padded tokens cannot affect graph 0.
    altered = batch.pad(hp.detach())
    altered[1:] = torch.randn_like(altered[1:]) * 100
    isolated = batched.forward_batch(altered, relations, batch.valid_nodes, gate)
    torch.testing.assert_close(isolated[0, 0], result[0, 0])


@pytest.mark.parametrize('predictor', ['graphsage', 'transformer'])
@pytest.mark.parametrize('categorical', [False, True])
def test_pretraining_matches_scalar_per_view_loss_and_gradients(predictor, categorical):
    cfg = config(predictor_type=predictor)
    schema = ({'input_dim':4, 'edge_dim':2, 'node_cardinalities':[3]*4, 'edge_cardinalities':[2]*2}
              if categorical else {'input_dim':4})
    scalar = NodewisePredictionModel(cfg, SharedBackbone(cfg), **schema).to(DEVICE).train()
    batched = deepcopy(scalar)
    inputs = pairs(categorical)
    losses, predictions, references, context = [], [], [], []
    for pair in inputs:
        loss, out = scalar(pair.to(DEVICE))
        losses.append(loss)
        predictions.append(out['prediction'])
        references.append(out['reference'])
        context.append(out['context_std'])
    torch.stack(losses).sum().backward()
    loss, out = batched.forward_batch(EdgePredictionBatch.collate(inputs).to(DEVICE))
    torch.testing.assert_close(out['losses'], torch.stack(losses), atol=3e-6, rtol=1e-5)
    torch.testing.assert_close(out['prediction'], torch.cat(predictions), atol=5e-6, rtol=2e-5)
    torch.testing.assert_close(out['reference'], torch.cat(references), atol=5e-6, rtol=2e-5)
    torch.testing.assert_close(out['context_std'], torch.stack(context), atol=5e-6, rtol=2e-5)
    torch.testing.assert_close(loss, torch.stack(losses).sum())
    loss.backward()
    assert_gradients(scalar, batched)
    assert all(p.grad is None for p in batched.momentum_branch.parameters())


@pytest.mark.parametrize('sampling_unit', ['node', 'graph'])
def test_client_keeps_rng_updates_ema_and_each_epoch_tail(sampling_unit):
    cfg = config()
    base = EdgePredictionClient('client', cfg, SharedBackbone(cfg), {'input_dim':4}, DEVICE)
    batched = deepcopy(base)
    graphs = [views()[-1]] if sampling_unit == 'node' else views()[1:]
    if sampling_unit == 'node':
        u = torch.arange(7)
        graphs[0].edge_index = torch.stack((u, (u + 1) % 7))
    factory = lambda: iter(graphs)
    sampled = []
    original = NeighborhoodEdgeSampler.sample_edge_view

    def traced(self, *args, **kwargs):
        pair = original(self, *args, **kwargs)
        sampled.append((pair.reference_view.original_node_id.tolist(), pair.visible_view.edge_index.tolist(),
                        pair.prediction_indices.tolist(), pair.feature_keep.tolist()))
        return pair

    kwargs = dict(epochs=2, seed=19, batch_size=5, sampling_unit=sampling_unit,
                  views_per_graph=1 if sampling_unit == 'node' else 3)
    with patch.object(NeighborhoodEdgeSampler, 'sample_edge_view', traced):
        a = base.pretrain(factory, **kwargs)
        before = list(sampled)
        sampled.clear()
        with patch.object(batched.model, 'update_momentum', wraps=batched.model.update_momentum) as ema:
            b = batched.pretrain(factory, physical_batch_size=3, **kwargs)
    assert sampled == before
    expected_views = [5, 2, 5, 2] if sampling_unit == 'node' else [5, 1, 5, 1]
    assert [s['views'] for s in a['steps']] == expected_views
    assert [s['views'] for s in b['steps']] == expected_views
    assert a['root_coverage'] == b['root_coverage']
    assert a['total_views'] == b['total_views']
    assert ema.call_count == len(expected_views)
    for name, state in base.model.state_dict().items():
        torch.testing.assert_close(state, batched.model.state_dict()[name], atol=5e-5, rtol=1e-3, msg=name)


def test_max_steps_stops_without_sampling_extra_views():
    cfg = config()
    client = EdgePredictionClient('client', cfg, SharedBackbone(cfg), {'input_dim':4}, DEVICE)
    graph = views()[-1]
    with patch.object(NeighborhoodEdgeSampler, 'sample_edge_view', autospec=True,
                      side_effect=NeighborhoodEdgeSampler.sample_edge_view) as sample:
        result = client.pretrain(lambda: iter([graph]), epochs=10, views_per_graph=3, seed=19,
                                 batch_size=5, physical_batch_size=3, max_steps=1, sampling_unit='node')
    assert result['total_views'] == 5
    assert len(result['steps']) == 1
    assert sample.call_count == 5


@pytest.mark.parametrize('mode', ['none', 'lora', 'adapter', 'prompt'])
@pytest.mark.parametrize('level', ['node', 'graph'])
def test_target_logits_loss_gradients_and_frozen_base(mode, level):
    cfg = config()
    task = {'level':level, 'objective':'multiclass', 'num_outputs':2}
    target = TargetConfig(adaptation=mode, microbatch_size=3, local_alpha=0.3)
    scalar = TargetModel(SharedBackbone(cfg), cfg, {'input_dim':4}, task, target).to(DEVICE).train()
    batched = deepcopy(scalar)
    samples = [TargetSample(v, torch.tensor([i % 2]), 0 if level == 'node' else None, str(i))
               for i, v in enumerate(views())]
    expected = torch.cat([predict(scalar, [s], DEVICE)[0] for s in samples])
    actual, labels = predict(batched, samples, DEVICE)
    torch.testing.assert_close(actual, expected, atol=4e-6, rtol=2e-5)
    classification_loss(expected, labels, 'multiclass')[0].backward()
    classification_loss(actual, labels, 'multiclass')[0].backward()
    assert_gradients(scalar, batched)
    assert scalar.frozen_base_unchanged() and batched.frozen_base_unchanged()


def test_multilabel_missing_labels_keep_logical_loss_weighting():
    cfg = config()
    task = {'level':'graph', 'objective':'multilabel', 'num_outputs':3}
    scalar = TargetModel(SharedBackbone(cfg), cfg, {'input_dim':4}, task,
                         TargetConfig(adaptation='lora')).to(DEVICE).train()
    batched = deepcopy(scalar)
    labels = (torch.tensor([1., float('nan'), 0.]),
              torch.full((3,), float('nan')), torch.tensor([0., 1., float('nan')]))
    samples = [TargetSample(v, y, None, str(i)) for i, (v, y) in enumerate(zip(views(), labels))]
    total = 0
    for sample in samples:
        logits, y = predict(scalar, [sample], DEVICE)
        loss, count = classification_loss(logits, y, 'multilabel')
        (loss * count / 4).backward()
        total += count
    assert total == 4
    logits, y = predict(batched, samples, DEVICE)
    loss, count = classification_loss(logits, y, 'multilabel')
    assert count == 4
    loss.backward()
    assert_gradients(scalar, batched)


def test_topology_cache_is_shared_bounded_and_keeps_support_and_labels(tmp_path):
    manifest = write_smoke_manifest(tmp_path / 'target', level='node')
    cache = TargetDataCache(1)
    cfg = TargetConfig(max_nodes=8)
    task, splits = load_target_manifest(manifest, cfg, data_cache=cache)
    original = splits['train'][0]
    before = cache.misses
    _, again = load_target_manifest(manifest, replace(cfg, seed=99), data_cache=cache)
    cached = again['train'][0]
    assert cache.misses == before and cache.hits == 1
    assert torch.equal(original.view.x, cached.view.x)
    assert torch.equal(original.view.edge_index, cached.view.edge_index)
    assert torch.equal(original.label, cached.label) and original.root == cached.root
    support, ids = select_support(again['train'], task['task'], 1)
    assert support.data_cache is cache
    assert all(torch.equal(support[i].label, again['train'].label(j)) for i, j in enumerate(ids))
    assert cache.crop_bytes <= cache.max_bytes
    # Only topology and the root's binary relation row, never features/labels.
    assert all(len(entry[0]) == 5 and entry[0][4].dtype == torch.bool
               for entry in cache.crops.values())
    cache.max_bytes = next(iter(cache.crops.values()))[1]
    again['train'][2]
    assert cache.crop_bytes <= cache.max_bytes and len(cache.crops) <= 1
    disabled = TargetDataCache(0)
    _, uncached = load_target_manifest(manifest, cfg, data_cache=disabled)
    uncached['train'][0]
    uncached['train'][0]
    assert disabled.misses == 2 and not disabled.crops


def test_nonzero_dropout_requires_scalar_pretraining():
    cfg = config(dropout=0.1)
    client = EdgePredictionClient('client', cfg, SharedBackbone(cfg), {'input_dim':4}, DEVICE)
    with pytest.raises(ValueError, match='dropout'):
        client.pretrain(lambda: iter(views()[1:]), batch_size=4, physical_batch_size=2)
