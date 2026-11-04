"""Root queries and cached topology versus the original all-query computation."""
from copy import deepcopy
from dataclasses import replace
import os
from unittest.mock import patch

import pytest
import torch

from fedg2c.batching import GraphBatch
from fedg2c.composer import RelationComposer
from fedg2c.context import GraphView, structural_relations, structural_relation_rows
from fedg2c.edge_config import EdgePredictionConfig
from fedg2c.model import SharedBackbone
from fedg2c.target_data import (TargetDataCache, load_target_manifest, precompute_topology,
                                select_support)
from fedg2c.target_model import TargetConfig, TargetModel
from fedg2c.target_smoke import write_smoke_manifest
from fedg2c.target_train import predict

DEVICE = os.environ.get('FEDG2C_TEST_DEVICE', 'cpu')
MODES = ['hypernetwork_relu', 'hypernetwork_linear', 'standard_attention', 'linear_gate']


@pytest.fixture(autouse=True)
def seed():
    torch.set_num_threads(1)
    torch.manual_seed(71)


def graphs():
    return [GraphView(torch.randn(n, 4),
                      torch.empty(2, 0, dtype=torch.long) if n == 1 else
                      torch.tensor([[0, 0, 1, 2, 2], [1, 1, 0, 1, 2]]),
                      None, torch.arange(n)) for n in (1, 3, 7)]


def compare_gradients(reference, optimized):
    for (name, left), (other, right) in zip(reference.named_parameters(), optimized.named_parameters()):
        assert name == other
        assert (left.grad is None) == (right.grad is None), name
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, atol=3e-5, rtol=3e-4, msg=name)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('condition', [False, True])
def test_root_queries_preserve_logits_routing_input_and_parameter_gradients(mode, condition):
    batch = GraphBatch.collate(graphs()).to(DEVICE)
    roots = torch.tensor([0, 2, 6], device=DEVICE)  # Nonzero root, self edge and isolate.
    rows = torch.arange(3, device=DEVICE)
    full = RelationComposer(8, 2, 2, mode, condition).to(DEVICE)
    optimized = deepcopy(full)
    h = torch.randn(3, batch.max_nodes, 2, 8, device=DEVICE, requires_grad=True)
    hp = h.detach().clone().requires_grad_()
    relations = batch.relations(h.dtype)
    root_relations = batch.root_relations(roots, h.dtype)
    torch.testing.assert_close(root_relations, relations[rows, roots], rtol=0, atol=0)
    gate = torch.rand(3, batch.max_nodes, batch.max_nodes, device=DEVICE)
    z, routing = full.forward_batch(h, relations, batch.valid_nodes, gate, return_routing=True)
    expected = z[rows, roots]
    result, root_routing = optimized.forward_roots(hp, roots, root_relations, batch.valid_nodes,
                                                  gate[rows, roots], return_routing=True)
    torch.testing.assert_close(result, expected, atol=3e-6, rtol=2e-5)
    if mode == 'linear_gate':
        expected_routing = routing[rows, roots]
    else:
        expected_routing = torch.stack([routing[i, :, r*2:(r+1)*2] for i, r in enumerate((0, 2, 6))])
    torch.testing.assert_close(root_routing, expected_routing, atol=3e-6, rtol=2e-5)
    probe = torch.randn_like(expected)
    (expected * probe).sum().backward()
    (result * probe).sum().backward()
    compare_gradients(full, optimized)
    torch.testing.assert_close(h.grad, hp.grad, atol=3e-5, rtol=3e-4)
    if mode != 'linear_gate':
        assert hp.grad[2, :6].abs().sum() > 0  # Neighbour K/V still receive gradients.
    assert torch.count_nonzero(hp.grad[~batch.valid_nodes]) == 0


def original_forward(model, view, root):
    """The pre-optimization all-node target path, independent of forward_roots."""
    x = model.node_adapter(view.x)
    pure_local = model.config.local_alpha == 1 and model.alpha_logit is None
    if not pure_local:
        h = model.shared.encode_experts(x, view.edge_index)
        if model.prompt is not None:
            h = h + model.prompt
        z = model.shared.composer(h, structural_relations(view.edge_index, view.num_nodes, h.dtype))
        if model.adapter is not None:
            z = model.adapter(z)
    if model.local_encoder is not None:
        local = model.local_encoder(x, view.edge_index)
        z = local if pure_local else model.alpha() * local + (1-model.alpha()) * z
    return model.head(z[root:root+1])


@pytest.mark.parametrize('adaptation', ['none', 'lora', 'adapter', 'prompt'])
@pytest.mark.parametrize('alpha,learnable', [(0., False), (.3, False), (.3, True), (1., False)])
def test_target_matches_original_all_query_model_and_gradients(adaptation, alpha, learnable):
    cfg = EdgePredictionConfig(edge_mask_rate=.2, feature_mask_rate=.2,
                               hidden_dim=8, num_experts=2, num_heads=2,
                               predictor_dim=8, predictor_heads=2)
    task = {'level':'node', 'objective':'multiclass', 'num_outputs':3}
    target = TargetConfig(adaptation=adaptation, local_alpha=alpha, learnable_alpha=learnable)
    reference = TargetModel(SharedBackbone(cfg), cfg, {'input_dim':4}, task, target).to(DEVICE).train()
    # Exercise learned adaptations rather than only zero-initialized residuals.
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if 'lora_B' in name or name.startswith('adapter.up') or name == 'prompt':
                parameter.normal_(0, .05)
    scalar, batched = deepcopy(reference), deepcopy(reference)
    inputs = [g.to(DEVICE) for g in graphs()]
    roots = (0, 2, 6)
    expected = torch.cat([original_forward(reference, v, r) for v, r in zip(inputs, roots)])
    scalar_result = torch.cat([scalar(v, r) for v, r in zip(inputs, roots)])
    batch = GraphBatch.collate(graphs())
    # Use exactly the same features as the scalar reference.
    batch.x = torch.cat([v.x.cpu() for v in inputs])
    root_tensor = torch.tensor(roots, device=DEVICE)
    batch = batch.to(DEVICE)
    cached_rows = batch.root_relations(root_tensor, torch.bool)
    actual = batched.forward_batch(batch, root_tensor, cached_rows)
    for result in (scalar_result, actual):
        torch.testing.assert_close(result, expected, atol=4e-6, rtol=3e-5)
    labels = torch.tensor([0, 2, 1], device=DEVICE)
    for result in (expected, scalar_result, actual):
        torch.nn.functional.cross_entropy(result, labels).backward()
    compare_gradients(reference, scalar)
    compare_gradients(reference, batched)
    assert scalar.frozen_base_unchanged() and batched.frozen_base_unchanged()


def test_static_root_relations_keep_nonroot_intermediates_and_directed_duplicates():
    edges = torch.tensor([[0, 0, 1, 2, 2], [1, 1, 0, 1, 2]], device=DEVICE)
    roots = torch.tensor([2, 3, 0], device=DEVICE)
    actual = structural_relation_rows(edges, 4, roots, torch.bool)
    expected = structural_relations(edges, 4)[roots].bool()
    assert torch.equal(actual, expected)
    assert actual[0, 0, 2] and not actual[0, 0, 1]  # 2 -> 1 -> 0 only.
    assert not actual[1, :, 1:].any()  # Isolate has self relation only.


def test_precompute_uses_support_and_validation_only_and_preserves_rng(tmp_path):
    manifest = write_smoke_manifest(tmp_path/'data', level='node')
    cfg = TargetConfig(max_nodes=8, k_shot=1)
    cache = TargetDataCache(1)
    meta, splits = load_target_manifest(manifest, cfg, data_cache=cache)
    splits['train'], support = select_support(splits['train'], meta['task'], 1)
    state = torch.get_rng_state().clone()
    with patch.object(splits['test'], 'topology', side_effect=AssertionError('Opened test topology')):
        report = precompute_topology(splits)
    assert torch.equal(state, torch.get_rng_state())
    assert report['enabled'] and report['working_set_fits']
    assert report['split_sizes'] == {'train':len(support), 'valid':4}
    expected_roots = {r[2] for name in ('train', 'valid') for r in splits[name].refs}
    assert {key[2] for key in cache.crops} == expected_roots
    assert report['working_set_tensor_bytes'] == cache.crop_bytes
    assert cache.crop_bytes == sum(cache.crop_size(crop) for crop, _ in cache.crops.values())
    misses = cache.misses
    for name in ('train', 'valid'):
        for sample in splits[name]:
            expected = structural_relations(sample.view.edge_index, sample.view.num_nodes)[sample.root]
            torch.testing.assert_close(sample.root_relations.float(), expected, rtol=0, atol=0)
    assert cache.misses == misses
    with patch.object(splits['train'], 'config', replace(cfg, precompute_topology=False)):
        assert precompute_topology(splits) == {'enabled':False}


def test_cached_and_disabled_topology_produce_same_target_logits_and_gradients(tmp_path):
    manifest = write_smoke_manifest(tmp_path/'data', level='node')
    cfg = TargetConfig(adaptation='lora', max_nodes=8)
    _, cached = load_target_manifest(manifest, cfg, data_cache=TargetDataCache(1))
    _, uncached = load_target_manifest(manifest, cfg, data_cache=TargetDataCache(0))
    precompute_topology(cached)
    source = EdgePredictionConfig(edge_mask_rate=.2, feature_mask_rate=.2,
                                  hidden_dim=8, num_experts=2, num_heads=2,
                                  predictor_dim=8, predictor_heads=2)
    model = TargetModel(SharedBackbone(source), source, {'input_dim':5}, cached['train'].task, cfg).to(DEVICE)
    other = deepcopy(model)
    # Physical batch and scalar paths both use the cached descriptors.
    for ids in ([0], [1, 2, 3]):
        left, labels = predict(model, [cached['train'][i] for i in ids], DEVICE)
        right, _ = predict(other, [uncached['train'][i] for i in ids], DEVICE)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        torch.nn.functional.cross_entropy(left, labels.flatten()).backward()
        torch.nn.functional.cross_entropy(right, labels.flatten()).backward()
    compare_gradients(model, other)
    assert not uncached['train'].data_cache.crops


def test_small_cache_reports_working_set_overflow_and_counts_relations(tmp_path):
    manifest = write_smoke_manifest(tmp_path/'data', level='node')
    cache = TargetDataCache(1)
    _, splits = load_target_manifest(manifest, TargetConfig(max_nodes=8), data_cache=cache)
    first = splits['train'].topology(0)
    cache.max_bytes = cache.crop_size(first) * 2
    report = precompute_topology(splits)
    assert not report['working_set_fits'] and report['evictions'] > 0
    assert cache.crop_bytes <= cache.max_bytes
    assert cache.crop_bytes == sum(size for _, size in cache.crops.values())


def test_cached_crop_reads_current_features_and_labels(tmp_path):
    manifest = write_smoke_manifest(tmp_path/'data', level='node')
    cache = TargetDataCache(1)
    _, splits = load_target_manifest(manifest, TargetConfig(max_nodes=8), data_cache=cache)
    first = splits['train'][0]
    first_label = first.label.item()
    _, _, root, record = splits['train']._record(0)
    # Mutable input tensors are intentionally never retained in the crop cache.
    record['x'].add_(.25)
    record['y'][root] = 1-record['y'][root]
    second = splits['train'][0]
    torch.testing.assert_close(second.view.x, first.view.x + .25)
    assert second.label.item() != first_label
    assert second.root_relations is first.root_relations
    assert cache.hits == 1
