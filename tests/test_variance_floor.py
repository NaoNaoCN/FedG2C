"""Check logical-root variance gradients and independent physical chunks."""
from copy import deepcopy
from dataclasses import replace
import os
from unittest.mock import patch

import pytest
import torch

from fedg2c.batching import EdgePredictionBatch
from fedg2c.context import GraphView
from fedg2c.edge_client import EdgePredictionClient
from fedg2c.edge_config import EdgePredictionConfig
from fedg2c.edge_model import variance_floor_loss
from fedg2c.edge_views import NeighborhoodEdgeSampler
from fedg2c.model import SharedBackbone

DEVICE = os.environ.get('FEDG2C_TEST_DEVICE', 'cpu')


def config(**overrides):
    values = dict(edge_mask_rate=0.2, feature_mask_rate=0.2, hidden_dim=8,
                  num_experts=2, num_heads=2, predictor_dim=8, predictor_heads=2,
                  max_nodes=4, walk_steps=(1, 2), undirected=False,
                  variance_weight=0.7, variance_target=0.8, variance_eps=1e-4)
    values.update(overrides)
    return EdgePredictionConfig(**values)


def graph():
    nodes = torch.arange(7)
    edges = torch.stack((nodes, (nodes+1) % len(nodes)))
    return GraphView(torch.randn(7, 4), edges, None, nodes)


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.set_num_threads(1)
    torch.manual_seed(17)


def test_variance_floor_is_translation_invariant_and_has_a_repulsive_gradient():
    roots = torch.tensor([[-0.03, 0.01], [0.03, -0.01]], requires_grad=True)
    loss = variance_floor_loss(roots, target=0.2, eps=1e-4)
    gradient, = torch.autograd.grad(loss, roots)
    moved = roots.detach() - 0.1 * gradient
    assert torch.all(moved.var(0, unbiased=False) > roots.var(0, unbiased=False))
    torch.testing.assert_close(loss, variance_floor_loss(roots+4, 0.2, 1e-4))
    assert variance_floor_loss(torch.ones(1, 3), 0.2, 1e-4) == 0


@pytest.mark.parametrize('representation', ['composed', 'experts', 'both'])
@pytest.mark.parametrize('view', ['visible', 'reference'])
@pytest.mark.parametrize('composer', ['hypernetwork_relu', 'linear_gate'])
def test_two_pass_gradient_and_report_match_one_dense_logical_batch(representation, view, composer):
    cfg = config(variance_representation=representation, variance_view=view,
                 composer=composer, relation_condition=composer != 'linear_gate')
    client = EdgePredictionClient('source', cfg, SharedBackbone(cfg), {'input_dim':4}, DEVICE)
    dense = deepcopy(client.model).train()
    source = graph()
    sampler = NeighborhoodEdgeSampler(source)
    generator = torch.Generator().manual_seed(29)
    order = sampler.eligible_roots[torch.randperm(len(sampler.eligible_roots), generator=generator)].tolist()
    pairs = [sampler.sample_edge_view(cfg, 4, generator, root) for root in order[:5]]
    batch = EdgePredictionBatch.collate(pairs).to(DEVICE)
    prediction_sum, output = dense.forward_batch(batch)
    if view == 'reference':
        roots = dense.main_branch.forward_batch(batch.reference_view, return_experts=representation in ('experts', 'both'))
        expert_roots = roots[1][batch.root_indices] if representation in ('experts', 'both') else None
        roots = (roots[1] if representation == 'experts' else roots[0] if representation == 'both' else roots)[batch.root_indices]
    else:
        roots = output['online_roots']
        expert_roots = output.get('online_expert_roots')
    auxiliary = variance_floor_loss(roots, cfg.variance_target, cfg.variance_eps)
    if representation == 'both':
        auxiliary = auxiliary + variance_floor_loss(expert_roots, cfg.variance_target, cfg.variance_eps)
    expected_loss = prediction_sum / len(pairs) + cfg.variance_weight * auxiliary
    expected_loss.backward()
    expected_gradients = {name:p.grad.detach().clone() for name, p in dense.named_parameters() if p.grad is not None}
    captured = {}
    original_clip = torch.nn.utils.clip_grad_norm_

    def capture(parameters, *args, **kwargs):
        captured.update({name:p.grad.detach().clone() for name, p in client.model.named_parameters() if p.grad is not None})
        return original_clip(parameters, *args, **kwargs)

    with patch('torch.nn.utils.clip_grad_norm_', capture):
        result = client.pretrain(lambda:iter([source]), epochs=1, seed=29, max_steps=1,
                                 batch_size=5, physical_batch_size=2, sampling_unit='node')
    assert captured.keys() == expected_gradients.keys()
    for name, value in captured.items():
        torch.testing.assert_close(value, expected_gradients[name], atol=3e-6, rtol=3e-4, msg=name)
    item = result['steps'][0]
    assert item['loss'] == pytest.approx(float(expected_loss.detach()), abs=2e-6)
    assert item['prediction_loss'] == pytest.approx(float(prediction_sum.detach()/5), abs=2e-6)
    assert item['variance_loss'] == pytest.approx(float(auxiliary.detach()), abs=2e-6)
    assert result['total_views'] == 5
    assert all(p.grad is None for p in client.model.momentum_branch.parameters())


@pytest.mark.parametrize('representation', ['composed', 'experts', 'both'])
@pytest.mark.parametrize('view', ['visible', 'reference'])
def test_physical_chunk_size_keeps_root_statistic_updates_and_epoch_tails(representation, view):
    cfg = config(variance_representation=representation, variance_view=view)
    scalar = EdgePredictionClient('source', cfg, SharedBackbone(cfg), {'input_dim':4}, DEVICE)
    batched = deepcopy(scalar)
    source = graph()
    arguments = dict(epochs=2, seed=13, batch_size=5, sampling_unit='node')
    first = scalar.pretrain(lambda:iter([source]), physical_batch_size=1, **arguments)
    second = batched.pretrain(lambda:iter([source]), physical_batch_size=3, **arguments)
    assert [item['views'] for item in first['steps']] == [5, 2, 5, 2]
    assert first['root_coverage'] == second['root_coverage']
    for left, right in zip(first['steps'], second['steps']):
        for metric in ('loss', 'prediction_loss', 'variance_loss', 'root_std'):
            assert left[metric] == pytest.approx(right[metric], abs=4e-6)
    for name, value in scalar.model.state_dict().items():
        torch.testing.assert_close(value, batched.model.state_dict()[name], atol=3e-5, rtol=2e-3, msg=name)


def test_singleton_batches_skip_auxiliary_and_keep_original_updates():
    cfg = config()
    regularized = EdgePredictionClient('source', cfg, SharedBackbone(cfg), {'input_dim':4}, DEVICE)
    original = EdgePredictionClient('source', replace(cfg, variance_weight=0), SharedBackbone(cfg), {'input_dim':4}, DEVICE)
    original.model.load_state_dict(regularized.model.state_dict())
    source = graph()
    arguments = dict(epochs=1, seed=5, batch_size=1, sampling_unit='node', max_steps=2)
    result = regularized.pretrain(lambda:iter([source]), **arguments)
    original.pretrain(lambda:iter([source]), **arguments)
    assert all(item['variance_loss'] == 0 for item in result['steps'])
    for name, value in regularized.model.state_dict().items():
        torch.testing.assert_close(value, original.model.state_dict()[name],
                                   atol=0 if DEVICE == 'cpu' else 3e-6,
                                   rtol=0 if DEVICE == 'cpu' else 3e-5, msg=name)


def test_rejects_stochastic_replay_and_repeated_or_graph_samples():
    with pytest.raises(ValueError, match='dropout=0'):
        config(dropout=0.1)
    cfg = config()
    client = EdgePredictionClient('source', cfg, SharedBackbone(cfg), {'input_dim':4}, DEVICE)
    with pytest.raises(ValueError, match='sampling_unit=node'):
        client.pretrain(lambda:iter([graph()]), sampling_unit='graph')
    with pytest.raises(ValueError, match='views_per_graph=1'):
        client.pretrain(lambda:iter([graph()]), sampling_unit='node', views_per_graph=2)
