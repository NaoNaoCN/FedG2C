"""Ablation controls: source counts, private initialization, LoRA and pipeline."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from fedg2c.context import GraphView
from fedg2c.data import count_source_samples
from fedg2c.edge_client import EdgePredictionClient
from fedg2c.edge_config import EdgePredictionConfig
from fedg2c.messages import client_message
from fedg2c.model import FeatureAdapter, SharedBackbone
from fedg2c.node_ablation_matrix import VARIANTS, TARGETS, apply_variant, build_plan
from fedg2c.pipeline import REPO, execute, preflight, resolve_config
from fedg2c.response_server import ResponsePreservingServer
from fedg2c.server_config import ResponseAggregationConfig
from fedg2c.server_io import state_digest, load_uploads
from fedg2c.target_model import TargetConfig, TargetModel
from fedg2c.target_smoke import write_smoke_manifest as write_target
from fedg2c.target_train import load_target_model


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.set_num_threads(1)
    torch.manual_seed(31)


def small_config(**overrides):
    raw = dict(edge_mask_rate=0.2, feature_mask_rate=0.2, hidden_dim=8,
               num_experts=2, num_heads=2, num_layers=1, predictor_dim=8,
               predictor_heads=2, max_nodes=4, walk_steps=(1, 2), undirected=False)
    raw.update(overrides)
    return EdgePredictionConfig(**raw)


def server_config(mode='sample_weighted'):
    raw = json.loads((REPO/'config/fedg2c_server_response.json').read_text())
    raw.update(mode=mode, block_sizes=[2, 2], edge_probs=[[0.5, 0.1], [0.1, 0.5]],
               calibration_graphs=1, diagnostic_graphs=1, readout_pairs=2,
               directions=1, steps=2)
    return ResponseAggregationConfig(**raw)


def test_private_bias_is_zero_trainable_rng_preserving_and_copied_to_ema():
    original = small_config()
    zero = replace(original, private_adapter_bias_init='zero')
    torch.manual_seed(8)
    old = EdgePredictionClient('s', original, SharedBackbone(original), {'input_dim': 5})
    rng = torch.get_rng_state().clone()
    torch.manual_seed(8)
    new = EdgePredictionClient('s', zero, SharedBackbone(zero), {'input_dim': 5})
    assert torch.equal(rng, torch.get_rng_state())
    changed = set()
    for name, tensor in old.model.state_dict().items():
        other = new.model.state_dict()[name]
        if not torch.equal(tensor, other):
            changed.add(name)
            assert torch.count_nonzero(other) == 0
    assert changed == {'main_branch.node_adapter.projection.bias',
                       'momentum_branch.node_adapter.projection.bias'}
    main, ema = new.model.main_branch, new.model.momentum_branch
    assert main.node_adapter.projection.bias.requires_grad
    for name, value in main.state_dict().items():
        assert torch.equal(value, ema.state_dict()[name])
    optimizer = torch.optim.SGD(main.node_adapter.parameters(), lr=0.1)
    main.node_adapter(torch.ones(2, 5)).sum().backward()
    optimizer.step()
    assert torch.count_nonzero(main.node_adapter.projection.bias) == 8
    # Downstream adapters retain the historical constructor initialization.
    target = TargetModel(deepcopy(main.shared), zero, {'input_dim':5},
                         {'level':'node', 'objective':'multiclass', 'num_outputs':2}, TargetConfig())
    assert torch.count_nonzero(target.node_adapter.projection.bias) > 0


def test_category_adapters_are_unchanged_by_bias_policy():
    torch.manual_seed(9)
    first = FeatureAdapter(2, 8, [3, 4])
    torch.manual_seed(9)
    second = FeatureAdapter(2, 8, [3, 4], bias_init='zero')
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])


def test_sample_weights_use_message_counts_and_never_evaluate_probes():
    cfg = small_config()
    shared = SharedBackbone(cfg)
    first = deepcopy(shared.state_dict())
    second = {k:v + 3 for k,v in first.items()}
    uploads = {'large': client_message(first, 9), 'small': client_message(second, 3)}
    server = ResponsePreservingServer(shared, server_config())
    with patch('fedg2c.response_server.sample_probe_bank', side_effect=AssertionError('No probes')):
        state, report = server.execute(uploads)
    assert report['weights'] == [0.75, 0.25]
    assert report['sample_counts'] == [9, 3]
    assert report['client_ids'] == ['large', 'small']
    assert report['optimization_steps'] == 0 and server.last_probe_banks is None
    for name in first:
        torch.testing.assert_close(state[name], 0.75*first[name] + 0.25*second[name])
    uniform, report = ResponsePreservingServer(shared, server_config('fedavg')).execute(uploads)
    assert report['weights'] == [0.5, 0.5]
    for name in first:
        torch.testing.assert_close(uniform[name], 0.5*first[name] + 0.5*second[name])


@pytest.mark.parametrize('counts', [None, [], [1], [1, 0], [1, True], [1, 1.2]])
def test_sample_weighted_requires_real_positive_integer_counts(counts):
    shared = SharedBackbone(small_config())
    server = ResponsePreservingServer(shared, server_config())
    with pytest.raises(ValueError, match='sample count'):
        server.execute([shared.state_dict(), shared.state_dict()], sample_counts=counts)


def test_source_count_matches_eligible_roots_and_excludes_isolated_or_self_only_nodes(tmp_path):
    graph = {'x':torch.randn(7, 5),
             'edge_index':torch.tensor([[0, 1, 2, 3, 3], [1, 0, 2, 4, 4]])}
    path = tmp_path/'source.pt'
    torch.save(graph, path)
    assert count_source_samples([path], 'node') == 4
    assert count_source_samples([path], 'graph') == 1
    torch.save([graph, graph], path)
    assert count_source_samples([path], 'graph') == 2
    assert count_source_samples([path], 'node') == 8


@pytest.mark.parametrize('mode', ['hypernetwork_relu', 'linear_gate'])
def test_lora_starts_at_base_and_updates_both_targets_with_frozen_base(mode):
    cfg = small_config(composer=mode, relation_condition=mode != 'linear_gate')
    shared = SharedBackbone(cfg)
    task = {'level':'node', 'objective':'multiclass', 'num_outputs':2}
    torch.manual_seed(23)
    plain = TargetModel(deepcopy(shared), cfg, {'input_dim':5}, task, TargetConfig(adaptation='none'))
    torch.manual_seed(23)
    model = TargetModel(deepcopy(shared), cfg, {'input_dim':5}, task,
                        TargetConfig(adaptation='lora', rank=2, lora_alpha=2))
    roots = torch.arange(6)
    edges = torch.stack((roots, (roots+1)%6))
    view = GraphView(torch.randn(6,5), edges, None, roots)
    torch.testing.assert_close(model(view, root=0), plain(view, root=0), rtol=0, atol=0)
    expected = (['shared.composer.gate.0','shared.composer.gate.2'] if mode == 'linear_gate'
                else ['shared.composer.query','shared.composer.key'])
    assert model.parameter_report()['lora_targets'] == expected
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(view, root=0), torch.tensor([1])).backward()
        optimizer.step()
    for name in expected:
        module = model.get_submodule(name)
        assert torch.count_nonzero(module.lora_B) > 0
    assert model.frozen_base_unchanged()


def test_planner_has_only_three_variants_six_targets_and_lora_without_execution():
    base = json.loads((REPO/'config/fedg2c_nodes_ablation_20261114.json').read_text())
    with patch('fedg2c.pipeline.execute', side_effect=AssertionError('Planner must not train')):
        configs, report = build_plan(base)
    assert report['training_started'] is False and report['plan_only']
    assert report['pretraining_runs'] == 18 and report['downstream_runs'] == 540
    assert report['targets'] == list(TARGETS) and report['variants'] == list(VARIANTS)
    for target in TARGETS:
        original, b, ab = [configs[v+'_'+target] for v in VARIANTS]
        assert original['stage1']['module_a'] == b['stage1']['module_a']
        assert original['stage1']['training'] == b['stage1']['training'] == ab['stage1']['training']
        assert original['stage2'] == b['stage2'] == ab['stage2']
        assert original['stage1']['module_b']['mode'] == 'response'
        assert b['stage1']['module_b']['mode'] == ab['stage1']['module_b']['mode'] == 'sample_weighted'
        assert ab['stage1']['module_a']['composer'] == 'linear_gate'
        for cfg in (original, b, ab):
            assert cfg['stage2']['adaptations'] == ['lora']
            assert cfg['stage2']['target']['evaluate_test'] is False
            assert cfg['stage2']['target']['early_stopping_patience'] == 15
    with pytest.raises(ValueError, match='A-only'):
        apply_variant(base, 'ablate_a')


def synthetic_config(tmp_path, variant):
    records=[]
    for i,n in enumerate((5,9)):
        nodes=torch.arange(n)
        edges=torch.stack((nodes,(nodes+1)%n))
        torch.save({'x':torch.randn(n,5),'edge_index':edges},tmp_path/('source'+str(i)+'.pt'))
        records.append({'id':str(i),'feature_schema':{'input_dim':5},'graphs':['source'+str(i)+'.pt']})
    manifest=tmp_path/'source_manifest.json'
    manifest.write_text(json.dumps({'schema_version':1,'protocol_id':'synthetic-ablation-only','clients':records}))
    target=write_target(tmp_path/'target',level='node')
    raw=json.loads((REPO/'config/fedg2c_molecular_pipeline.json').read_text())
    raw['run'].update(output=str(tmp_path/'pipeline'),device='cpu',threads=1)
    raw['data']={'source_manifest':str(manifest),'target_manifest':str(target)}
    raw['stage1']['training'].update(rounds=2,local_budget={'unit':'epochs','amount':1},
                                      views_per_graph=1,batch_size=4,physical_batch_size=2,sampling_unit='node',log_every=0)
    raw['stage1']['module_a']=small_config(private_adapter_bias_init='zero',variance_weight=1.0,
                        variance_target=0.15,variance_representation='both',variance_view='reference').to_dict()
    raw['stage1']['module_b']=server_config().to_dict()
    raw['stage2']['adaptations']=['lora']
    raw['stage2']['target'].update(epochs=3,batch_size=4,microbatch_size=2,eval_batch_size=2,
                                  max_nodes=6,eval_interval=1,early_stopping_patience=2,evaluate_test=False,
                                  selection_metric='accuracy')
    return resolve_config(apply_variant(raw,variant))


@pytest.mark.parametrize('variant', VARIANTS)
def test_two_round_synthetic_pipeline_lora_save_reload_and_source_count_audit(tmp_path,variant):
    cfg=synthetic_config(tmp_path,variant)
    execute(cfg)
    root=Path(cfg['run']['output'])
    assert json.loads((root/'status.json').read_text())['state']=='completed'
    summary=json.loads((root/'summary.json').read_text())['lora']
    assert summary['frozen_base_unchanged'] and summary['test_evaluated'] is False
    model,_=load_target_model(root/'stage2/lora/target_model.pt')
    assert model.frozen_base_unchanged()
    assert len(model.parameter_report()['lora_targets'])==2
    for metrics in (root/'stage1').glob('round_*/client_*/metrics.json'):
        data=json.loads(metrics.read_text())
        assert data['variance_regularization']['representation']=='both'
        assert data['variance_regularization']['view']=='reference'
        assert all(c['complete'] for c in data['root_coverage'])
    if variant!='baseline':
        # The independent aggregation CLI must reproduce the federated path
        # from saved client counts, and require counts for legacy checkpoints.
        from aggregate_fedg2c import main as aggregate
        local_paths=sorted((root/'stage1/round_001').glob('client_*/shared_backbone.pt'))
        _,_,provenance=load_uploads(local_paths)
        assert provenance['sample_counts']==[5,9]
        server_config_path=root/'module_b.json'
        aggregate(['--client-checkpoints', *map(str,local_paths),
                   '--server-config',str(server_config_path),'--output',str(tmp_path/'offline')])
        online=torch.load(root/'stage1/round_001/server/shared_backbone.pt',weights_only=True)
        offline=torch.load(tmp_path/'offline/shared_backbone.pt',weights_only=True)
        assert state_digest(online['state_dict'])==state_digest(offline['state_dict'])
        legacy=[]
        for i,path in enumerate(local_paths):
            payload=torch.load(path,weights_only=True)
            payload['metadata'].pop('num_samples')
            payload['metadata'].pop('sample_unit')
            dest=tmp_path/f'legacy_{i}.pt'
            torch.save(payload,dest)
            legacy.append(str(dest))
        args=['--client-checkpoints',*legacy,'--server-config',str(server_config_path)]
        with pytest.raises(SystemExit):
            aggregate([*args,'--output',str(tmp_path/'missing_counts')])
        assert not (tmp_path/'missing_counts').exists()
        aggregate([*args,'--client-sample-counts','5','9','--output',str(tmp_path/'explicit_counts')])
        explicit=torch.load(tmp_path/'explicit_counts/shared_backbone.pt',weights_only=True)
        assert state_digest(explicit['state_dict'])==state_digest(online['state_dict'])
        for server in (root/'stage1').glob('round_*/server/aggregation_report.json'):

            report=json.loads(server.read_text())
            assert report['sample_counts']==[5,9]
            assert report['weights']==pytest.approx([5/14,9/14])
            assert report['optimization_steps']==0
