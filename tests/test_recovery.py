"""Interrupted training must reproduce uninterrupted private and shared states."""
from dataclasses import asdict
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from fedg2c.edge_config import EdgePredictionConfig
from fedg2c.model import SharedBackbone
from fedg2c.recovery import atomic_save
from fedg2c.server_config import ResponseAggregationConfig
from fedg2c.server_io import RELATION_SCHEMA, state_digest
from fedg2c.target_model import TargetConfig
from fedg2c.target_smoke import write_smoke_manifest
from fedg2c.target_train import run_target
from pretrain_federated_fedg2c import main as pretrain


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def small_config():
    return EdgePredictionConfig(edge_mask_rate=.2, feature_mask_rate=.2, hidden_dim=8,
        num_experts=2, num_heads=2, predictor_dim=8, predictor_heads=2, max_nodes=4,
        walk_steps=(1, 2), variance_weight=1., variance_target=.15,
        variance_representation='both', variance_view='reference')


@pytest.mark.parametrize('mode,composer', [('response','hypernetwork_relu'), ('sample_weighted','hypernetwork_relu'), ('sample_weighted','linear_gate')])
def test_pretraining_recovers_private_states_after_an_interrupted_client(tmp_path,mode,composer):
    torch.set_num_threads(1)
    cfg = tmp_path/'module_a.json'
    raw = small_config().to_dict()
    raw.update(composer=composer, relation_condition=composer != 'linear_gate', private_adapter_bias_init='zero')
    cfg.write_text(json.dumps(raw))
    server = tmp_path/'module_b.json'
    server_values = json.loads((Path(__file__).resolve().parents[1]/'config/fedg2c_server_response.json').read_text())
    server_values.update(mode=mode, steps=1, calibration_graphs=1, diagnostic_graphs=1, readout_pairs=2, directions=1)
    server.write_text(json.dumps(server_values))
    common = ['--smoke', '--config', str(cfg), '--server-config', str(server),
              '--rounds', '3', '--local-epochs', '2', '--batch-size', '5',
              '--physical-batch-size', '2', '--sampling-unit', 'node',
              '--source-order', 'shuffled_epoch', '--recover']
    pretrain(common + ['--output', str(tmp_path/'reference')])
    calls = 0
    def interrupted(value, path):
        nonlocal calls
        atomic_save(value, path)
        calls += 1
        if calls == 3:
            raise RuntimeError('simulated worker interruption')
    with patch('fedg2c.recovery.atomic_save', interrupted):
        with pytest.raises(RuntimeError, match='simulated'):
            pretrain(common + ['--output', str(tmp_path/'resumed')])
    pretrain(common + ['--output', str(tmp_path/'resumed')])
    for filename in ['recovery.pt', 'round_002/server/shared_backbone.pt',
                     'client_0_private_training_state.pt', 'client_1_private_training_state.pt']:
        values = [torch.load(tmp_path/name/filename, map_location='cpu', weights_only=False)
                  for name in ['reference', 'resumed']]
        if filename == 'recovery.pt':
            values[0].pop('signature'); values[1].pop('signature')
        equal(*values)


@pytest.mark.parametrize('adaptation', ['lora', 'none', 'adapter', 'prompt'])
def test_target_recovers_adam_rng_best_model_and_early_stopping(tmp_path, adaptation):
    torch.set_num_threads(1)
    torch.manual_seed(3)
    cfg = small_config()
    state = SharedBackbone(cfg).state_dict()
    checkpoint = tmp_path/'shared.pt'
    torch.save(dict(metadata=dict(schema_version=3, shared_schema_version=1,
        stage='server_aggregation', config=cfg.to_dict(), relation_schema=RELATION_SCHEMA,
        state_sha256=state_digest(state), protocol_id='smoke', round_id=2), state_dict=state), checkpoint)
    manifest = write_smoke_manifest(tmp_path/'data', level='node')
    target = TargetConfig(adaptation=adaptation, epochs=7, eval_interval=2, max_nodes=5,
                          batch_size=0, microbatch_size=2, eval_batch_size=2,
                          early_stopping_patience=2, selection_metric='accuracy', evaluate_test=False)
    first = run_target(checkpoint, manifest, target, tmp_path/'reference', recover=True)
    calls = 0
    def interrupted(value, path):
        nonlocal calls
        atomic_save(value, path)
        calls += 1
        if calls == 2:
            raise RuntimeError('simulated worker interruption')
    with patch('fedg2c.recovery.atomic_save', interrupted):
        with pytest.raises(RuntimeError, match='simulated'):
            run_target(checkpoint, manifest, target, tmp_path/'resumed', recover=True)
    second = run_target(checkpoint, manifest, target, tmp_path/'resumed', recover=True)
    for key in ['best_epoch', 'epochs_ran', 'history', 'optimizer_steps', 'validation', 'support_sha256']:
        equal(first[key], second[key])
    equal(*[torch.load(tmp_path/name/'target_model.pt', weights_only=True)
            for name in ['reference', 'resumed']])
