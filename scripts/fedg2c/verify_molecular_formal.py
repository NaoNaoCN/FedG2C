"""Audit completed formal runs without evaluating the test set again."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fedg2c.server_io import state_digest
from fedg2c.source_sampling import cycle_indices
from fedg2c.target_model import load_shared
from fedg2c.target_train import load_target_model


def read_json(path):
    return json.loads(Path(path).read_text())


def verify(root):
    config = read_json(root / 'formal_config.json')
    assert (root / 'exit_code').read_text().strip() == '0', 'Run did not finish successfully'
    for name, digest in read_json(root / 'code_sha256.json').items():
        assert hashlib.sha256((root / 'code_snapshot' / name).read_bytes()).hexdigest() == digest, name
    data = Path(config['data_root'])
    identity_path = root / 'molecular_identity_audit.json'
    identity = read_json(identity_path) if identity_path.exists() else None
    if identity is not None:
        assert identity['source_target_overlap']['valid'] == identity['source_target_overlap']['test'] == 0
        assert all(value == 0 for value in identity['target_partition_overlap'].values())
    for name in ('source', 'target'):
        path = data / f'{name}_manifest.json'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == config[f'{name}_manifest_sha256']
    sources = read_json(data / 'source_manifest.json')['clients']
    sizes = {'molpcba': 10000, 'proteins': 1113, 'imdb_binary': 1000}
    coverage = {source['id']: Counter() for source in sources}
    previous = None
    aggregation = []
    for round_id in range(config['rounds']):
        folder = root / 'federated' / f'round_{round_id:03d}'
        uploaded = []
        initial = set()
        for index, source in enumerate(sources):
            local = folder / f'client_{index}'
            metrics = read_json(local / 'metrics.json')
            assert len(metrics['steps']) == config['local_steps']
            assert [step['step'] for step in metrics['steps']] == list(range(1, config['local_steps'] + 1))
            assert all(torch.isfinite(torch.tensor(step['loss'])) for step in metrics['steps'])
            sampling = metrics['sampling']
            assert sampling['pool_size'] == sizes[source['id']]
            expected = cycle_indices(sampling['pool_size'], round_id * config['local_steps'],
                                     config['local_steps'], config['seed'] + index * 104729)
            assert sampling['graph_indices'] == expected
            coverage[source['id']].update(expected)
            payload = torch.load(local / 'shared_backbone.pt', map_location='cpu', weights_only=True)
            assert payload['client_id'] == source['id']
            assert payload['metadata']['device'] == config['device']
            initial.add(payload['metadata']['initial_shared_sha256'])
            uploaded.append(state_digest(payload['state_dict']))
        assert len(initial) == 1
        if previous is not None:
            assert initial == {previous}, 'Round did not receive the previous server checkpoint'
        _, _, metadata = load_shared(folder / 'server' / 'shared_backbone.pt')
        assert metadata['provenance']['uploaded_state_sha256'] == uploaded
        assert metadata['provenance']['initial_shared_sha256'] in initial
        previous = metadata['state_sha256']
        report = read_json(folder / 'server' / 'aggregation_report.json')
        assert report['optimization_steps'] == config['configs']['fedg2c_server_response']['steps']
        assert abs(sum(report['weights']) - 1) < 1e-5
        aggregation.append({'round': round_id + 1, 'weights': report['weights'],
                            'diagnostic_before': report['diagnostic_before'],
                            'diagnostic_after': report['diagnostic_after']})
    coverage_report = {}
    for name, counts in coverage.items():
        assert set(counts) == set(range(sizes[name]))
        assert sum(counts.values()) == config['rounds'] * config['local_steps']
        coverage_report[name] = {'unique_graphs': len(counts), 'graph_visits': sum(counts.values()),
                                 'minimum_visits': min(counts.values()), 'maximum_visits': max(counts.values())}
    targets = {}
    for mode in config['adaptations']:
        folder = root / f'target_{mode}'
        metrics = read_json(folder / 'metrics.json')
        assert len(metrics['history']) == config['target_epochs']
        assert metrics['best_epoch'] == min(metrics['history'], key=lambda item: item['valid']['loss'])['epoch']
        assert metrics['frozen_base_unchanged']
        assert metrics['split_sizes'] == {'train': 4096, 'valid': 1024, 'test': 1024}
        model, metadata = load_target_model(folder / 'target_model.pt')
        assert model.frozen_base_unchanged()
        assert metadata['source_state_sha256'] == previous
        assert metadata['source_round_id'] == config['rounds'] - 1
        assert metadata['config'] == config['effective_target_configs'][mode]
        assert all(torch.isfinite(value).all() for value in model.state_dict().values())
        targets[mode] = {'best_epoch': metrics['best_epoch'], 'test': metrics['test'],
                         'trainable_parameters': metrics['parameters']['trainable_total'],
                         'checkpoint_reload_and_base_hash': 'passed'}
    result = {'verification': 'passed', 'rounds': config['rounds'],
              'local_optimization_steps': sum(item['graph_visits'] for item in coverage_report.values()),
              'coverage': coverage_report, 'targets': targets, 'aggregation': aggregation,
              'molecular_identity': identity,
              'test_evaluation': 'used saved final metrics; no repeat test evaluation',
              'final_shared_sha256': previous}
    (root / 'verification.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    verify(parser.parse_args().run_root)
