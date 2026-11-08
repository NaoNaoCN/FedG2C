"""Expand one JSON matrix definition into independent Gefra LODO FedG2C runs."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from .gefra_protocol import GRAPH_SOURCES, NODE_SOURCES, TARGET_BUDGETS, sources_for
from .pipeline import execute, keys, plan, positive, resolve_config

REPO = Path(__file__).resolve().parents[1]


def rooted(path):
    value = Path(path)
    return str(value if value.is_absolute() else (REPO / value).resolve())


def target_config(base, target):
    raw = deepcopy(base)
    matrix = raw.pop('matrix')
    validate_matrix(matrix)
    level, sources = sources_for(target)
    run_root, data_root = rooted(raw['run']['output']), rooted(matrix['data_root'])
    run_output = Path(run_root) / level / target
    data_output = Path(data_root) / level / target
    raw['run']['output'] = str(run_output)
    raw['data'] = {
        'source_manifest': str(data_output / 'source_manifest.json'),
        'target_manifest': str(data_output / 'target_manifest.json'),
        'prepare': {'kind': 'gefra_loo_v1', 'cache_root': rooted(matrix['cache_root']),
                    'output': str(data_output), 'seed': raw['run']['seed'],
                    'target': target, 'feature_policy': matrix['feature_policy'],
                    'source_ineligible_policy': matrix['source_ineligible_policy']},
    }
    training = raw['stage1']['training']
    training['sampling_unit'] = 'node' if level == 'node' else 'graph'
    training['batch_size'] = matrix['source_batch_size'][level]
    raw['stage1']['module_a']['num_experts'] = len(sources)
    raw['stage1']['module_a']['undirected'] = level == 'graph'
    epochs, early_stop, batch_size, lr = TARGET_BUDGETS[target]
    target_cfg = raw['stage2']['target']
    target_cfg.update(epochs=epochs, early_stopping_patience=max(1, early_stop // 10),
                      batch_size=batch_size, lr=lr, eval_interval=10,
                      max_nodes=0 if level == 'graph' else matrix['node_target_max_nodes'],
                      selection_metric='loss' if level == 'graph' else 'accuracy', k_shot=-1)
    # Explicit overrides are applied after Gefra's dataset budgets. The full
    # effective config is persisted by execute() for every target.
    target_cfg.update(matrix.get('target_overrides', {}).get(target, {}))
    training.update(matrix.get('training_overrides', {}).get(target, {}))
    raw['stage2']['k_shots'] = matrix['k_shots']
    raw['stage2']['seeds'] = matrix['fine_tune_seeds']
    cfg = resolve_config(raw)
    return cfg, {'level': level, 'target': target, 'sources': sources,
                 'num_experts': len(sources), 'run_output': str(run_output),
                 'data_output': str(data_output), 'graph_finetune_batch_size':
                 (batch_size or 256) if level == 'graph' else batch_size}


def validate_matrix(matrix):
    required = ('groups', 'data_root', 'cache_root', 'feature_policy', 'source_ineligible_policy',
                'node_target_max_nodes', 'k_shots', 'fine_tune_seeds', 'source_batch_size')
    keys(matrix, required + ('target_overrides','training_overrides'), required, 'matrix')
    if matrix['feature_policy'] not in ('native','gefra_compat'):
        raise ValueError('matrix.feature_policy must be native or gefra_compat')
    if matrix['source_ineligible_policy'] not in ('error','exclude'):
        raise ValueError('matrix.source_ineligible_policy must be error or exclude')
    positive(matrix['node_target_max_nodes'], 'matrix.node_target_max_nodes')
    keys(matrix['source_batch_size'], ('node','graph'), ('node','graph'), 'matrix.source_batch_size')
    for level in ('node','graph'):
        positive(matrix['source_batch_size'][level], f'matrix.source_batch_size.{level}')
    for key in ('target_overrides','training_overrides'):
        for name in matrix.get(key, {}):
            sources_for(name)


def selected_targets(matrix, groups=None, only=None):
    if only:
        result = only
    else:
        groups = groups or matrix['groups']
        if any(group not in ('graph', 'node') for group in groups):
            raise ValueError('matrix.groups must contain graph and/or node')
        result = []
        if 'graph' in groups:
            result.extend(GRAPH_SOURCES)
        if 'node' in groups:
            result.extend(NODE_SOURCES)
    for target in result:
        sources_for(target)
    if len(result) != len(set(result)):
        raise ValueError('Target list contains duplicates')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--targets', nargs='+', help='Subset of configured graph/node targets')
    parser.add_argument('--group', choices=('graph', 'node'), action='append')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    base = json.loads(args.config.read_text(encoding='utf-8'))
    matrix = base.get('matrix')
    if not isinstance(matrix, dict):
        parser.error('Config needs a matrix object')
    validate_matrix(matrix)
    targets = selected_targets(matrix, args.group, args.targets)
    executions = []
    for name in targets:
        cfg, info = target_config(base, name)
        info['execution_plan'] = plan(cfg, 'all')
        executions.append((cfg, info))
    print(json.dumps({'protocol': 'gefra_loo_v1', 'dry_run': args.dry_run,
                      'targets': [info for _, info in executions]}, indent=2), flush=True)
    if args.dry_run:
        return
    for cfg, info in executions:
        print(json.dumps({'stage': 'target_start', **{k: info[k] for k in
                          ('level', 'target', 'sources', 'num_experts', 'run_output')}}), flush=True)
        execute(cfg, 'all', config_path=args.config)


if __name__ == '__main__':
    main()
