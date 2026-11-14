"""Plan three node ablations and optionally write configs; never start training."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from .gefra_loo_matrix import selected_targets, target_config
from .pipeline import plan, write_json

VARIANTS = ('baseline', 'ablate_b', 'ablate_ab')
TARGETS = ('cora', 'citeseer', 'pubmed', 'photo', 'computers', 'wikics')


def apply_variant(raw, variant):
    if variant not in VARIANTS:
        raise ValueError('Use baseline, ablate_b or ablate_ab; A-only is not defined')
    raw = deepcopy(raw)
    a, b = raw['stage1']['module_a'], raw['stage1']['module_b']
    if (a.get('private_adapter_bias_init') != 'zero'
            or a.get('variance_weight') != 1.0 or a.get('variance_target') != 0.15
            or a.get('variance_representation') != 'both' or a.get('variance_view') != 'reference'):
        raise ValueError('All ablation variants require zero private bias and both/reference variance (1, 0.15)')
    if raw['stage2']['adaptations'] != ['lora']:
        raise ValueError('Node ablations require LoRA only')
    a.update(composer='linear_gate' if variant == 'ablate_ab' else 'hypernetwork_relu',
             relation_condition=variant != 'ablate_ab')
    b['mode'] = 'response' if variant == 'baseline' else 'sample_weighted'
    return raw


def build_plan(base, targets=None):
    targets = selected_targets(base['matrix'], only=targets if targets is not None else list(TARGETS))
    if any(target not in TARGETS for target in targets):
        raise ValueError('This ablation plan supports the six node classification targets only')
    configs, entries = {}, []
    for variant in VARIANTS:
        raw = apply_variant(base, variant)
        raw['run']['output'] = str(Path(raw['run']['output']) / variant)
        for target in targets:
            cfg, info = target_config(raw, target)
            name = variant + '_' + target
            execution = plan(cfg, 'all')
            configs[name] = cfg
            entries.append(dict(variant=variant, target=target, sources=info['sources'],
                                composer=cfg['stage1']['module_a']['composer'],
                                aggregation=cfg['stage1']['module_b']['mode'],
                                lora_targets=['gate.0', 'gate.2'] if variant == 'ablate_ab' else ['query', 'key'],
                                checkpoint=execution['checkpoint'],
                                downstream_runs=len(execution['target_runs'])))
    return configs, dict(plan_only=True, training_started=False, variants=list(VARIANTS),
                         targets=targets, pretraining_runs=len(entries),
                         downstream_runs=sum(e['downstream_runs'] for e in entries),
                         baseline_policy='Retrain with zero private input bias; old baseline is historical reference only',
                         comparisons=[dict(left='baseline', right='ablate_b', effect='B under the original A'),
                                      dict(left='ablate_b', right='ablate_ab', effect='A under sample-weighted aggregation')],
                         entries=entries)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--targets', nargs='+')
    parser.add_argument('--data-root', help='Reuse prepared source/target data outside the checkout')
    parser.add_argument('--write-configs', type=Path, help='New output directory; writes plans/configs only')
    args = parser.parse_args(argv)
    base = json.loads(args.config.read_text(encoding='utf-8'))
    if args.data_root:
        base['matrix']['data_root'] = str(Path(args.data_root).resolve())
    configs, report = build_plan(base, args.targets)
    if args.write_configs:
        args.write_configs.mkdir(parents=True, exist_ok=False)
        for name, cfg in configs.items():
            write_json(args.write_configs / (name + '.json'), cfg)
        write_json(args.write_configs / 'plan.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
