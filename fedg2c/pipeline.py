# ===== FedG2C ADD BEGIN A/B/C: unified two-stage experiment configuration =====
"""One JSON config for federated pretraining and independent target adaptations."""
import argparse
from copy import deepcopy
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import traceback

from .edge_config import EdgePredictionConfig
from .server_config import ResponseAggregationConfig
from .target_model import TargetConfig

REPO = Path(__file__).resolve().parents[1]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def keys(value, allowed, required, where):
    if not isinstance(value, dict):
        raise ValueError(f'{where} must be an object')
    extra, missing = set(value)-set(allowed), set(required)-set(value)
    if extra or missing:
        raise ValueError(f'{where}: unknown keys {sorted(extra)}, missing keys {sorted(missing)}')


def positive(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def path_from_repo(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Paths must be nonempty strings')
    path = Path(value)
    return str((REPO / path).resolve())


def resolve_config(path):
    cfg = (deepcopy(path) if isinstance(path, dict)
           else json.loads(Path(path).read_text(encoding='utf-8')))
    keys(cfg, ('schema_version','run','data','stage1','stage2'),
         ('schema_version','run','data','stage1','stage2'), 'pipeline')
    if cfg['schema_version'] != 1:
        raise ValueError('Require pipeline schema_version=1')
    run, data, s1, s2 = (cfg[k] for k in ('run','data','stage1','stage2'))
    keys(run, ('output','seed','device','threads'), ('output','seed','device','threads'), 'run')
    positive(run['seed'], 'seed', 0)
    positive(run['threads'], 'threads')
    if not isinstance(run['device'], str) or not run['device']:
        raise ValueError('run.device must be a device string')
    run['output'] = path_from_repo(run['output'])
    keys(data, ('source_manifest','target_manifest','prepare'), ('source_manifest','target_manifest'), 'data')
    for name in ('source_manifest','target_manifest'):
        data[name] = path_from_repo(data[name])
    prep = data.setdefault('prepare', None)
    if prep is not None:
        keys(prep, ('kind','cache_root','output','seed','target','feature_policy','source_ineligible_policy'),
             ('kind','cache_root','output','seed'), 'data.prepare')
        if prep['kind'] not in ('molecular_m1', 'gefra_loo_v1'):
            raise ValueError('Unsupported data preparation protocol')
        positive(prep['seed'], 'data.prepare.seed', 0)
        for name in ('cache_root','output'):
            prep[name] = path_from_repo(prep[name])
        if prep['kind'] == 'gefra_loo_v1':
            if prep.get('target') not in {
                'chemhiv','chempcba','proteins','imdb-binary','cora','citeseer','pubmed',
                'photo','computers','wikics','arxiv','reddit','airport','disease'}:
                raise ValueError('gefra_loo_v1 requires an explicit graph/node target')
            if prep.get('feature_policy', 'native') not in ('native', 'gefra_compat'):
                raise ValueError('Unsupported Gefra feature_policy')
            if prep.get('source_ineligible_policy', 'exclude') not in ('error', 'exclude'):
                raise ValueError('Unsupported source_ineligible_policy')
        for kind in ('source','target'):
            if data[kind+'_manifest'] != str(Path(prep['output']) / (kind+'_manifest.json')):
                raise ValueError('Prepared manifest paths must match data.prepare.output')
    keys(s1, ('training','module_a','module_b'), ('training','module_a','module_b'), 'stage1')
    training = s1['training']
    keys(training, ('rounds','local_budget','views_per_graph','log_every','batch_size','sampling_unit'),
         ('rounds','local_budget','views_per_graph','log_every'), 'stage1.training')
    for name in ('rounds','views_per_graph'):
        positive(training[name], name)
    positive(training['log_every'], 'log_every', 0)
    training.setdefault('batch_size', 1)
    training.setdefault('sampling_unit', 'graph')
    positive(training['batch_size'], 'batch_size')
    if training['sampling_unit'] not in ('graph', 'node'):
        raise ValueError('sampling_unit must be graph or node')
    budget = training['local_budget']
    keys(budget, ('unit','amount'), ('unit','amount'), 'local_budget')
    if budget['unit'] not in ('steps','epochs'):
        raise ValueError('local_budget.unit must be steps or epochs')
    positive(budget['amount'], 'local_budget.amount')
    s1['module_a'] = EdgePredictionConfig(**s1['module_a']).to_dict()
    s1['module_b'] = ResponseAggregationConfig(**s1['module_b']).to_dict()
    keys(s2, ('checkpoint','adaptations','target','k_shots','seeds'), ('adaptations','target'), 'stage2')
    modes = s2['adaptations']
    if (not isinstance(modes, list) or not modes or len(set(modes)) != len(modes)
            or any(m not in ('none','lora','adapter','prompt') for m in modes)):
        raise ValueError('stage2.adaptations must contain unique supported modes')
    target = s2['target']
    if 'seed' in target or 'adaptation' in target:
        raise ValueError('Set seed in run and adaptation modes in stage2.adaptations')
    effective = asdict(TargetConfig(**target, seed=run['seed']))
    del effective['seed'], effective['adaptation']
    s2['target'] = effective
    for key, fallback in (('k_shots', [effective['k_shot']]), ('seeds', [run['seed']])):
        values = s2.setdefault(key, fallback)
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f'stage2.{key} must be a nonempty list of unique integers')
        for value in values:
            if type(value) is not int or (key == 'seeds' and value < 0) or (key == 'k_shots' and value != -1 and value < 1):
                raise ValueError(f'Invalid stage2.{key}')
    s2['checkpoint'] = path_from_repo(s2['checkpoint']) if s2.get('checkpoint') else None
    return cfg


def plan(cfg, stage):
    root = Path(cfg['run']['output'])
    s1, s2 = cfg['stage1'], cfg['stage2']
    training, budget = s1['training'], s1['training']['local_budget']
    last = root / 'stage1' / f"round_{training['rounds']-1:03d}" / 'server/shared_backbone.pt'
    if stage != 'stage2' and s2['checkpoint']:
        raise ValueError('stage2.checkpoint is only used with --stage stage2; use null for a complete pipeline')
    if stage == 'stage2' and not s2['checkpoint']:
        raise ValueError('--stage stage2 requires stage2.checkpoint in the config')
    checkpoint = Path(s2['checkpoint']) if stage == 'stage2' else last
    pretrain_args = [
        '--source-manifest', cfg['data']['source_manifest'],
        '--config', str(root/'module_a.json'), '--server-config', str(root/'module_b.json'),
        '--rounds', str(training['rounds']), '--views-per-graph', str(training['views_per_graph']),
        '--seed', str(cfg['run']['seed']), '--threads', str(cfg['run']['threads']),
        '--device', cfg['run']['device'], '--log-every', str(training['log_every']),
        '--batch-size', str(training['batch_size']), '--sampling-unit', training['sampling_unit'],
        '--output', str(root/'stage1')]
    if budget['unit'] == 'steps':
        pretrain_args += ['--local-epochs','1','--local-steps',str(budget['amount']),
                         '--source-order','shuffled_cycle']
    else:
        pretrain_args += ['--local-epochs',str(budget['amount']),'--source-order','shuffled_epoch']
    runs = []
    multiple = len(s2.get('k_shots', [s2['target'].get('k_shot', -1)])) * len(s2.get('seeds', [cfg['run']['seed']])) > 1
    for shot in s2.get('k_shots', [s2['target'].get('k_shot', -1)]):
        for seed in s2.get('seeds', [cfg['run']['seed']]):
            for mode in s2['adaptations']:
                key = f'k{shot}/seed{seed}/{mode}' if multiple else mode
                runs.append({'key':key, 'adaptation':mode, 'k_shot':shot, 'seed':seed,
                             'output':str(root/'stage2'/key)})
    return {'stage':stage, 'output':str(root), 'checkpoint':str(checkpoint),
            'pretrain_args':pretrain_args if stage in ('all','stage1') else None,
            'target_outputs':{r['key']:r['output'] for r in runs}
                             if stage in ('all','stage2') else {},
            'target_runs':runs if stage in ('all','stage2') else [],
            'test_evaluation':s2['target']['evaluate_test'],
            'local_budget_semantics':('exact optimizer updates per client per round; mean independent views per batch'
                                      if budget['unit']=='steps' else
                                      'complete source graph/root passes; batch_size independent views per update')}


def preflight(cfg, stage):
    import torch
    device = torch.device(cfg['run']['device'])
    if device.type not in ('cpu','cuda'):
        raise ValueError('Use a CPU or CUDA device')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('Configured CUDA device is unavailable')
        torch.cuda.get_device_properties(device)
    output = Path(cfg['run']['output'])
    if output.exists():
        raise FileExistsError(f'Use a new run.output: {output}')
    prep = cfg['data']['prepare']
    for kind in (('source','target') if stage == 'all' else ('source',) if stage == 'stage1' else ('target',)):
        path = Path(cfg['data'][kind+'_manifest'])
        if output == path.parent or output in path.parents:
            raise ValueError('Run output cannot contain the prepared input data')
        if not path.is_file() and (prep is None or Path(prep['output']).exists()):
            raise FileNotFoundError(path)
    if prep is not None and not Path(prep['output']).exists() and not Path(prep['cache_root']).is_dir():
        raise FileNotFoundError(prep['cache_root'])
    if stage == 'stage2':
        from .target_model import load_shared
        _, source_cfg, _ = load_shared(cfg['stage2']['checkpoint'])
    # Catch incompatible mechanisms and selection rules before a long stage 1 run.
    from .data import read_manifest
    if stage in ('all','stage1') and Path(cfg['data']['source_manifest']).is_file():
        manifest = read_manifest(cfg['data']['source_manifest'])
        for client in manifest['clients']:
            for graph in client['graphs']:
                if not Path(graph).is_file():
                    raise FileNotFoundError(graph)
        if prep and prep['kind'] == 'gefra_loo_v1':
            if cfg['stage1']['module_a']['num_experts'] != len(manifest['clients']):
                raise ValueError('module_a.num_experts must equal the explicit source-client count')
    module_a, module_b = cfg['stage1']['module_a'], cfg['stage1']['module_b']
    if stage in ('all','stage1') and module_b['mode'] == 'response' and (
            module_a['composer'] == 'linear_gate' or not module_a['relation_condition']):
        raise ValueError('Response aggregation requires a relation-conditioned composer')
    if stage in ('all','stage2'):
        if stage == 'stage2':
            composer = source_cfg.composer
        else:
            composer = module_a['composer']
        if 'lora' in cfg['stage2']['adaptations'] and composer == 'linear_gate':
            raise ValueError('LoRA requires query/key projections; linear_gate has neither')
        target_path = Path(cfg['data']['target_manifest'])
        if target_path.is_file():
            manifest = json.loads(target_path.read_text())
            metric, objective = cfg['stage2']['target']['selection_metric'], manifest['task']['objective']
            if ((metric == 'accuracy' and objective != 'multiclass')
                    or (metric in ('roc_auc','average_precision') and objective == 'multiclass')):
                raise ValueError('Selection metric does not support the configured target objective')
            wanted = ('train','valid','test') if cfg['stage2']['target']['evaluate_test'] else ('train','valid')
            checked_paths = set()
            for split in wanted:
                for entry in manifest['splits'][split]:
                    relative = Path(entry['path'])
                    if relative.is_absolute() or '..' in relative.parts:
                        raise ValueError('Target tensors must stay inside the manifest folder')
                    tensor_path = target_path.parent / relative
                    if tensor_path not in checked_paths and not tensor_path.is_file():
                        raise FileNotFoundError(tensor_path)
                    checked_paths.add(tensor_path)


def validate_prepared_roles(cfg):
    """Check Gefra source/target roles after preparation and before training."""
    prep = cfg['data'].get('prepare')
    if not prep or prep['kind'] != 'gefra_loo_v1':
        return
    import torch
    from .data import read_manifest
    from .gefra_protocol import sources_for
    source = read_manifest(cfg['data']['source_manifest'])
    target = json.loads(Path(cfg['data']['target_manifest']).read_text())
    level, expected_sources = sources_for(prep['target'])
    if (source.get('target_dataset') != prep['target'] or target.get('dataset') != prep['target']
            or source.get('level') != level or target['task']['level'] != level
            or source['protocol_id'] != target['protocol_id']):
        raise ValueError('Prepared source/target task roles differ from configuration')
    if [client['id'] for client in source['clients']] != expected_sources:
        raise ValueError('Prepared source client IDs/order differ from the Gefra target configuration')
    expected_unit = 'node' if source.get('level') == 'node' else 'graph'
    if cfg['stage1']['training']['sampling_unit'] != expected_unit:
        raise ValueError('sampling_unit must match the Gefra graph/node target level')
    if len(source['clients']) != cfg['stage1']['module_a']['num_experts']:
        raise ValueError('Source client count and num_experts differ')
    for client in source['clients']:
        if level == 'node' and (len(client['graphs']) != 1 or client.get('graph_counts') != [1]):
            raise ValueError('Gefra node clients must each contain one full source graph')
        for graph in client['graphs']:
            payload = torch.load(graph, map_location='cpu', weights_only=True, mmap=True)
            if isinstance(payload, dict) and 'y' in payload:
                raise ValueError('Source tensor unexpectedly contains labels')
            if isinstance(payload, list) and any('y' in item for item in payload):
                raise ValueError('Source tensor unexpectedly contains labels')


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, value):
        self.stream.write(value)
        self.log.write(value)

    def flush(self):
        self.stream.flush()
        self.log.flush()


def execute(cfg, stage='all', config_path=None):
    import torch
    execution = plan(cfg, stage)
    preflight(cfg, stage)
    root = Path(execution['output'])
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/'resolved_config.json', cfg)
    write_json(root/'execution_plan.json', execution)
    write_json(root/'module_a.json', cfg['stage1']['module_a'])
    write_json(root/'module_b.json', cfg['stage1']['module_b'])
    if config_path:
        shutil.copyfile(config_path, root/'input_config.json')
    paths = [*sorted((REPO/'fedg2c').rglob('*.py')), *sorted((REPO/'model').rglob('*.py')),
             REPO/'run_fedg2c_pipeline.py', REPO/'pretrain_federated_fedg2c.py', REPO/'finetune_fedg2c.py']
    hashes = {}
    for path in paths:
        name = path.relative_to(REPO)
        destination = root/'code_snapshot'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        hashes[name.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(root/'code_sha256.json', hashes)
    write_json(root/'environment.json', {'python':platform.python_version(),'torch':str(torch.__version__),
               'device':cfg['run']['device'], 'gpu':torch.cuda.get_device_name(cfg['run']['device'])
               if cfg['run']['device'].startswith('cuda') else None,
               'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
               'precision':'float32; AMP disabled', 'pid':os.getpid()})
    status = {'stage':stage, 'started_at':datetime.now(timezone.utc).isoformat(), 'state':'running'}
    write_json(root/'status.json', status)
    with (root/'console.log').open('w', encoding='utf-8', buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                torch.set_num_threads(cfg['run']['threads'])
                prep = cfg['data']['prepare']
                if prep is not None and not Path(prep['output']).exists():
                    if prep['kind'] == 'molecular_m1':
                        from .prepare_molecular_data import prepare_molecular_data
                        prepare_molecular_data(prep['cache_root'], prep['output'], prep['seed'])
                    else:
                        from .prepare_gefra_loo_data import prepare_gefra_loo_data
                        prepare_gefra_loo_data(prep['cache_root'], prep['output'], prep['seed'],
                                              prep['target'], prep.get('feature_policy', 'native'),
                                              prep.get('source_ineligible_policy', 'exclude'))
                manifests = {}
                for kind in (('source','target') if stage=='all' else ('source',) if stage=='stage1' else ('target',)):
                    path = Path(cfg['data'][kind+'_manifest'])
                    manifests[kind] = {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                write_json(root/'data_manifests.json', manifests)
                if prep is not None:
                    report = Path(prep['output'])/'preparation_report.json'
                    record = json.loads(report.read_text())
                    if record['seed'] != prep['seed']:
                        raise ValueError('Existing prepared data seed does not match data.prepare.seed')
                    for kind in manifests:
                        if json.loads(Path(manifests[kind]['path']).read_text())['protocol_id'] != record['protocol_id']:
                            raise ValueError('Preparation report and manifest protocol IDs differ')
                    if prep['kind'] == 'gefra_loo_v1':
                        for key, default in (('target', None), ('feature_policy', 'native'),
                                             ('source_ineligible_policy', 'exclude')):
                            report_key = 'target_dataset' if key == 'target' else key
                            if record.get(report_key) != prep.get(key, default):
                                raise ValueError(f'Prepared data {key} differs from configuration')
                    shutil.copyfile(report, root/'preparation_report.json')
                # Recheck the actual manifests after preparation; fail before
                # source training if data roles or target settings are invalid.
                validate_prepared_roles(cfg)
                if execution['pretrain_args'] is not None:
                    from pretrain_federated_fedg2c import main as pretrain
                    pretrain(execution['pretrain_args'])
                reports = {}
                if execution['target_outputs']:
                    from .target_train import run_target
                    for run in execution['target_runs']:
                        target_raw = dict(cfg['stage2']['target'], adaptation=run['adaptation'],
                                          seed=run['seed'], k_shot=run['k_shot'])
                        target = TargetConfig(**target_raw)
                        reports[run['key']] = run_target(execution['checkpoint'], cfg['data']['target_manifest'],
                                                        target, run['output'], cfg['run']['device'])
                write_json(root/'summary.json', {mode:{k:r[k] for k in
                           ('best_epoch','epochs_ran','validation','test','test_evaluated','frozen_base_unchanged')}
                           for mode,r in reports.items()})
                status.update(state='completed', exit_code=0)
            except BaseException as error:
                status.update(state='failed', exit_code=1, error=f'{type(error).__name__}: {error}')
                traceback.print_exc()
                raise
            finally:
                status['finished_at'] = datetime.now(timezone.utc).isoformat()
                write_json(root/'status.json', status)
                (root/'exit_code').write_text(str(status['exit_code'])+'\n')
    return execution


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=('all','stage1','stage2'), default='all')
    parser.add_argument('--dry-run', action='store_true', help='Validate config and print plan without training or writes')
    args = parser.parse_args(argv)
    cfg = resolve_config(args.config)
    execution = plan(cfg, args.stage)
    if args.dry_run:
        print(json.dumps({'resolved_config':cfg, 'execution_plan':execution}, indent=2))
        return
    execute(cfg, args.stage, args.config)

# ===== FedG2C ADD END A/B/C =====
