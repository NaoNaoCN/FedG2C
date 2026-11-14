"""Deployment scheduling, final-test barrier and three-variant worker integration."""
import importlib.util
import json
from pathlib import Path

import pytest

REPO=Path(__file__).resolve().parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


suite=module('ablation_deploy_suite',REPO/'scripts/fedg2c/run_node_ablation_suite.py')
helpers=module('ablation_test_helpers',REPO/'tests/test_node_ablation.py')


def test_scheduler_waits_for_entire_previous_phase_and_skips_busy_jobs():
    jobs=[dict(id=stage+'_'+name,stage=stage) for stage in ('pretrain','train','test') for name in ('a','b')]
    assert suite.pick_job(jobs,set(),set())['id']=='pretrain_a'
    assert suite.pick_job(jobs,set(),{'pretrain_a'})['id']=='pretrain_b'
    assert suite.pick_job(jobs,{'pretrain_a'},{'pretrain_b'}) is None
    assert suite.pick_job(jobs,{'pretrain_a','pretrain_b'},set())['id']=='train_a'
    assert suite.pick_job(jobs,{'pretrain_a','pretrain_b','train_a'},{'train_b'}) is None
    assert suite.pick_job(jobs,{j['id'] for j in jobs},set()) is None


def test_freeze_checks_support_consistency_and_selected_file_hashes(tmp_path):
    suite.write(tmp_path/'plan.json',dict(downstream_runs=2))
    first=tmp_path/'first.pt'
    second=tmp_path/'second.pt'
    first.write_bytes(b'first selected model')
    second.write_bytes(b'second selected model')
    rows=[]
    for variant,path in [('baseline',first),('ablate_b',second)]:
        row=dict(variant=variant,target='cora',shot=5,seed=0,path=str(path),sha256=suite.digest(path),support_sha256='fixed-support')
        suite.write(tmp_path/'results'/('train_'+variant+'.json'),row)
        rows.append(row)
    suite.freeze_selections(tmp_path)
    assert len(suite.read(tmp_path/'validation_selection_manifest.json')['selections'])==2
    rows[1]['support_sha256']='different-support'
    suite.write(tmp_path/'results/train_ablate_b.json',rows[1])
    with pytest.raises(ValueError,match='support differs'):
        suite.freeze_selections(tmp_path)
    rows[1]['support_sha256']='fixed-support'
    suite.write(tmp_path/'results/train_ablate_b.json',rows[1])
    first.write_bytes(b'changed model')
    with pytest.raises(ValueError,match='hash changed'):
        suite.freeze_selections(tmp_path)


def test_three_variant_workers_train_select_then_test_with_frozen_manifest(tmp_path):
    root=tmp_path/'suite'
    root.mkdir()
    (root/'code_snapshot').symlink_to(REPO,target_is_directory=True)
    suite.write(root/'code_sha256.json',{})
    suite.write(root/'plan.json',dict(pretraining_runs=3,downstream_runs=3))
    for name in ('jobs','results','progress','logs','summary'):
        (root/name).mkdir()
    variants=('baseline','ablate_b','ablate_ab')
    shared_data = None
    for variant in variants:
        folder=tmp_path/variant
        folder.mkdir()
        cfg=helpers.synthetic_config(folder,variant)
        if shared_data is None:
            shared_data = cfg['data']
        cfg['data'] = dict(shared_data)
        cfg['stage2']['seeds']=[0]
        cfg['stage2']['k_shots']=[1]
        suite.write(root/'configs'/(variant+'_cora.json'),cfg)
        suite.write(Path(cfg['run']['output'])/'module_a.json',cfg['stage1']['module_a'])
        suite.write(Path(cfg['run']['output'])/'module_b.json',cfg['stage1']['module_b'])
    jobs=[dict(id=stage+'_'+v+'_cora',stage=stage,variant=v,target='cora')
          for stage in ('pretrain','train','test') for v in variants]
    suite.write(root/'jobs.json',jobs)
    with pytest.raises(FileNotFoundError):
        suite.worker(root,'test_baseline_cora')
    for job in jobs:
        if job['stage']=='test':
            continue
        suite.worker(root,job['id'])
    assert not list(tmp_path.rglob('final_test.json'))
    assert suite.summary(root)==dict(completed_pretraining=3,completed_finetuning=3,completed_tests=0)
    suite.freeze_selections(root)
    before=(root/'validation_selection_manifest.json').read_bytes()
    for variant in variants:
        suite.worker(root,'test_'+variant+'_cora')
        suite.worker(root,'test_'+variant+'_cora')
    assert suite.summary(root)==dict(completed_pretraining=3,completed_finetuning=3,completed_tests=3)
    assert (root/'validation_selection_manifest.json').read_bytes()==before
    for row in (root/'results').glob('train_*.json'):
        selected=suite.read(row)
        assert suite.digest(selected['path'])==selected['sha256']
