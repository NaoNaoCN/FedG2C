"""Three node ablations: frozen code, resource guards, recovery and independent tests."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
GIB=1024**3


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def use_code(root):
    sys.path.insert(0,str(root/'code_snapshot'))
    for name,expected in read(root/'code_sha256.json').items():
        if digest(root/'code_snapshot'/name)!=expected:
            raise ValueError('Frozen code changed: '+name)


def cfg_for(root,job):
    return read(root/'configs'/(job['variant']+'_'+job['target']+'.json'))


def checkpoint_for(cfg):
    return Path(cfg['run']['output'])/'stage1'/f"round_{cfg['stage1']['training']['rounds']-1:03d}"/'server/shared_backbone.pt'


def result_id(job,shot,seed):
    return f"{job['variant']}_{job['target']}_k{shot}_seed{seed}"


def prepare(args):
    root=args.root.resolve()
    if root.exists():
        raise FileExistsError('Use new suite directory: '+str(root))
    sys.path.insert(0,str(REPO))
    from fedg2c.node_ablation_matrix import build_plan,TARGETS,VARIANTS
    from fedg2c.pipeline import preflight,validate_prepared_roles
    base=read(args.config)
    base['run'].update(output=str(root),device='cuda:0',threads=1)
    base['matrix']['data_root']=str(args.data_root.resolve())
    configs,report=build_plan(base)
    for cfg in configs.values():
        preflight(cfg,'all')
        validate_prepared_roles(cfg)
    root.mkdir(parents=True)
    files=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard'],cwd=REPO,text=True).splitlines()
    hashes={}
    for name in sorted(set(files)):
        source=REPO/name
        if not source.is_file() or source.suffix not in ('.py','.json','.md','.txt','.yaml','.toml','.sh'):
            continue
        dest=root/'code_snapshot'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,dest)
        hashes[name]=digest(dest)
    write(root/'code_sha256.json',hashes)
    for name,cfg in configs.items():
        write(root/'configs'/(name+'.json'),cfg)
        folder=Path(cfg['run']['output'])
        write(folder/'resolved_config.json',cfg)
        write(folder/'module_a.json',cfg['stage1']['module_a'])
        write(folder/'module_b.json',cfg['stage1']['module_b'])
    jobs=[dict(id=f'{stage}_{variant}_{target}',stage=stage,variant=variant,target=target)
          for stage in ('pretrain','train','test') for target in TARGETS for variant in VARIANTS]
    report.update(created_at=now(),formal_training_started=False,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        policy=dict(max_gpu_workers=2,max_target_workers=1,min_available_ram_gib=6,
        min_free_pretrain_gpu_mib=10000,min_free_target_gpu_mib=26000,
        gpu_stop_free_mib=1800,memory_stop_fraction=.86,retries=2),
        schedule='18 pretraining groups, then 540 LoRA selections, then 540 independent final tests',
        downstream_initialization='historical RNG order; no reseeding after shared checkpoint loading')
    write(root/'jobs.json',jobs)
    write(root/'plan.json',report)
    write(root/'input_matrix.json',base)
    for name in ('logs','jobs','results','progress','preflight','summary'):
        (root/name).mkdir(exist_ok=True)
    write(root/'status.json',dict(state='prepared',updated_at=now(),completed_pretraining=0,completed_finetuning=0,completed_tests=0))
    print(json.dumps(dict(prepared=True,root=str(root),pretraining=18,finetuning=540,tests=540)),flush=True)


def completed(root):
    return {p.name.removesuffix('.done.json') for p in (root/'jobs').glob('*.done.json')}


def pick_job(jobs,done,busy):
    for stage in ('pretrain','train','test'):
        phase=[j for j in jobs if j['stage']==stage]
        if all(j['id'] in done for j in phase):
            continue
        return next((j for j in phase if j['id'] not in done and j['id'] not in busy),None)
    return None


def freeze_selections(root):
    rows={p.name.removeprefix('train_').removesuffix('.json'):read(p) for p in (root/'results').glob('train_*.json')}
    if len(rows)!=read(root/'plan.json')['downstream_runs']:
        raise ValueError('Not all downstream selections are ready')
    supports={}
    for row in rows.values():
        if digest(row['path'])!=row['sha256']:
            raise ValueError('Selected model hash changed')
        key=row['target'],row['shot']
        supports.setdefault(key,row['support_sha256'])
        if supports[key]!=row['support_sha256']:
            raise ValueError('Fixed support differs across variants or seeds')
    write(root/'validation_selection_manifest.json',dict(frozen_at=now(),selections=rows))


def worker(root,job_id):
    use_code(root)
    import torch
    torch.set_num_threads(1)
    job=next(j for j in read(root/'jobs.json') if j['id']==job_id)
    cfg=cfg_for(root,job)
    device=cfg['run']['device']
    progress=root/'progress'/(job_id+'.json')
    write(progress,dict(job=job,pid=os.getpid(),started_at=now()))
    if job['stage']=='pretrain':
        from fedg2c.pipeline import plan
        from pretrain_federated_fedg2c import main
        from fedg2c.target_model import load_shared
        output=Path(cfg['run']['output'])/'stage1'
        if output.exists() and not (output/'recovery.pt').is_file():
            archive=root/'interrupted_attempts'/job_id/str(time.time_ns())
            archive.parent.mkdir(parents=True,exist_ok=True)
            output.rename(archive)
        main(plan(cfg,'stage1')['pretrain_args']+['--recover'])
        _,_,meta=load_shared(checkpoint_for(cfg))
        if meta['round_id']!=cfg['stage1']['training']['rounds']-1:
            raise ValueError('Pretraining rounds incomplete')
        write(root/'jobs'/(job_id+'.done.json'),dict(**job,finished_at=now(),checkpoint=str(checkpoint_for(cfg)),state_sha256=meta['state_sha256']))
        return
    from fedg2c.target_data import TargetDataCache,load_target_manifest
    from fedg2c.target_model import TargetConfig
    from fedg2c.target_train import run_target,load_target_model,evaluate
    cache=TargetDataCache(cfg['stage2']['target']['crop_cache_mb'])
    if job['stage']=='test':
        selections=read(root/'validation_selection_manifest.json')['selections']
        if len(selections)!=read(root/'plan.json')['downstream_runs'] or not all(j['id'] in completed(root) for j in read(root/'jobs.json') if j['stage']=='train'):
            raise ValueError('Final test requires every frozen validation selection')
    for shot in cfg['stage2']['k_shots']:
      for seed in cfg['stage2']['seeds']:
        identity=result_id(job,shot,seed)
        result=root/'results'/(job['stage']+'_'+identity+'.json')
        if result.exists():
            continue
        write(progress,dict(job=job,pid=os.getpid(),shot=shot,seed=seed,updated_at=now()))
        output=Path(cfg['run']['output'])/'stage2'/f'k{shot}'/f'seed{seed}'/'lora'
        target=TargetConfig(**dict(cfg['stage2']['target'],adaptation='lora',seed=seed,k_shot=shot,evaluate_test=False))
        if job['stage']=='train':
            if not (output/'metrics.json').is_file() or not (output/'target_model.pt').is_file():
                if output.exists() and not (output/'recovery.pt').is_file():
                    archive=root/'interrupted_attempts'/identity/str(time.time_ns())
                    archive.parent.mkdir(parents=True,exist_ok=True)
                    output.rename(archive)
                run_target(checkpoint_for(cfg),cfg['data']['target_manifest'],target,output,device,data_cache=cache,recover=True)
            report=read(output/'metrics.json')
            if report['test_evaluated'] or not report['frozen_base_unchanged']:
                raise ValueError('Training protocol failed')
            selected=dict(path=str(output/'target_model.pt'),sha256=digest(output/'target_model.pt'),best_epoch=report['best_epoch'],validation_accuracy=report['validation']['accuracy'])
            saved=output/'selected_checkpoint.json'
            if saved.exists() and read(saved)['sha256']!=selected['sha256']:
                raise ValueError('Validation-selected checkpoint changed')
            write(saved,selected)
            write(result,dict(variant=job['variant'],target=job['target'],shot=shot,seed=seed,**selected,support_sha256=report['support_sha256'],epochs_ran=report['epochs_ran'],finished_at=now()))
        else:
            selected=selections[identity]
            checkpoint=output/'target_model.pt'
            if digest(checkpoint)!=selected['sha256']:
                raise ValueError('Frozen checkpoint changed')
            model,_=load_target_model(checkpoint,device)
            _,splits=load_target_manifest(cfg['data']['target_manifest'],target,include_test=True,data_cache=cache)
            tested=evaluate(model,splits['test'],device)
            if digest(checkpoint)!=selected['sha256']:
                raise ValueError('Checkpoint changed during testing')
            write(output/'final_test.json',dict(tested_at=now(),checkpoint_sha256=selected['sha256'],validation_selected=True,test=tested))
            write(result,dict(variant=job['variant'],target=job['target'],shot=shot,seed=seed,test_accuracy=tested['accuracy'],checkpoint_sha256=selected['sha256'],support_sha256=selected['support_sha256'],finished_at=now()))
            del model,splits
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps(dict(event='result_complete',stage=job['stage'],result=identity,at=now())),flush=True)
    write(root/'jobs'/(job_id+'.done.json'),dict(**job,finished_at=now()))


def resources():
    cg=Path('/sys/fs/cgroup')
    current=int((cg/'memory.current').read_text())
    stat=dict(row.split() for row in (cg/'memory.stat').read_text().splitlines())
    mem=dict(row.split(':',1) for row in Path('/proc/meminfo').read_text().splitlines())
    available=int(mem['MemAvailable'].split()[0])*1024
    maximum=(cg/'memory.max').read_text().strip()
    limit=int(maximum) if maximum!='max' else current+available
    working=max(0,current-int(stat.get('inactive_file',0)))
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15)
    free,used,util=[int(v.strip()) for v in gpu.splitlines()[0].split(',')]
    return dict(at=now(),memory_working=working,memory_limit=limit,available_ram=min(available,limit-working),gpu_free_mib=free,gpu_used_mib=used,gpu_utilization=util)


def summary(root,active=()):
    done=completed(root)
    trains=[read(p) for p in (root/'results').glob('train_*.json')]
    tests=[read(p) for p in (root/'results').glob('test_*.json')]
    counts=dict(completed_pretraining=sum(i.startswith('pretrain_') for i in done),completed_finetuning=len(trains),completed_tests=len(tests))
    groups={}
    for row in tests:
        key=row['variant']+'/'+row['target']+'/k'+str(row['shot'])
        groups.setdefault(key,[]).append(row)
    stats={k:dict(n=len(v),mean=statistics.mean(r['test_accuracy'] for r in v),sd=statistics.stdev(r['test_accuracy'] for r in v) if len(v)>1 else None) for k,v in groups.items()}
    write(root/'summary/report.json',dict(updated_at=now(),**counts,groups=stats))
    lines=['# 三组节点分类消融进度','',f'更新时间（UTC）：{now()}','',f'预训练 {counts["completed_pretraining"]}/18；LoRA 微调 {len(trains)}/540；独立测试 {len(tests)}/540。','', '全部验证选模与 checkpoint 哈希固定后才开启独立测试。']
    if active:
        lines+=['','当前任务：'+', '.join(e['job']['id'] for e in active)]
    if stats:
        lines+=['','| 设置／目标／支持集 | 测试准确率 % | seeds |','| --- | --- | --- |']
        for k,v in sorted(stats.items()):
            sd='—' if v['sd'] is None else f'{100*v["sd"]:.2f}'
            lines.append(f'| {k} | {100*v["mean"]:.2f} ± {sd} | {v["n"]}/10 |')
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return counts


def run(args):
    import fcntl
    root=args.root.resolve()
    use_code(root)
    if (root/'PAUSED.json').exists():
        raise RuntimeError('Suite was explicitly paused')
    if not read(root/'preflight/deployment.json')['passed']:
        raise ValueError('Deployment preflight not passed')
    lock=(root/'suite.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    policy=read(root/'plan.json')['policy']
    jobs=read(root/'jobs.json')
    active,retries=[],{}
    limit=policy['max_gpu_workers']
    started=now()
    def persist(state,error=None):
        counts=summary(root,active)
        value=dict(state=state,supervisor_pid=os.getpid(),started_at=started,updated_at=now(),**counts,
            active=[dict(job=e['job'],pid=e['process'].pid,started_at=e['started_at']) for e in active],resources=resources())
        if error:
            value['error']=error
        write(root/'status.json',value)
    def stop(sig,frame):
        for e in active:
            if e['process'].poll() is None:
                os.killpg(e['process'].pid,signal.SIGTERM)
        persist('stopped')
        raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    try:
      while True:
        for entry in list(active):
            code=entry['process'].poll()
            if code is None:
                continue
            active.remove(entry)
            jid=entry['job']['id']
            if code==0 and jid in completed(root):
                continue
            tail=(root/'logs'/(jid+'.log')).read_text(errors='replace')[-12000:]
            attempt=retries.get(jid,0)+1
            retries[jid]=attempt
            transient=entry.get('guarded',False) or code in (-9,-15,137,143) or 'out of memory' in tail.lower()
            write(root/'jobs'/f'{jid}.failure{attempt}.json',dict(at=now(),exit_code=code,transient=transient,tail=tail))
            if not transient or attempt>policy['retries']:
                raise RuntimeError('Worker failed: '+jid)
            limit=1
        snap=resources()
        high=snap['memory_working']>policy['memory_stop_fraction']*snap['memory_limit'] or snap['gpu_free_mib']<policy['gpu_stop_free_mib']
        if high and active:
            entry=active[-1]
            if not entry.get('guarded'):
                entry.update(guarded=True,guard_deadline=time.monotonic()+30)
                os.killpg(entry['process'].pid,signal.SIGTERM)
        for entry in active:
            if entry.get('guarded') and time.monotonic()>entry['guard_deadline'] and entry['process'].poll() is None:
                os.killpg(entry['process'].pid,signal.SIGKILL)
        done=completed(root)
        if len(done)==len(jobs) and not active:
            counts=summary(root)
            plan=read(root/'plan.json')
            if counts!=dict(completed_pretraining=plan['pretraining_runs'],completed_finetuning=plan['downstream_runs'],completed_tests=plan['downstream_runs']):
                raise ValueError('Final completeness audit failed')
            freeze_selections(root)
            selections=read(root/'validation_selection_manifest.json')['selections']
            for p in (root/'results').glob('test_*.json'):
                row=read(p)
                selected=selections[p.name.removeprefix('test_').removesuffix('.json')]
                if row['checkpoint_sha256']!=selected['sha256']:
                    raise ValueError('Final checkpoint audit failed')
            write(root/'summary/audit.json',dict(passed=True,finished_at=now(),**counts))
            persist('completed')
            return
        job=pick_job(jobs,done,{e['job']['id'] for e in active})
        targets_busy=sum(e['job']['stage']!='pretrain' for e in active)
        floor=policy['min_free_pretrain_gpu_mib'] if job and job['stage']=='pretrain' else policy['min_free_target_gpu_mib']
        capacity=job and (job['stage']=='pretrain' or targets_busy<policy['max_target_workers'])
        if capacity and len(active)<limit and not high and not (root/'PAUSED.json').exists() and snap['available_ram']>=policy['min_available_ram_gib']*GIB and snap['gpu_free_mib']>=floor:
            if job['stage']=='test' and not (root/'validation_selection_manifest.json').exists():
                freeze_selections(root)
            command=[sys.executable,'-u',str(root/'code_snapshot/scripts/fedg2c/run_node_ablation_suite.py'),'worker','--root',str(root),'--job',job['id']]
            with (root/'logs'/(job['id']+'.log')).open('a',buffering=1) as log:
                process=subprocess.Popen(command,cwd=root/'code_snapshot',stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                    start_new_session=True,pass_fds=(lock.fileno(),),env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1'))
            active.append(dict(job=job,process=process,started_at=now()))
            plan=read(root/'plan.json')
            if not plan.get('formal_training_started'):
                plan.update(formal_training_started=True,started_at=now())
                write(root/'plan.json',plan)
            write(root/'jobs'/(job['id']+'.command.json'),dict(command=command,pid=process.pid,started_at=now()))
            print(json.dumps(dict(event='worker_started',job=job['id'],pid=process.pid,at=now())),flush=True)
        persist('running' if active else 'waiting_for_resources')
        with (root/'logs/resources.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(active=[e['job']['id'] for e in active],**snap))+'\n')
        time.sleep(10)
    except Exception as error:
        for e in active:
            if e['process'].poll() is None:
                os.killpg(e['process'].pid,signal.SIGTERM)
        persist('failed',repr(error))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','worker','summary'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--job')
    args=parser.parse_args()
    if args.action=='prepare':
        prepare(args)
    elif args.action=='worker':
        worker(args.root.resolve(),args.job)
    elif args.action=='run':
        run(args)
    else:
        print(json.dumps(summary(args.root.resolve()),indent=2))


if __name__=='__main__':
    main()
