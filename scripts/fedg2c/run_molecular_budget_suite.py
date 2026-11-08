# ===== FedG2C ADD BEGIN A/B/C: validation-only budget experiment queue =====
"""Run the confirmed BCE/unchanged-objective budget comparisons sequentially."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from fedg2c.pipeline import plan, preflight, resolve_config

JOBS = [('target_budget','config/fedg2c_molecular_target_budget.json','stage2'),
        ('round20_server10','config/fedg2c_molecular_round20_server10.json','all'),
        ('round20_server20','config/fedg2c_molecular_round20_server20.json','all')]


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    jobs = []
    for name, path, stage in JOBS:
        cfg = resolve_config(REPO/path)
        if cfg['stage2']['target']['evaluate_test'] or cfg['stage2']['target']['selection_metric'] != 'loss':
            raise ValueError('This queue requires validation BCE selection and disabled test evaluation')
        jobs.append({'name':name,'config':cfg,'stage':stage,'plan':plan(cfg,stage),'state':'pending'})
    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        return
    for job in jobs:
        preflight(job['config'], job['stage'])
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    status = {'state':'running','pid':os.getpid(),'started_at':now(),'jobs':jobs}

    def save():
        path = root/'status.json'
        temporary = root/'status.pending.json'
        temporary.write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
        temporary.replace(path)

    save()
    try:
        for job in jobs:
            cfg = root/(job['name']+'.json')
            cfg.write_text(json.dumps(job['config'],indent=2)+'\n',encoding='utf-8')
            job.update(state='running',started_at=now())
            save()
            command = [sys.executable,'-u',str(REPO/'run_fedg2c_pipeline.py'),
                       '--config',str(cfg),'--stage',job['stage']]
            print(json.dumps({'job':job['name'],'command':command}),flush=True)
            with (root/(job['name']+'.log')).open('w',encoding='utf-8') as log:
                child = subprocess.Popen(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
                job['pid'] = child.pid
                save()
                code = child.wait()
            job.update(state='completed' if code==0 else 'failed',exit_code=code,finished_at=now())
            save()
            if code:
                raise RuntimeError(f"{job['name']} failed with exit {code}; remaining jobs were not launched")
        status['state'] = 'completed'
    except BaseException as error:
        status.update(state='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        status['finished_at'] = now()
        save()


if __name__ == '__main__':
    main()
# ===== FedG2C ADD END A/B/C =====
