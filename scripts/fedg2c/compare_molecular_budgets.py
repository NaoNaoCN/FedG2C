# ===== FedG2C ADD BEGIN A/B/C: report completed validation-only comparisons =====
"""Compare validation-selected models; never use test metrics to rank budgets."""
import argparse
import json
from pathlib import Path
import time


def compare(repo, jobs):
    candidates = [('historical_round10_server10_epochs20',repo/'runs/molecular_formal_gpu_20261023')]
    candidates += [(j['name'],Path(j['config']['run']['output'])) for j in jobs]
    rows = []
    for name, root in candidates:
        for mode in ('none','lora','adapter','prompt'):
            path = root/'stage2'/mode/'metrics.json'
            if name.startswith('historical_'):
                path = root/f'target_{mode}'/'metrics.json'
            result = json.loads(path.read_text())
            selected = result['history'][result['best_epoch']-1]['valid']
            rows.append({'experiment':name,'adaptation':mode,'best_epoch':result['best_epoch'],
                         'epochs_ran':len(result['history']),'validation':selected,
                         'frozen_base_unchanged':result['frozen_base_unchanged']})
            if not name.startswith('historical_') and result.get('test_evaluated') is not False:
                raise ValueError('A tuning job unexpectedly evaluated the test set')
    return {'selection':'minimum validation BCE; first tie', 'rows':rows,
            'scope':'single training seed 2026 and fixed data; no statistical superiority claim',
            'caution':'Historical test metrics were not read into this comparison. Other hyperparameters are not searched.'}


def write_comparison(repo, status, output):
    result = compare(repo,status['jobs'])
    output = Path(output)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines = ['# Validation budget comparison','',
             'Selected by minimum validation BCE (first tie); tuning jobs did not evaluate test data.',
             '', '| Experiment | Adaptation | Epochs run | Best epoch | Validation BCE | Validation ROC-AUC | Validation AP |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in result['rows']:
        valid = row['validation']
        lines.append('| '+ ' | '.join(str(v) for v in (row['experiment'],row['adaptation'],row['epochs_ran'],
                     row['best_epoch'],f"{valid['loss']:.6f}",f"{valid['roc_auc']:.6f}",
                     f"{valid['average_precision']:.6f}"))+' |')
    lines += ['',result['scope'], '']
    output.with_suffix('.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite',type=Path)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--wait',action='store_true',help='Wait for the detached experiment queue to complete')
    args = parser.parse_args()
    while True:
        status = json.loads((args.suite/'status.json').read_text())
        if status['state'] == 'completed':
            break
        if status['state'] == 'failed':
            raise RuntimeError('Budget queue failed; inspect its status and job logs')
        if not args.wait:
            raise RuntimeError('Budget queue is still running; pass --wait or retry after completion')
        time.sleep(30)
    write_comparison(args.repo,status,args.suite/'comparison.json')
# ===== FedG2C ADD END A/B/C =====
