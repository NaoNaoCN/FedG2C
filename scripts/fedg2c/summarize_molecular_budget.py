# ===== FedG2C ADD BEGIN A/B/C: describe saved budgets without test-driven tuning =====
"""Summarize source learning curves, conditional B replay and target validation."""
import argparse
import json
from pathlib import Path
import statistics
import torch


def summarize(run, replay):
    source = {}
    for folder in sorted((run/'federated').glob('round_*')):
        for path in sorted(folder.glob('client_*/metrics.json')):
            record = json.loads(path.read_text())
            meta = torch.load(path.parent/'shared_backbone.pt', map_location='cpu', weights_only=True)
            client = meta['client_id']
            steps = record['steps']
            source.setdefault(client, []).append({
                'round':int(folder.name.split('_')[1])+1, 'steps':len(steps),
                'mean_loss':statistics.mean(s['loss'] for s in steps),
                'context_std':statistics.mean(s['context_std'] for s in steps),
                'reference_std':record['reference_std'], 'sampling':{
                    k:v for k,v in record.get('sampling',{}).items() if k!='graph_indices'}})
    raw = json.loads(replay.read_text())
    server = []
    for budget in raw['budgets']:
        rows = [r for r in raw['rows'] if r['steps']==budget]
        server.append({'steps':budget,'rounds':len(rows),
                       'objective':statistics.mean(r['objective'] for r in rows),
                       'diagnostic_output_mse':statistics.mean(r['diagnostic']['output_mse'] for r in rows),
                       'diagnostic_response_mse':statistics.mean(r['diagnostic']['response_mse'] for r in rows),
                       'mean_gradient_norm_ratio_response_to_output':statistics.mean(
                           r['response_gradient_norm']/max(r['output_gradient_norm'],1e-30) for r in rows)})
    targets = {}
    for mode in ('none','lora','adapter','prompt'):
        history = json.loads((run/f'target_{mode}/history.json').read_text())
        choices = {}
        for metric in ('loss','roc_auc','average_precision'):
            rows = [r for r in history if r['valid'].get(metric) is not None]
            chosen = (min if metric=='loss' else max)(rows, key=lambda r:r['valid'][metric])
            choices[metric] = {'epoch':chosen['epoch'], 'validation':chosen['valid'][metric]}
        targets[mode] = {'epochs':len(history), 'validation_best':choices,
                         'history':[{'epoch':r['epoch'],'train_loss':r['train_loss'],'valid':r['valid']} for r in history]}
    return {'source':source,'server':server,'target':targets,
            'limits':['Source losses use changing graphs/views/EMA targets and do not establish saturation.',
                      'B replays are conditional on saved uploads; diagnostic probes do not select weights.',
                      'Target comparisons use validation history only; test scores are not read.']}


def plot(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(15,4.3))
    for name, rows in report['source'].items():
        axes[0].plot([r['round'] for r in rows], [r['mean_loss'] for r in rows], marker='.', label=name)
    axes[0].set(xlabel='Federated round', ylabel='Mean local training loss', title='A: changing training views')
    axes[0].legend(fontsize=8)
    rows = report['server']
    axes[1].plot([r['steps'] for r in rows], [r['objective'] for r in rows], marker='.', label='Calibration objective')
    axes[1].plot([r['steps'] for r in rows], [r['diagnostic_output_mse'] for r in rows], marker='.', label='Diagnostic output MSE')
    axes[1].set(xlabel='Server steps', ylabel='Mean over 10 saved rounds', title='B: independent conditional replays')
    axes[1].legend(fontsize=8)
    for mode, item in report['target'].items():
        axes[2].plot([r['epoch'] for r in item['history']], [r['valid']['loss'] for r in item['history']], label=mode)
    axes[2].set(xlabel='Fine-tuning epoch', ylabel='Validation BCE', title='Stage 2: validation curves')
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output,dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    result = summarize(args.run, args.replay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    if args.plot:
        plot(result,args.output.with_suffix('.png'))
# ===== FedG2C ADD END A/B/C =====
