"""Replay saved uploads to diagnose server budgets; never update the FL run."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from fedg2c.model import SharedBackbone
from fedg2c.edge_config import EdgePredictionConfig
from fedg2c.server_config import ResponseAggregationConfig
from fedg2c.response_server import ResponsePreservingServer, distribution, mix_parameters
from fedg2c.server_probes import RelationProbe


def audit(root, output, budgets):
    torch.set_num_threads(1)
    rows = []
    skipped_rounds = []
    for folder in sorted((root / 'federated').glob('round_*')):
        if not (folder / 'server/shared_backbone.pt').is_file():
            skipped_rounds.append(folder.name)
            continue
        payloads = [torch.load(p / 'shared_backbone.pt', map_location='cpu', weights_only=True)
                    for p in sorted(folder.glob('client_*'))]
        saved = torch.load(folder / 'server/shared_backbone.pt', map_location='cpu', weights_only=True)
        cfg = ResponseAggregationConfig(**saved['metadata']['server_config'])
        shared = SharedBackbone(EdgePredictionConfig(**saved['metadata']['config']))
        server = ResponsePreservingServer(shared, replace(cfg, steps=max(budgets)))
        states = server.prepare_states([item['state_dict'] for item in payloads])
        banks = torch.load(folder / 'server/server_probes.pt', map_location='cpu', weights_only=True)
        calibration = [RelationProbe(**item) for item in banks['calibration']]
        diagnostic = [RelationProbe(**item) for item in banks['diagnostic']]
        refs = server.reference_cache(states, calibration)
        diag_refs = server.reference_cache(states, diagnostic)
        baseline = distribution(None, len(states), server.device)
        rho = baseline.clone()
        logits = baseline.log().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([logits], lr=cfg.lr)
        for step in range(max(budgets) + 1):
            optimizer.zero_grad(set_to_none=True)
            loss, terms = server.objective(logits, states, calibration, refs, baseline, rho)
            if step in budgets:
                weights = logits.detach().softmax(0)
                candidate = mix_parameters(states, weights, server.parameter_names)
                output_grad = torch.autograd.grad(cfg.output_weight * terms[0], logits, retain_graph=True)[0]
                response_grad = torch.autograd.grad(cfg.response_weight * terms[1], logits, retain_graph=True)[0]
                row = {'round': int(folder.name.split('_')[1]) + 1, 'steps': step,
                       'objective': float(loss.detach()), 'output_mse': float(terms[0].detach()),
                       'response_mse': float(terms[1].detach()), 'weights': weights.tolist(),
                       'output_gradient_norm': float(output_grad.norm()),
                       'response_gradient_norm': float(response_grad.norm()),
                       'diagnostic': server.diagnostics(candidate, diagnostic, diag_refs, rho)}
                rows.append(row)
                print(json.dumps(row), flush=True)
            if step < max(budgets):
                loss.backward()
                optimizer.step()
    result = {'method': 'CPU replay of identical saved uploads and probe banks; each round independent',
              'scope': 'conditional server optimization only; no changed FL trajectory or downstream claims',
              'selection': 'diagnostic probes reported only; never select weights/checkpoints',
              'budgets': budgets, 'rows': rows, 'skipped_rounds_without_checkpoints': skipped_rounds}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budgets', type=int, nargs='+', default=[0, 5, 10, 20, 50])
    args = parser.parse_args()
    if min(args.budgets) < 0 or max(args.budgets) < 1:
        parser.error('Use nonnegative budgets including at least one update')
    audit(args.run_root, args.output, args.budgets)
