"""Record per-client pre-training losses, gradients, and diversity metrics.

Serialized JSON format:
{
  "client_0": {
    "rounds": [0, 1, ...],
    "raw_losses": {"feat_recon": [...], ...},
    "weighted_losses": {"feat_recon": [...], ...},
    "grad_norms": {"moe_encoder": [...], ...}
  },
  ...
}
"""

import json
import os

LOSS_KEYS = [
    'topo_recon', 'riem_topo_recon', 'commit', 'align',
    'struct_align', 'riem_proj', 'curvature_div_loss', 'total'
]

# Expert-diversity metrics recorded alongside losses and gradients.
DIVERSITY_KEYS = ['curvature_std', 'expert_entropy']

# Each group selects the parameters whose gradient norm is reported.
GRAD_GROUPS = {
    'moe_encoder':   lambda m: list(m.moe_encoder.parameters()),
    'local_encoder': lambda m: list(m.local_encoder.parameters()),
    'gating':        lambda m: list(m.gating.parameters()),
    'decoders':      lambda m: (list(m.feat_recon_decoder.parameters())
                                + list(m.topo_recon_decoder.parameters())
                                + list(m.topo_sem_recon_decoder.parameters())),
    'geo_proj':      lambda m: (list(m.geo_projector.parameters())),
}


def compute_grad_norms(pretrain_model):
    """Return gradient norms after backward propagation and before optimizer steps."""
    import torch
    norms = {}
    for gname, param_fn in GRAD_GROUPS.items():
        params = [p for p in param_fn(pretrain_model) if p.grad is not None]
        if params:
            norms[gname] = torch.stack(
                [p.grad.detach().norm() for p in params]
            ).norm().item()
        else:
            norms[gname] = 0.0
    return norms


def compute_diversity_metrics(pretrain_model, gate_probs_all):
    """Compute curvature spread and normalized routing entropy for one batch.

    ``curvature_std`` measures the spread of expert curvatures. Higher
    normalized ``expert_entropy`` indicates more balanced expert routing.
    """
    import torch
    import math
    experts = pretrain_model.moe_encoder.experts
    # Euclidean experts have no curvature and therefore use a zero spread.
    if hasattr(experts[0], 'encoder1'):
        ks = torch.stack([e.encoder1.manifold.k.detach() for e in experts])
        curvature_std = ks.std().item()
    else:
        curvature_std = 0.0
    # Compute entropy from the mean routing distribution.
    mean_probs = gate_probs_all.detach().mean(dim=0)
    entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum().item()
    max_entropy = math.log(len(experts))
    return {
        'curvature_std': curvature_std,
        'expert_entropy': entropy / (max_entropy + 1e-8),
    }


class PretrainMetricsLogger:
    def __init__(self, num_clients):
        self.data = {
            cid: {
                'rounds': [],
                'raw_losses':      {k: [] for k in LOSS_KEYS},
                'weighted_losses': {k: [] for k in LOSS_KEYS},
                'grad_norms':      {g: [] for g in GRAD_GROUPS},
                'diversity':       {k: [] for k in DIVERSITY_KEYS},
            }
            for cid in range(num_clients)
        }

    def log(self, round_id, client_id, raw_losses, weighted_losses, grad_norms, diversity=None):
        """Record epoch-mean metrics for one client and communication round."""
        d = self.data[client_id]
        d['rounds'].append(round_id)
        for k in LOSS_KEYS:
            d['raw_losses'][k].append(float(raw_losses.get(k, 0.0)))
            d['weighted_losses'][k].append(float(weighted_losses.get(k, 0.0)))
        for g in GRAD_GROUPS:
            d['grad_norms'][g].append(float(grad_norms.get(g, 0.0)))
        if diversity is not None:
            for k in DIVERSITY_KEYS:
                d['diversity'][k].append(float(diversity.get(k, 0.0)))

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.data, f, indent=2)
        print(f"[MetricsLogger] Saved to {path}")

    @staticmethod
    def load(path):
        with open(path) as f:
            return json.load(f)
