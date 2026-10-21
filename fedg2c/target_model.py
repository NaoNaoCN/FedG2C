# ===== FedG2C ADD BEGIN C：冻结共享基座与三种独立的下游适配 =====
"""Official loralib; AdapterGNN/GPF mechanism adaptations. See module_c manifest."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from .edge_config import EdgePredictionConfig
from .model import SharedBackbone, FeatureAdapter, EdgePassingGraphEncoder
from .context import structural_relations
from .server_io import state_digest, RELATION_SCHEMA
from .vendor.loralib import Linear as LoRALinear


@dataclass(frozen=True)
class TargetConfig:
    adaptation: str = 'lora'
    rank: int = 4
    lora_alpha: float = 4.0
    bottleneck: int = 8
    local_alpha: float = 0.0
    learnable_alpha: bool = False
    epochs: int = 20
    batch_size: int = 8
    lr: float = 0.001
    weight_decay: float = 0.0
    max_nodes: int = 128
    node_hops: int = 2
    seed: int = 2026

    def __post_init__(self):
        if self.adaptation not in ('none', 'lora', 'adapter', 'prompt'):
            raise ValueError('Unknown adaptation')
        for k in ('rank', 'bottleneck', 'epochs', 'batch_size', 'max_nodes', 'node_hops'):
            if type(getattr(self, k)) is not int or getattr(self, k) < 1:
                raise ValueError(f'{k} must be a positive integer')
        if not 0 <= self.local_alpha <= 1 or (self.learnable_alpha and not 0 < self.local_alpha < 1):
            raise ValueError('Alpha must be in [0,1]; learnable alpha needs an interior initialization')
        if not all(math.isfinite(v) for v in (self.lr, self.weight_decay, self.lora_alpha)):
            raise ValueError('Nonfinite optimizer/LoRA setting')
        if self.lr <= 0 or self.weight_decay < 0 or self.lora_alpha <= 0:
            raise ValueError('Invalid optimizer/LoRA setting')


def load_shared(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    meta = payload['metadata']
    if (meta.get('stage') != 'server_aggregation' or meta.get('schema_version') != 3
            or meta.get('shared_schema_version') != 1 or meta.get('relation_schema') != RELATION_SCHEMA):
        raise ValueError('Require a module B aggregated shared checkpoint with the current schema')
    if state_digest(payload['state_dict']) != meta.get('state_sha256'):
        raise ValueError('Shared checkpoint digest mismatch')
    cfg = EdgePredictionConfig(**meta['config'])
    shared = SharedBackbone(cfg)
    shared.load_state_dict(payload['state_dict'], strict=True)
    return shared, cfg, meta


class ResidualAdapter(nn.Module):
    """AdapterGNN GNN.__init__: down/ReLU/up, zero up; relocated after composer.

    Independent implementation, without its per-layer dual branches, BN or gates.
    """
    def __init__(self, dim, bottleneck):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, z):
        return z + self.up(torch.relu(self.down(z)))


def install_lora(composer, config):
    """Use official microsoft/LoRA Linear on Q and K; never merge base weights."""
    for name in ('query', 'key'):
        old = getattr(composer, name, None)
        if not isinstance(old, nn.Linear):
            raise ValueError('LoRA requires a composer with query/key Linear projections')
        new = LoRALinear(old.in_features, old.out_features, r=config.rank,
                         lora_alpha=config.lora_alpha, lora_dropout=0,
                         bias=old.bias is not None, merge_weights=False)
        with torch.no_grad():
            new.weight.copy_(old.weight)
            if old.bias is not None:
                new.bias.copy_(old.bias)
                new.bias.requires_grad_(False)
        setattr(composer, name, new)


class TargetModel(nn.Module):
    def __init__(self, shared, shared_config, feature_schema, task, config):
        super().__init__()
        self.config, self.task = config, dict(task)
        self.shared = shared.requires_grad_(False).eval()
        self.base_keys = tuple(shared.state_dict())
        self.base_sha256 = state_digest(shared.state_dict())
        dim = shared_config.hidden_dim
        # Common modules are initialized BEFORE mode-specific ones for paired seeds.
        self.node_adapter = FeatureAdapter(feature_schema['input_dim'], dim, feature_schema.get('node_cardinalities'))
        edge_dim = feature_schema.get('edge_dim', 0)
        if edge_dim and shared_config.backbone != 'mysage':
            raise ValueError('Edge attributes require mysage in this implementation')
        self.edge_adapter = FeatureAdapter(edge_dim, dim, feature_schema.get('edge_cardinalities')) if edge_dim else None
        self.head = nn.Linear(dim, task['num_outputs'])
        self.local_encoder = (EdgePassingGraphEncoder(dim, dim, nn.ReLU, shared_config.num_layers,
                              backbone=shared_config.backbone, normalize='layer', dropout=0)
                              if config.local_alpha > 0 or config.learnable_alpha else None)
        self.register_parameter('alpha_logit', nn.Parameter(torch.tensor(math.log(config.local_alpha / (1-config.local_alpha))))
                                if config.learnable_alpha else None)
        self.adapter = None
        self.register_parameter('prompt', None)
        if config.adaptation == 'lora':
            install_lora(self.shared.composer, config)
        elif config.adaptation == 'adapter':
            self.adapter = ResidualAdapter(dim, config.bottleneck)
        elif config.adaptation == 'prompt':
            # GPF SimplePrompt.add: additive feature vector. Our placement is the
            # expert-token input to the frozen composer; zero starts at the base.
            self.prompt = nn.Parameter(torch.zeros(1, 1, dim))
        if config.local_alpha == 1 and not config.learnable_alpha:
            # Pure-local control: exclude unused adaptation parameters from optimizer/counts.
            self.shared.requires_grad_(False)
            if self.adapter is not None:
                self.adapter.requires_grad_(False)
            if self.prompt is not None:
                self.prompt.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.shared.eval()  # Fixed dropout/statistics; autograd still traverses it.
        return self

    def alpha(self):
        return self.alpha_logit.sigmoid() if self.alpha_logit is not None else self.config.local_alpha

    def forward(self, view, root=None):
        x = self.node_adapter(view.x)
        if (self.edge_adapter is None) != (view.edge_attr is None):
            raise ValueError('Edge attributes must match the explicit target schema')
        edge = self.edge_adapter(view.edge_attr) if self.edge_adapter is not None else None
        pure_local = self.config.local_alpha == 1 and self.alpha_logit is None
        if not pure_local:
            h = self.shared.encode_experts(x, view.edge_index, edge)
            if self.prompt is not None:
                h = h + self.prompt
            relations = structural_relations(view.edge_index, view.num_nodes, dtype=h.dtype)
            z = self.shared.composer(h, relations)
            if self.adapter is not None:
                z = self.adapter(z)
        if self.local_encoder is not None:
            local = self.local_encoder(x, view.edge_index, edge)
            z = local if pure_local else self.alpha() * local + (1-self.alpha()) * z
        if self.task['level'] == 'graph':
            z = z.mean(dim=0, keepdim=True)
        else:
            if root is None or not 0 <= root < view.num_nodes:
                raise ValueError('Node prediction needs the remapped root index')
            z = z[root:root+1]
        return self.head(z)

    def frozen_base_unchanged(self):
        return state_digest({k:self.shared.state_dict()[k] for k in self.base_keys}) == self.base_sha256

    def parameter_report(self):
        groups = {}
        for name, value in self.named_parameters():
            if value.requires_grad:
                group = 'lora' if '.lora_' in name else name.split('.')[0]
                groups[group] = groups.get(group, 0) + value.numel()
        return {'trainable_by_group':groups, 'trainable_total':sum(groups.values()),
                'total_parameters':sum(p.numel() for p in self.parameters()),
                'trainable_names':[n for n,p in self.named_parameters() if p.requires_grad]}
# ===== FedG2C ADD END C =====
