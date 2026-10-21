# ===== FedG2C ADD BEGIN A： nodewise latent prediction configuration (revision 2) =====
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class EdgePredictionConfig:
    """
    模块A：关系条件专家组合学习配置类
    """
    
    # Required: no silent transfer of augmentation rates between domains.
    edge_mask_rate: float
    feature_mask_rate: float
    hidden_dim: int = 64
    num_experts: int = 4
    num_heads: int = 4
    num_layers: int = 2
    backbone: str = "mysage"
    composer: str = "hypernetwork_relu"
    relation_condition: bool = True
    max_nodes: int = 32
    context_hops: int = 2
    prediction_hops: int = 1
    undirected: bool = True
    position_encoding: str = "random_walk"
    walk_steps: tuple = (1, 2, 3, 4, 5, 6, 7, 8)
    # ===== FedG2C ADD BEGIN A： predictor choice; legacy depth/heads apply to Transformer =====
    predictor_type: str = "graphsage"
    # ===== FedG2C ADD END A =====
    predictor_dim: int = 64
    predictor_heads: int = 4
    predictor_depth: int = 2
    momentum: float = 0.99
    dropout: float = 0.0
    lr: float = 0.001
    weight_decay: float = 0.00001

    def __post_init__(self):
        # ===== FedG2C ADD BEGIN A： validate explicit predictor route =====
        if self.predictor_type not in ("graphsage", "transformer"):
            raise ValueError("predictor_type must be graphsage or transformer")
        # ===== FedG2C ADD END A =====
        for key in ("hidden_dim", "num_experts", "num_heads", "num_layers", "max_nodes",
                    "context_hops", "predictor_dim", "predictor_heads", "predictor_depth"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.max_nodes < 2 or not isinstance(self.prediction_hops, int) or self.prediction_hops < 0:
            raise ValueError("Need at least two view nodes and nonnegative prediction_hops")
        # ===== FedG2C ADD BEGIN A： head divisibility only for attention modules =====
        # Previous condition: self.hidden_dim % self.num_heads or self.predictor_dim % self.predictor_heads
        if self.hidden_dim % self.num_heads or (self.predictor_type == "transformer" and self.predictor_dim % self.predictor_heads):
            raise ValueError("Attention dimensions must be divisible by head counts")
        # ===== FedG2C ADD END A =====
        if not 0 <= self.edge_mask_rate < 1 or not 0 <= self.feature_mask_rate < 1:
            raise ValueError("Masking rates must be in [0,1); zero is an explicit ablation")
        if not 0 <= self.momentum < 1 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid momentum/dropout")
        if self.lr <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer parameters")
        if self.backbone not in ("mysage", "gcn", "sage", "gat", "gin"):
            raise ValueError("Unsupported backbone")
        if self.composer not in ("hypernetwork_relu", "hypernetwork_linear", "standard_attention", "linear_gate"):
            raise ValueError("Unsupported composer")
        if self.composer == "linear_gate" and self.relation_condition:
            raise ValueError("linear_gate requires relation_condition=false")
        if self.position_encoding not in ("random_walk", "none"):
            raise ValueError("Unsupported structural encoding")
        if not self.walk_steps or any(not isinstance(k, int) or k < 1 for k in self.walk_steps):
            raise ValueError("walk_steps must contain positive integers")
        if len(set(self.walk_steps)) != len(self.walk_steps):
            raise ValueError("walk_steps must be unique")
        if not isinstance(self.undirected, bool) or not isinstance(self.relation_condition, bool):
            raise ValueError("undirected and relation_condition must be boolean")

    def to_dict(self):
        return asdict(self)
# ===== FedG2C ADD END A =====
