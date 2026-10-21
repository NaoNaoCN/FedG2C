# ===== FedG2C ADD BEGIN A： explicit stage-one switches =====
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class LocalConfig:
    hidden_dim: int = 64
    num_experts: int = 4
    num_heads: int = 4
    num_layers: int = 2
    backbone: str = "mysage"
    composer: str = "hypernetwork_relu"
    relation_condition: bool = True
    objective: str = "joint_embedding_prediction"
    max_nodes: int = 32
    prediction_ratio: float = 0.25
    momentum: float = 0.99
    dropout: float = 0.0
    lr: float = 0.001
    weight_decay: float = 0.00001

    def __post_init__(self):
        if min(self.hidden_dim, self.num_experts, self.num_heads, self.num_layers) < 1:
            raise ValueError("Dimensions and counts must be positive")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.max_nodes < 2 or not 0 < self.prediction_ratio < 1:
            raise ValueError("Need max_nodes >= 2 and 0 < prediction_ratio < 1")
        if not 0 <= self.momentum < 1 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid momentum/dropout")
        if self.lr <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer settings")
        if self.backbone not in ("mysage", "gcn", "sage", "gat", "gin"):
            raise ValueError("Unsupported backbone")
        if self.composer not in ("hypernetwork_relu", "hypernetwork_linear", "standard_attention", "linear_gate"):
            raise ValueError("Unsupported composer")
        if self.objective not in ("joint_embedding_prediction", "masked_feature_prediction"):
            raise ValueError("Unsupported objective")

    def to_dict(self):
        return asdict(self)
# ===== FedG2C ADD END A =====
