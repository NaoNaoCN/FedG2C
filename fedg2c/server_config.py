# ===== FedG2C ADD BEGIN B：服务器响应保持配置 =====
from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class ResponseAggregationConfig:
    # Explicit settings: these are experiment inputs, not claimed paper optima.
    block_sizes: tuple
    edge_probs: tuple
    calibration_graphs: int
    diagnostic_graphs: int
    readout_pairs: int
    directions: int
    epsilon: float
    steps: int
    lr: float
    output_weight: float
    response_weight: float
    prior_weight: float
    seed: int = 2026
    mode: str = 'response'
    feature_std: float = 1.0
    directed: bool = False

    def __post_init__(self):
        if self.mode not in ('fedavg', 'output_only', 'response'):
            raise ValueError('Unknown aggregation mode')
        if not self.block_sizes or any(type(n) is not int or n < 1 for n in self.block_sizes):
            raise ValueError('Positive integer block sizes required')
        n, b = sum(self.block_sizes), len(self.block_sizes)
        if n < 2 or len(self.edge_probs) != b or any(len(row) != b for row in self.edge_probs):
            raise ValueError('Invalid block probability matrix')
        for i in range(b):
            for j in range(b):
                if not 0 <= self.edge_probs[i][j] <= 1:
                    raise ValueError('Invalid edge probability')
                if not self.directed and self.edge_probs[i][j] != self.edge_probs[j][i]:
                    raise ValueError('Undirected block probabilities must be symmetric')
        for key in ('calibration_graphs','diagnostic_graphs','readout_pairs','directions','steps'):
            if type(getattr(self,key)) is not int or getattr(self,key) < 1:
                raise ValueError(key+' must be a positive integer')
        if self.readout_pairs > n*(n-1)//2:
            raise ValueError('Too many distinct readout pairs')
        if self.feature_std != 1.0:
            raise ValueError('This protocol fixes Gaussian feature standard deviation to 1')
        for key in ('epsilon','lr','output_weight','response_weight','prior_weight'):
            if not math.isfinite(getattr(self,key)) or getattr(self,key) < 0:
                raise ValueError('Invalid '+key)
        if not 0 < self.epsilon < 1 or self.lr <= 0 or self.output_weight <= 0:
            raise ValueError('Need 0<epsilon<1, positive lr and output_weight')
        if self.mode == 'response' and self.response_weight <= 0:
            raise ValueError('Response mode needs a positive response weight')
        if type(self.seed) is not int or self.seed < 0 or type(self.directed) is not bool:
            raise ValueError('Invalid seed/directed')

    def to_dict(self):
        return asdict(self)
# ===== FedG2C ADD END B =====
