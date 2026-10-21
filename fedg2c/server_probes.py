# ===== FedG2C ADD BEGIN B：随机块模型探针与关系响应测量 =====
"""Library SBM generator; relation interventions are this project's adaptation.

Reference idea: FED-PUB (ICML 2023) uses random graph functional embeddings.
Here Gaussian mean=0,std=1 inputs enter the shared space, not private adapters.
GAD (ICML 2026) motivates response matching; we use central differences on
relation gates instead of its forward differences on diffusion noise.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import stochastic_blockmodel_graph
from .context import structural_relations


@dataclass
class RelationProbe:
    x: torch.Tensor
    edge_index: torch.Tensor
    relations: torch.Tensor
    pairs: torch.Tensor
    directions: torch.Tensor

    def to(self, device):
        return RelationProbe(*(getattr(self,k).to(device) for k in self.__dataclass_fields__))

    def tensors(self):
        return {k:getattr(self,k).detach().cpu().clone() for k in self.__dataclass_fields__}


def sample_probe_bank(config, hidden_dim, count, seed):
    """No labels, edge attributes or client statistics. Preserve global RNG state."""
    bank=[]
    n=sum(config.block_sizes)
    generator=torch.Generator().manual_seed(seed)
    for _ in range(count):
        # PyG's public generator has no generator argument. Isolate its RNG.
        graph_seed=int(torch.randint(0,2**31-1,(1,),generator=generator))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(graph_seed)
            edges=stochastic_blockmodel_graph(config.block_sizes,config.edge_probs,directed=config.directed)
        x=torch.randn(n,hidden_dim,generator=generator)  # fixed sigma=1
        relations=structural_relations(edges,n)
        all_pairs=torch.combinations(torch.arange(n),r=2)
        pairs=all_pairs[torch.randperm(len(all_pairs),generator=generator)[:config.readout_pairs]]
        support=relations.bool().any(-1)  # self/direct/two-hop, same schema as A
        directions=[]
        for _ in range(config.directions):
            delta=torch.randn(n,n,generator=generator)
            if not config.directed:
                delta=(delta+delta.t())/2
            delta=delta*support
            delta=delta/torch.linalg.vector_norm(delta).clamp_min(1e-12)
            directions.append(delta)
        bank.append(RelationProbe(x,edges,relations,pairs,torch.stack(directions)))
    return bank


def pair_readout(z,pairs):
    """Cosine similarities; not link probabilities and not task predictions."""
    unit=F.normalize(z,dim=-1,eps=1e-8)
    return (unit[pairs[:,0]]*unit[pairs[:,1]]).sum(-1)


class ProbeEvaluator(nn.Module):
    def __init__(self,shared):
        super().__init__()
        self.shared=shared

    def forward(self,probe,epsilon,measure_response=True):
        # One encoding per candidate/probe. Reuse for + and - perturbations,
        # retaining the candidate's graph back to aggregation weights.
        h=self.shared.encode_experts(probe.x,probe.edge_index,edge_attr=None)
        ones=torch.ones_like(probe.relations[:,:,0])
        output=pair_readout(self.shared.composer(h,probe.relations,relation_gate=ones),probe.pairs)
        responses=[]
        if measure_response:
            for delta in probe.directions:
                positive=self.shared.composer(h,probe.relations,relation_gate=ones+epsilon*delta)
                negative=self.shared.composer(h,probe.relations,relation_gate=ones-epsilon*delta)
                responses.append((pair_readout(positive,probe.pairs)-pair_readout(negative,probe.pairs))/(2*epsilon))
        response=torch.stack(responses) if responses else output.new_empty((0,len(output)))
        return output,response
# ===== FedG2C ADD END B =====
