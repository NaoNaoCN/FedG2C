# ===== FedG2C ADD BEGIN B：可微参数聚合与服务器响应保持 =====
"""Gefra uniform averaging boundary + learnable aggregation via functional_call.

FedLAW motivates weight learning, FedDF motivates server model evaluation;
our unlabeled relation-response objective is neither paper's original algorithm.
Only client logits are optimized. No private data, adapters or feedback calls.
"""
from copy import deepcopy
import torch
from torch.func import functional_call
from .server_probes import ProbeEvaluator, sample_probe_bank


def distribution(values,count,device):
    p=torch.ones(count,device=device,dtype=torch.float32) if values is None else torch.as_tensor(values,device=device,dtype=torch.float32)
    if p.shape != (count,) or not torch.isfinite(p).all() or (p<=0).any():
        raise ValueError('Fixed weights must be finite, positive and match clients')
    return (p/p.sum()).detach()


def mix_parameters(states,weights,parameter_names):
    """Named differentiable convex combination. Non-parameter buffers must agree."""
    mixed={}
    for name in states[0]:
        if name in parameter_names:
            mixed[name]=sum(weights[i]*state[name] for i,state in enumerate(states))
        else:
            if any(not torch.equal(states[0][name],s[name]) for s in states[1:]):
                raise ValueError('Nonidentical shared buffer cannot be averaged: '+name)
            mixed[name]=states[0][name]
    return mixed


class ResponsePreservingServer:
    def __init__(self,shared,config,device='cpu'):
        self.config=config
        self.device=torch.device(device)
        self.evaluator=ProbeEvaluator(deepcopy(shared).to(self.device).eval().requires_grad_(False))
        self.parameter_names=set(dict(shared.named_parameters()))
        composer=shared.composer
        self.response_available=composer.mode!='linear_gate' and composer.relation_condition
        if config.mode=='response' and not self.response_available:
            raise ValueError('Response mode requires an active relation-conditioned composer')
        self.last_probe_banks=None

    def prepare_states(self,uploads):
        if not uploads:
            raise ValueError('No uploaded shared models')
        expected=self.evaluator.shared.state_dict()
        states=[]
        for state in uploads:
            if state.keys()!=expected.keys():
                raise ValueError('Shared schema mismatch; require experts.* and composer.* only')
            clean={}
            for key,value in state.items():
                if value.shape!=expected[key].shape or value.dtype!=expected[key].dtype:
                    raise ValueError('Shared shape/dtype mismatch: '+key)
                if not torch.isfinite(value).all():
                    raise ValueError('Nonfinite uploaded state: '+key)
                clean[key]=value.detach().to(self.device).clone()
            states.append(clean)
        # Validate buffer rules before any optimization.
        mix_parameters(states,distribution(None,len(states),self.device),self.parameter_names)
        return states

    def evaluate_state(self,state,bank,measure_response=None):
        if measure_response is None:
            measure_response=self.response_available
        params={'shared.'+k:v for k,v in state.items()}
        values=[functional_call(self.evaluator,params,(probe,self.config.epsilon,measure_response),strict=True) for probe in bank]
        return torch.stack([v[0] for v in values]),torch.stack([v[1] for v in values])

    @torch.no_grad()
    def reference_cache(self,states,bank):
        values=[self.evaluate_state(s,bank) for s in states]
        return torch.stack([v[0] for v in values]),torch.stack([v[1] for v in values])

    def matching_terms(self,candidate,reference,rho):
        out,response=candidate
        ref_out,ref_response=reference
        per_out=(out.unsqueeze(0)-ref_out).square().flatten(1).mean(1)
        if response.numel():
            per_response=(response.unsqueeze(0)-ref_response).square().flatten(1).mean(1)
        else:
            per_response=torch.zeros_like(per_out)
        return (rho*per_out).sum(),(rho*per_response).sum(),per_out,per_response

    def objective(self,logits,states,bank,reference,baseline,rho):
        weights=logits.softmax(0)
        state=mix_parameters(states,weights,self.parameter_names)
        candidate=self.evaluate_state(state,bank,measure_response=self.config.mode=='response')
        out,response,_,_=self.matching_terms(candidate,reference,rho)
        prior=(weights-baseline).square().sum()
        loss=self.config.output_weight*out+self.config.prior_weight*prior
        if self.config.mode=='response':
            loss=loss+self.config.response_weight*response
        return loss,(out,response,prior)

    @torch.no_grad()
    def diagnostics(self,state,bank,reference,rho):
        output,response,per_output,per_response=self.matching_terms(self.evaluate_state(state,bank),reference,rho)
        return {'output_mse':float(output),'response_mse':float(response) if self.response_available else None,
                'per_client_output_mse':per_output.cpu().tolist(),
                'per_client_response_mse':per_response.cpu().tolist() if self.response_available else None,
                'reference_response_rms':float(reference[1].square().mean().sqrt()) if self.response_available else None}

    def execute(self,uploads,round_id=0,baseline_weights=None,reference_weights=None):
        if type(round_id) is not int or round_id<0:
            raise ValueError('Nonnegative integer round_id required')
        # ===== FedG2C ADD BEGIN B：Gefra 按客户端组织的上传消息；原列表接口保留 =====
        from collections.abc import Mapping
        if isinstance(uploads, Mapping):
            from .messages import states_from_client_messages
            uploads = states_from_client_messages(uploads)
        # ===== FedG2C ADD END B =====
        states=self.prepare_states(uploads)
        baseline=distribution(baseline_weights,len(states),self.device)
        rho=distribution(reference_weights,len(states),self.device)
        base_state=mix_parameters(states,baseline,self.parameter_names)
        if self.config.mode=='fedavg':
            # Exact Gefra fixed-average control; no probes or server optimization.
            self.last_probe_banks=None
            return {k:v.detach().cpu().clone() for k,v in base_state.items()}, {
                'mode':'fedavg','round_id':round_id,'weights':baseline.cpu().tolist(),
                'reference_weights':rho.cpu().tolist(),'optimization_steps':0,
                'extra_client_exchanges':0,'probe_diagnostics':None}
        seed=self.config.seed+round_id*1000003
        train=[p.to(self.device) for p in sample_probe_bank(self.config,self.evaluator.shared.composer.dim,self.config.calibration_graphs,seed)]
        diagnostic=[p.to(self.device) for p in sample_probe_bank(self.config,self.evaluator.shared.composer.dim,self.config.diagnostic_graphs,seed+500009)]
        # Per-call/per-round cache; never reuse old client reference values.
        refs=self.reference_cache(states,train)
        diagnostic_refs=self.reference_cache(states,diagnostic)
        logits=baseline.log().detach().clone().requires_grad_(True)
        optimizer=torch.optim.Adam([logits],lr=self.config.lr)
        history=[]
        for step in range(self.config.steps):
            optimizer.zero_grad(set_to_none=True)
            loss,terms=self.objective(logits,states,train,refs,baseline,rho)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite aggregation loss')
            loss.backward()
            if logits.grad is None or not torch.isfinite(logits.grad).all():
                raise FloatingPointError('Missing/nonfinite aggregation weight gradient')
            history.append({'step':step+1,'loss_before_update':float(loss.detach()),
                            'output_mse':float(terms[0].detach()),'response_mse':float(terms[1].detach()),
                            'weight_prior':float(terms[2].detach()),'gradient_norm':float(logits.grad.norm()),
                            'weights_before_update':logits.detach().softmax(0).cpu().tolist()})
            optimizer.step()
        weights=logits.detach().softmax(0)
        final=mix_parameters(states,weights,self.parameter_names)
        report={'mode':self.config.mode,'round_id':round_id,'weights':weights.cpu().tolist(),
                'baseline_weights':baseline.cpu().tolist(),'reference_weights':rho.cpu().tolist(),
                'optimization_steps':self.config.steps,'extra_client_exchanges':0,'history':history,
                'calibration_seed':seed,'diagnostic_seed':seed+500009,
                'calibration_before':self.diagnostics(base_state,train,refs,rho),
                'calibration_after':self.diagnostics(final,train,refs,rho),
                'diagnostic_before':self.diagnostics(base_state,diagnostic,diagnostic_refs,rho),
                'diagnostic_after':self.diagnostics(final,diagnostic,diagnostic_refs,rho),
                'selection':'last configured step; diagnostic probes never select weights',
                'probe_edge_attributes':'none; shared expert input only; not a molecular data generator'}
        self.last_probe_banks={'calibration':[p.tensors() for p in train],'diagnostic':[p.tensors() for p in diagnostic]}
        return {k:v.detach().cpu().clone() for k,v in final.items()},report
# ===== FedG2C ADD END B =====
