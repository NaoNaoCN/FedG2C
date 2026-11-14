"""Real-data GPU deployment check; outputs are never formal checkpoints."""
import gc
import json
from pathlib import Path
import sys
import time
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from fedg2c.data import iter_graphs, read_manifest
from fedg2c.edge_client import EdgePredictionClient
from fedg2c.edge_config import EdgePredictionConfig
from fedg2c.model import SharedBackbone
from fedg2c.node_ablation_matrix import build_plan, TARGETS, VARIANTS
from fedg2c.target_model import TargetConfig, TargetModel
from fedg2c.target_data import TargetDataCache, load_target_manifest
from fedg2c.target_train import predict, classification_loss

def main():
    torch.set_num_threads(1)
    base=json.loads((REPO/'config/fedg2c_nodes_ablation_20261114.json').read_text())
    base['matrix']['data_root']='/home/chenyh26/FedG2C/runs/gefra_loo_v1_data'
    configs,_=build_plan(base)
    report=dict(pretraining=[],target=[],passed=False)
    output=REPO/'runs/deployment_preflight_20261114/gpu_execution.json'
    def record(stage,row):
        report[stage].append(row)
        output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(stage=stage,**row)),flush=True)
    for variant in VARIANTS:
      cfg=configs[variant+'_cora']
      config=EdgePredictionConfig(**cfg['stage1']['module_a'])
      manifest=read_manifest(cfg['data']['source_manifest'])
      for name in ('citeseer','wikics'):
        torch.manual_seed(2026)
        item=next(v for v in manifest['clients'] if v['id']==name)
        client=EdgePredictionClient(name,config,SharedBackbone(config),item['feature_schema'],'cuda:0')
        torch.cuda.reset_peak_memory_stats()
        started=time.monotonic()
        metrics=client.pretrain(lambda:iter_graphs(item['graphs']),epochs=1,views_per_graph=1,seed=2026,max_steps=1,batch_size=1024,physical_batch_size=128,sampling_unit='node')
        torch.cuda.synchronize()
        row=dict(variant=variant,client=name,logical_batch=1024,physical_batch=128,seconds=round(time.monotonic()-started,2),peak_allocated_mib=round(torch.cuda.max_memory_allocated()/1024**2),peak_reserved_mib=round(torch.cuda.max_memory_reserved()/1024**2),loss=metrics['steps'][0]['loss'],optimizer_steps=len(metrics['steps']))
        assert len(metrics['steps'])==1 and all(torch.isfinite(p).all() for p in client.model.parameters())
        record('pretraining',row)
        del client,metrics
        gc.collect();torch.cuda.empty_cache()
    for variant in ('baseline','ablate_ab'):
      for name in TARGETS:
        cfg=configs[variant+'_'+name]
        shared_cfg=EdgePredictionConfig(**cfg['stage1']['module_a'])
        target_cfg=TargetConfig(**dict(cfg['stage2']['target'],seed=0,adaptation='lora'))
        cache=TargetDataCache(target_cfg.crop_cache_mb)
        manifest,splits=load_target_manifest(cfg['data']['target_manifest'],target_cfg,include_test=False,data_cache=cache)
        samples=[splits['train'][i] for i in range(min(target_cfg.microbatch_size,len(splits['train'])))]
        torch.manual_seed(2026)
        model=TargetModel(SharedBackbone(shared_cfg),shared_cfg,manifest['feature_schema'],manifest['task'],target_cfg).to('cuda:0').train()
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=target_cfg.lr)
        torch.cuda.reset_peak_memory_stats()
        started=time.monotonic()
        logits,labels=predict(model,samples,'cuda:0')
        loss,_=classification_loss(logits,labels,manifest['task']['objective'])
        loss.backward();optimizer.step()
        assert torch.isfinite(loss) and model.frozen_base_unchanged()
        assert all(torch.count_nonzero(model.get_submodule(n).lora_B)>0 for n in model.parameter_report()['lora_targets'])
        torch.cuda.synchronize()
        record('target',dict(variant=variant,target=name,microbatch=len(samples),max_nodes=target_cfg.max_nodes,seconds=round(time.monotonic()-started,2),loss=float(loss.detach()),peak_allocated_mib=round(torch.cuda.max_memory_allocated()/1024**2),peak_reserved_mib=round(torch.cuda.max_memory_reserved()/1024**2)))
        del model,optimizer,logits,labels,loss,samples,splits,cache
        gc.collect();torch.cuda.empty_cache()
    report['passed']=True
    output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    main()
