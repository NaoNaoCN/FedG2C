# ===== FedG2C ADD BEGIN C：独立下游入口，旧 Gefra 微调代码原样保留 =====
"""Run from FedG2C/. Without --use-new-design, delegate to original Gefra."""
import argparse
import json
from pathlib import Path
import runpy
import sys


def main():
    switch=argparse.ArgumentParser(add_help=False)
    switch.add_argument('--use-new-design',action=argparse.BooleanOptionalAction,default=False)
    choice,rest=switch.parse_known_args()
    if not choice.use_new_design:
        sys.argv=[str(Path(__file__).with_name('finetune_gefra.py'))]+rest
        runpy.run_path(sys.argv[0],run_name='__main__')
        return
    parser=argparse.ArgumentParser(parents=[switch])
    parser.add_argument('--checkpoint',required=True)
    data=parser.add_mutually_exclusive_group(required=True)
    data.add_argument('--target-manifest')
    data.add_argument('--smoke',action='store_true')
    parser.add_argument('--smoke-level',choices=['graph','node'],default='graph')
    parser.add_argument('--config',default='config/fedg2c_target.json')
    parser.add_argument('--adaptation',choices=['none','lora','adapter','prompt'])
    parser.add_argument('--local-alpha',type=float)
    parser.add_argument('--learnable-alpha',action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    from fedg2c.target_model import TargetConfig
    from fedg2c.target_train import run_target
    from fedg2c.target_smoke import write_smoke_manifest
    values=json.loads(Path(args.config).read_text())
    for name in ['adaptation','local_alpha','learnable_alpha','epochs','seed']:
        if getattr(args,name) is not None:
            values[name]=getattr(args,name)
    config=TargetConfig(**values)
    if Path(args.output).exists():
        raise FileExistsError('Use a new output directory')
    manifest=args.target_manifest
    if args.smoke:
        manifest=write_smoke_manifest(str(args.output)+'_data',args.smoke_level)
    result=run_target(args.checkpoint,manifest,config,args.output,args.device)
    print(json.dumps({'best_epoch':result['best_epoch'],'test':result['test'],
                      'frozen_base_unchanged':result['frozen_base_unchanged']},indent=2))


if __name__=='__main__':
    main()
# ===== FedG2C ADD END C =====
