# ===== FedG2C ADD BEGIN B：已上传共享模型的独立聚合入口 =====
import argparse
import json
from pathlib import Path
import torch
from fedg2c.server_config import ResponseAggregationConfig
from fedg2c.response_server import ResponsePreservingServer
from fedg2c.server_io import load_uploads,save_aggregation
from fedg2c.model import SharedBackbone


def main(argv=None):
    parser=argparse.ArgumentParser(description='B: server-only aggregation of A shared checkpoints')
    parser.add_argument('--client-checkpoints',nargs='+',type=Path,required=True)
    parser.add_argument('--server-config',type=Path,required=True)
    parser.add_argument('--aggregation',choices=['fedavg','output_only','response'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=1)
    args=parser.parse_args(argv)
    if args.threads<1:
        parser.error('Positive thread count required')
    torch.set_num_threads(args.threads)
    raw=json.loads(args.server_config.read_text())
    if args.aggregation is not None:
        raw['mode']=args.aggregation
    config=ResponseAggregationConfig(**raw)
    model_config,states,provenance=load_uploads(args.client_checkpoints)
    args.output.mkdir(parents=True,exist_ok=False)
    server=ResponsePreservingServer(SharedBackbone(model_config),config,args.device)
    state,report=server.execute(states,round_id=provenance['round_id'])
    save_aggregation(args.output,state,model_config,config,report,provenance,server.last_probe_banks)
    print(json.dumps({'mode':config.mode,'weights':report['weights'],'extra_client_exchanges':0}))


if __name__=='__main__':
    main()
# ===== FedG2C ADD END B =====
