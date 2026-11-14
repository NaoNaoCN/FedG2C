# ===== FedG2C ADD BEGIN B：连接模块 A 与 B 的联邦轮次入口 =====
"""Single-process FL simulation, reusing Gefra's client set/pretrain/get lifecycle.

Clients keep private adapters/predictors locally. Only shared state_dict objects
are passed to the server. This is not a production networking implementation.
"""
# ===== FedG2C ADD BEGIN B：传输说明更新 =====
# The original docstring describes the payload. The active runtime envelope is
# now Gefra's client_id -> {num_samples, weight: [(name, tensor), ...]} mapping.
# ===== FedG2C ADD END B =====
import argparse
import json
from pathlib import Path
import torch
from fedg2c.edge_config import EdgePredictionConfig
from fedg2c.edge_client import EdgePredictionClient
from fedg2c.model import SharedBackbone
from fedg2c.data import read_manifest,iter_graphs,write_smoke_manifest,count_source_samples
from fedg2c.server_config import ResponseAggregationConfig
from fedg2c.response_server import ResponsePreservingServer
from fedg2c.server_io import state_digest,save_aggregation,RELATION_SCHEMA
from fedg2c.checkpointing import DEFAULT_CHECKPOINT_INTERVAL,should_save_checkpoint


def main(argv=None):
    parser=argparse.ArgumentParser(description='FedG2C modules A+B; original Gefra entry unchanged')
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--smoke',action='store_true')
    source.add_argument('--source-manifest',type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--server-config',type=Path,required=True)
    parser.add_argument('--aggregation',choices=['fedavg','sample_weighted','output_only','response'])
    parser.add_argument('--rounds',type=int,default=2)
    parser.add_argument('--checkpoint-interval',type=int,default=DEFAULT_CHECKPOINT_INTERVAL,
                        help='Save client/server weights and probes every N completed rounds and the final round')
    parser.add_argument('--local-epochs',type=int,default=1)
    parser.add_argument('--views-per-graph',type=int,default=1)
    parser.add_argument('--batch-size',type=int,default=1,
                        help='Independent source views per optimizer update')
    parser.add_argument('--physical-batch-size',type=int,default=1,
                        help='Views computed together; leaves optimizer/EMA update boundaries unchanged')
    parser.add_argument('--sampling-unit',choices=['graph','node'],default='graph')
    parser.add_argument('--local-steps',type=int)
    parser.add_argument('--source-order',choices=['sequential','shuffled_cycle','shuffled_epoch'],default='sequential')
    parser.add_argument('--log-every',type=int,default=0)
    parser.add_argument('--seed',type=int,default=2025)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.checkpoint_interval < 1:
        parser.error('--checkpoint-interval must be positive')
    if min(args.rounds,args.local_epochs,args.views_per_graph,args.threads,args.batch_size,args.physical_batch_size)<1 or (args.local_steps is not None and args.local_steps<1):
        parser.error('Positive counts required')
    if args.log_every < 0:
        parser.error('--log-every must be nonnegative')
    if args.source_order == 'shuffled_cycle' and (args.local_steps is None or args.local_epochs != 1):
        parser.error('shuffled_cycle requires --local-steps and --local-epochs 1')
    if args.source_order == 'shuffled_epoch' and args.local_steps is not None:
        parser.error('shuffled_epoch uses complete client passes; omit --local-steps')
    config_path=args.config
    if config_path is None and args.smoke:
        config_path=Path(__file__).parent/'config/fedg2c_edge_prediction.json'
    if config_path is None:
        parser.error('Real-source training requires explicit --config')
    config=EdgePredictionConfig(**json.loads(config_path.read_text()))
    raw=json.loads(args.server_config.read_text())
    if args.aggregation is not None:
        raw['mode']=args.aggregation
    server_config=ResponseAggregationConfig(**raw)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if not args.smoke:
        manifest=read_manifest(args.source_manifest)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'checkpoint_policy.json').write_text(json.dumps({
        'interval':args.checkpoint_interval,'round_numbering':'one-based completed rounds',
        'keep_final':True,'saved_round_ids':[r for r in range(args.rounds)
            if should_save_checkpoint(r,args.rounds,args.checkpoint_interval)]},indent=2)+'\n')
    if args.smoke:
        manifest=read_manifest(write_smoke_manifest(args.output/'smoke_inputs'))
    source_counts = ([count_source_samples(r['graphs'], args.sampling_unit) for r in manifest['clients']]
                     if server_config.mode == 'sample_weighted' else None)
    shared=SharedBackbone(config)
    server=ResponsePreservingServer(shared,server_config,args.device)
    clients=[EdgePredictionClient(r['id'],config,shared,r['feature_schema'],args.device) for r in manifest['clients']]
    source_pools = None
    if args.source_order in ('shuffled_cycle','shuffled_epoch'):
        from fedg2c.source_sampling import ShuffledGraphCycle
        source_pools = [ShuffledGraphCycle(record['graphs'], args.seed + index * 104729,
                                           record.get('graph_counts'))
                        for index, record in enumerate(manifest['clients'])]
        print(json.dumps({'stage':'source_pools_ready',
                          'graph_counts':{record['id']:len(pool.graphs) for record,pool in zip(manifest['clients'],source_pools)}}),flush=True)
    global_state={k:v.detach().cpu().clone() for k,v in shared.state_dict().items()}
    traffic=[]
    for round_id in range(args.rounds):
        save_checkpoint=should_save_checkpoint(round_id,args.rounds,args.checkpoint_interval)
        folder=args.output/f'round_{round_id:03d}'
        folder.mkdir()
        start_hash=state_digest(global_state)
        uploads=[]
        # ===== FedG2C ADD BEGIN B：按 Gefra 的下发列表／上传消息字典传输 =====
        from fedg2c.messages import named_weights
        global_message = named_weights(global_state)
        local_messages = {}
        # ===== FedG2C ADD END B =====
        
        
        for index,(client,record) in enumerate(zip(clients,manifest['clients'])):
            # ===== FedG2C ADD BEGIN B：替代原状态字典下发，旧调用保留供审阅 =====
            # Previous: client.set_pretrain_model(global_state)
            client.set_pretrain_model(global_message)
            # ===== FedG2C ADD END B =====
            graphs_factory = lambda record=record:iter_graphs(record['graphs'])
            selected = None
            if source_pools is not None:
                pool = source_pools[index]
                graph_budget = ((args.local_steps * args.batch_size + args.views_per_graph - 1) // args.views_per_graph
                                if args.source_order == 'shuffled_cycle' else len(pool.graphs)*args.local_epochs)
                if args.sampling_unit == 'node' and args.source_order == 'shuffled_cycle':
                    graph_budget = args.local_steps * args.batch_size
                selected = pool.indices(round_id * graph_budget, graph_budget)
                if args.source_order == 'shuffled_cycle':
                    graphs_factory = lambda pool=pool,selected=selected:pool.iter_selected(selected)
                else:
                    # Each factory call is one complete independently shuffled pass.
                    epoch_slices = iter([selected[start:start+len(pool.graphs)]
                                         for start in range(0,graph_budget,len(pool.graphs))])
                    graphs_factory = lambda pool=pool,slices=epoch_slices:pool.iter_selected(next(slices))
            print(json.dumps({'stage':'client_start','round':round_id,'client':record['id']}),flush=True)
            local=client.pretrain(graphs_factory,args.local_epochs,
                                 args.views_per_graph,args.seed+round_id*10007+index,args.local_steps,
                                 log_every=args.log_every, batch_size=args.batch_size,
                                 sampling_unit=args.sampling_unit, physical_batch_size=args.physical_batch_size)
            if selected is not None:
                local['sampling'] = {'mode':args.source_order,'pool_size':len(pool.graphs),
                                     'seed':pool.seed,'stream_start':round_id*graph_budget,
                                     'graph_indices':selected,'unique_graphs':len(set(selected))}
            state=client.get_pretrain_model()
            uploads.append(state)
            # ===== FedG2C ADD BEGIN B：上传 Gefra 形状消息，不包含磁盘元数据 =====
            num_samples = source_counts[index] if source_counts is not None else 1
            local_messages[record['id']] = client.get_local_message(num_samples=num_samples)
            if source_counts is not None:
                local.update(num_samples=num_samples, sample_unit=args.sampling_unit)
            # ===== FedG2C ADD END B =====
            client_folder=folder/f'client_{index}'
            client_folder.mkdir()
            metadata={'schema_version':2,'config':config.to_dict(),'protocol_id':manifest['protocol_id'],
                      'round_id':round_id,'seed':args.seed,'initial_shared_sha256':start_hash,
                      'relation_schema':RELATION_SCHEMA,'stage':'local_pretraining_only',
                      'device':str(args.device),'source_order':args.source_order,
                      'local_epochs':args.local_epochs,'views_per_graph':args.views_per_graph,
                      'local_steps':args.local_steps,'batch_size':args.batch_size,
                      'sampling_unit':args.sampling_unit,'physical_batch_size':args.physical_batch_size}
            if source_counts is not None:
                metadata.update(num_samples=num_samples, sample_unit=args.sampling_unit)
            if save_checkpoint:
                torch.save({'metadata':metadata,'client_id':record['id'],'state_dict':state},client_folder/'shared_backbone.pt')
            (client_folder/'metrics.json').write_text(json.dumps(local,indent=2,allow_nan=False)+'\n')
        # ===== FedG2C ADD BEGIN B：替代原上传列表；服务器内部仍用相同参数聚合 =====
        # Previous: global_state,report=server.execute(uploads,round_id=round_id)
        global_state,report=server.execute(local_messages,round_id=round_id)
        # ===== FedG2C ADD END B =====
        provenance={'protocol_id':manifest['protocol_id'],'client_ids':[r['id'] for r in manifest['clients']],
                    'initial_shared_sha256':start_hash,'initialization_check':'all clients received same round state',
                    'uploaded_state_sha256':[state_digest(s) for s in uploads]}
        server_folder=folder/'server';server_folder.mkdir()
        save_aggregation(server_folder,global_state,config,server_config,report,provenance,server.last_probe_banks,
                         save_checkpoint=save_checkpoint)
        state_bytes=sum(v.numel()*v.element_size() for v in global_state.values())
        traffic.append({'round_id':round_id,'clients':len(clients),'downlink_shared_bytes':len(clients)*state_bytes,
                        'uplink_shared_bytes':len(clients)*state_bytes,'extra_candidate_messages':0,
                        'note':'logical tensor payload; no serialization/network overhead counted'})
        print(json.dumps({'round':round_id,'mode':server_config.mode,'weights':report['weights']}),flush=True)
    (args.output/'communication.json').write_text(json.dumps(traffic,indent=2)+'\n')
    # Local snapshots are private files, never passed to ResponsePreservingServer.
    for index,client in enumerate(clients):
        torch.save({'state_dict':client.model.state_dict(),'config':config.to_dict(),
                    'feature_schema':client.feature_schema,'note':'weights only, not exact optimizer/RNG resume'},
                   args.output/f'client_{index}_private_training_state.pt')


if __name__=='__main__':
    main()
# ===== FedG2C ADD END B =====
