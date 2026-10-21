# ===== FedG2C ADD BEGIN C：不代表泛化结果的合成接口检查数据 =====
from pathlib import Path
import json
import torch


def write_smoke_manifest(folder, level='graph'):
    folder = Path(folder)
    folder.mkdir(parents=True,exist_ok=False)
    g = torch.Generator().manual_seed(123)
    splits = {k:[] for k in ('train','valid','test')}
    def graph(n, gid):
        u = torch.arange(n)
        edges = torch.stack((u,(u+1)%n))
        return {'x':torch.randn(n,5,generator=g),'edge_index':torch.cat((edges,edges.flip(0)),dim=1),'graph_id':gid}
    if level=='graph':
        for split,count in [('train',6),('valid',4),('test',4)]:
            for i in range(count):
                name=f'{split}_{i}.pt'
                data=graph(6, name)
                data['y']=torch.tensor([float(i%2)])
                torch.save(data,folder/name)
                splits[split].append({'path':name})
        task={'level':'graph','objective':'binary','num_outputs':1}
        visibility='split_graphs'
    else:
        data=graph(14,'smoke_nodes');data['y']=torch.arange(14)%2
        torch.save(data,folder/'graph.pt')
        for split,ids in [('train',list(range(6))),('valid',list(range(6,10))),('test',list(range(10,14)))]:
            splits[split]=[{'path':'graph.pt','indices':ids}]
        task={'level':'node','objective':'multiclass','num_outputs':2}
        visibility='transductive'
    manifest={'schema_version':1,'protocol_id':'smoke-only-not-compositional-evidence','smoke_only':True,
              'task':task,'visibility':visibility,'feature_schema':{'input_dim':5,'edge_dim':0},'splits':splits}
    path=folder/'manifest.json';path.write_text(json.dumps(manifest,indent=2)+'\n')
    return path
# ===== FedG2C ADD END C =====
