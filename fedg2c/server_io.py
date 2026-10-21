# ===== FedG2C ADD BEGIN B：共享检查点检查与可移植来源记录 =====
from pathlib import Path
import hashlib
import json
import torch
from .edge_config import EdgePredictionConfig

SHARED_FIELDS=('hidden_dim','num_experts','num_heads','num_layers','backbone','composer','relation_condition','dropout')
RELATION_SCHEMA=['self','direct_edge','two_hop']


def state_digest(state):
    digest=hashlib.sha256()
    for key in sorted(state):
        value=state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str((str(value.dtype),tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_uploads(paths):
    payloads=[torch.load(p,map_location='cpu',weights_only=True) for p in paths]
    if not payloads:
        raise ValueError('No checkpoints')
    # ===== FedG2C ADD BEGIN B：仅磁盘封装校验，运行消息不需要阶段和版本字段 =====
    for payload in payloads:
        if not isinstance(payload, dict) or not isinstance(payload.get('metadata'), dict):
            raise ValueError('Expected a FedG2C local checkpoint envelope; runtime weight messages are not disk checkpoints')
        metadata = payload['metadata']
        if metadata.get('stage') != 'local_pretraining_only':
            raise ValueError('Checkpoint stage must be local_pretraining_only')
        if type(metadata.get('schema_version')) is not int or metadata['schema_version'] != 2:
            raise ValueError('Unsupported local checkpoint schema_version; expected 2')
        # Both previously saved local and federated v2 checkpoints remain valid:
        # these optional fields were not present in every historical v2 writer.
        if metadata.get('shared_schema_version', 1) != 1:
            raise ValueError('Unsupported shared_schema_version')
        if metadata.get('local_design', 'edge_prediction') != 'edge_prediction':
            raise ValueError('Expected edge_prediction local design')
        if not isinstance(payload.get('state_dict'), dict) or 'client_id' not in payload:
            raise ValueError('Local checkpoint requires state_dict and client_id')
    # ===== FedG2C ADD END B =====
    configs=[EdgePredictionConfig(**p['metadata']['config']) for p in payloads]
    first=configs[0]
    signature=tuple(getattr(first,k) for k in SHARED_FIELDS)
    ids=[str(p['client_id']) for p in payloads]
    if len(set(ids))!=len(ids):
        raise ValueError('Duplicate client ids')
    meta=[p['metadata'] for p in payloads]
    for p,c in zip(payloads,configs):
        if tuple(getattr(c,k) for k in SHARED_FIELDS)!=signature:
            raise ValueError('Incompatible shared architecture or relation settings')
        if p['metadata']['protocol_id']!=meta[0]['protocol_id']:
            raise ValueError('Mixed source protocols')
        if p['metadata'].get('relation_schema')!=RELATION_SCHEMA:
            raise ValueError('Relation schema mismatch')
        if p['metadata'].get('round_id',0)!=meta[0].get('round_id',0):
            raise ValueError('Cannot aggregate different communication rounds')
    fingerprints=[m.get('initial_shared_sha256') for m in meta]
    if any(fingerprints) and (not all(fingerprints) or len(set(fingerprints))!=1):
        raise ValueError('Common initialization fingerprint mismatch')
    if not any(fingerprints) and len({m.get('seed') for m in meta})!=1:
        raise ValueError('Legacy initialization metadata differs')
    provenance={'client_ids':ids,'protocol_id':meta[0]['protocol_id'],
                'round_id':meta[0].get('round_id',0),
                'initial_shared_sha256':fingerprints[0],
                'initialization_check':'matching fingerprint' if all(fingerprints) else 'legacy seed metadata only; common initialization and slot alignment assumed',
                'uploads':[{'filename':Path(p).name,'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in paths]}
    return first,[p['state_dict'] for p in payloads],provenance


def save_aggregation(folder,state,model_config,server_config,report,provenance,probe_banks):
    folder=Path(folder)
    metadata={'schema_version':3,'shared_schema_version':1,'stage':'server_aggregation',
              'config':model_config.to_dict(),'server_config':server_config.to_dict(),
              'protocol_id':provenance['protocol_id'],'round_id':report['round_id'],
              'relation_schema':RELATION_SCHEMA,'provenance':provenance,
              'state_sha256':state_digest(state),'torch_version':str(torch.__version__),
              'reference_code_manifest':'third_party_references/module_b/manifest.json'}
    torch.save({'metadata':metadata,'state_dict':state},folder/'shared_backbone.pt')
    (folder/'aggregation_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (folder/'run_metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    if probe_banks is not None:
        torch.save(probe_banks,folder/'server_probes.pt')
# ===== FedG2C ADD END B =====
