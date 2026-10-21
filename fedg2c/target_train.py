# ===== FedG2C ADD BEGIN C：目标域训练、验证选择和一次最终测试 =====
from dataclasses import asdict
from pathlib import Path
import json
import hashlib
import torch
from torch.nn import functional as F
from .target_model import TargetConfig, TargetModel, load_shared
from .target_data import load_target_manifest
from .model import SharedBackbone
from .edge_config import EdgePredictionConfig


def classification_loss(logits, labels, objective):
    if objective == 'multiclass':
        return F.cross_entropy(logits, labels.reshape(-1).long()), labels.numel()
    mask = torch.isfinite(labels)
    count = int(mask.sum())
    if count == 0:
        return logits.sum() * 0, 0
    return F.binary_cross_entropy_with_logits(logits[mask], labels[mask]), count


def binary_ranking(scores, labels):
    """ROC trapezoids and non-interpolated AP, grouping tied scores explicitly."""
    positive, negative = int(labels.sum()), int((1-labels).sum())
    if positive == 0 or negative == 0:
        return {'roc_auc':None, 'average_precision':None}
    order = scores.argsort(descending=True, stable=True)
    score, y = scores[order], labels[order].double()
    endpoints = torch.cat((score[1:] != score[:-1], torch.ones(1, dtype=torch.bool)))
    tp = y.cumsum(0)[endpoints]
    fp = (1-y).cumsum(0)[endpoints]
    tpr, fpr = torch.cat((torch.zeros(1), tp / positive)), torch.cat((torch.zeros(1), fp / negative))
    return {'roc_auc':float(torch.trapezoid(tpr, fpr)),
            'average_precision':float(((tpr[1:]-tpr[:-1]) * tp / (tp+fp)).sum())}


def metrics(logits, labels, objective):
    loss, count = classification_loss(logits, labels, objective)
    result = {'loss':float(loss), 'observed_labels':count, 'samples':len(logits)}
    if objective == 'multiclass':
        result['accuracy'] = float((logits.argmax(-1)==labels.reshape(-1)).float().mean())
    else:
        per_task = []
        for j in range(logits.shape[1]):
            mask = torch.isfinite(labels[:,j])
            y, z = labels[mask,j], logits[mask,j]
            per_task.append(dict(task=j, observed=int(mask.sum()), positives=int(y.sum()), **binary_ranking(z,y)))
        result['per_task'] = per_task
        for key in ('roc_auc', 'average_precision'):
            valid = [r[key] for r in per_task if r[key] is not None]
            result[key] = sum(valid)/len(valid) if valid else None
            result[key+'_valid_tasks'] = len(valid)
    return result


def predict(model, samples, device):
    logits, labels = [], []
    for sample in samples:
        logits.append(model(sample.view.to(device), sample.root))
        labels.append(sample.label.to(device).reshape(1,-1))
    return torch.cat(logits), torch.cat(labels)


@torch.no_grad()
def evaluate(model, samples, device):
    model.eval()
    logits, labels = predict(model, samples, device)
    return metrics(logits.cpu(), labels.cpu(), model.task['objective'])


def run_target(checkpoint, manifest_path, config, output, device='cpu'):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new output directory')
    torch.manual_seed(config.seed)
    if str(device).startswith('cuda'):
        torch.cuda.manual_seed_all(config.seed)
    manifest, splits = load_target_manifest(manifest_path, config)
    shared, shared_config, meta = load_shared(checkpoint)
    model = TargetModel(shared, shared_config, manifest['feature_schema'], manifest['task'], config).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.lr, weight_decay=config.weight_decay)
    generator = torch.Generator().manual_seed(config.seed)
    history, best_loss, best_state, best_epoch = [], float('inf'), None, None
    for epoch in range(config.epochs):
        model.train()
        order = torch.randperm(len(splits['train']), generator=generator).tolist() # 打乱训练集样本顺序
        loss_sum, observed = 0.0, 0
        for start in range(0,len(order),config.batch_size):
            samples = [splits['train'][i] for i in order[start:start+config.batch_size]]
            logits, labels = predict(model, samples, device)
            loss, count = classification_loss(logits,labels,manifest['task']['objective'])
            if count == 0:
                continue
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5)
            optimizer.step()
            loss_sum += float(loss.detach())*count
            observed += count
        valid = evaluate(model,splits['valid'],device)
        history.append({'epoch':epoch+1,'train_loss':loss_sum/observed,'valid':valid})
        if valid['loss'] < best_loss:
            best_loss, best_epoch = valid['loss'], epoch+1
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if best_state is None:
        raise FloatingPointError('No finite validation checkpoint')
    model.load_state_dict(best_state, strict=True)
    if not model.frozen_base_unchanged():
        raise RuntimeError('A pretrained base parameter changed')
    # Test is evaluated ONLY after validation has selected a checkpoint.
    test = evaluate(model,splits['test'],device)
    report = {'stage':'target_adaptation', 'best_epoch':best_epoch, 'selection':'minimum validation loss; first tie',
              'history':history, 'test':test, 'frozen_base_unchanged':True,
              'local_alpha':float(torch.as_tensor(model.alpha()).detach().cpu()), 'parameters':model.parameter_report(),
              'split_sizes':{k:len(v) for k,v in splits.items()},
              'visibility':manifest['visibility'], 'protocol_id':manifest['protocol_id'],
              'smoke_only':manifest.get('smoke_only',False)}
    metadata = {'schema_version':1, 'stage':'target_adaptation', 'config':asdict(config),
                'shared_config':shared_config.to_dict(), 'task':manifest['task'],
                'feature_schema':manifest['feature_schema'], 'source_state_sha256':meta['state_sha256'],
                'source_protocol_id':meta['protocol_id'], 'source_round_id':meta['round_id'],
                'target_protocol_id':manifest['protocol_id'], 'visibility':manifest['visibility'],
                'manifest_sha256':hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
                'reference_code_manifest':'third_party_references/module_c/manifest.json'}
    output.mkdir(parents=True)
    torch.save({'metadata':metadata,'state_dict':best_state},output/'target_model.pt')
    (output/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'run_metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    return report


def load_target_model(path, device='cpu'):
    """Self-contained best-checkpoint reload; not an optimizer-resume checkpoint."""
    payload = torch.load(path,map_location='cpu',weights_only=True)
    meta = payload['metadata']
    if meta.get('stage')!='target_adaptation' or meta.get('schema_version')!=1:
        raise ValueError('Not a target model checkpoint')
    cfg = EdgePredictionConfig(**meta['shared_config'])
    model = TargetModel(SharedBackbone(cfg),cfg,meta['feature_schema'],meta['task'],TargetConfig(**meta['config']))
    model.load_state_dict(payload['state_dict'],strict=True)
    model.base_sha256 = meta['source_state_sha256']
    if not model.frozen_base_unchanged():
        raise ValueError('Saved target model contains modified pretrained base weights')
    return model.to(device).eval(), meta
# ===== FedG2C ADD END C =====
