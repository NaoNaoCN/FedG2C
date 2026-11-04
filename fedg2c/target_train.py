# ===== FedG2C ADD BEGIN C：目标域训练、验证选择和一次最终测试 =====
from dataclasses import asdict
from pathlib import Path
import json
import hashlib
import time
import math
import torch
from torch.nn import functional as F
from .target_model import TargetConfig, TargetModel, load_shared
from .target_data import load_target_manifest, select_support, precompute_topology
from .model import SharedBackbone
from .edge_config import EdgePredictionConfig
from .batching import GraphBatch


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
    if len(samples) > 1:
        view = GraphBatch.collate([s.view for s in samples]).to(device, non_blocking=True)
        roots, relations = None, None
        if model.task['level'] == 'node':
            if any(s.root is None or not 0 <= s.root < s.view.num_nodes for s in samples):
                raise ValueError('Node prediction needs the remapped root index')
            roots = torch.tensor([s.root for s in samples], device=device)
            if all(s.root_relations is not None for s in samples):
                relations = torch.stack([F.pad(s.root_relations, (0, 0, 0, view.max_nodes-s.view.num_nodes))
                                         for s in samples]).to(device, non_blocking=True)
        labels = torch.cat([s.label.reshape(1, -1) for s in samples]).to(device, non_blocking=True)
        return model.forward_batch(view, roots, relations), labels
    logits, labels = [], []
    for sample in samples:
        relations = (None if sample.root_relations is None else
                     sample.root_relations.to(device, non_blocking=True))
        logits.append(model(sample.view.to(device), sample.root, relations))
        labels.append(sample.label.to(device).reshape(1,-1))
    return torch.cat(logits), torch.cat(labels)


@torch.no_grad()
def evaluate(model, samples, device):
    model.eval()
    logits, labels = [], []
    for start in range(0, len(samples), model.config.eval_batch_size):
        batch = [samples[i] for i in range(start, min(start + model.config.eval_batch_size, len(samples)))]
        z, y = predict(model, batch, device)
        logits.append(z.detach().cpu())
        labels.append(y.cpu())
    return metrics(torch.cat(logits), torch.cat(labels), model.task['objective'])


def run_target(checkpoint, manifest_path, config, output, device='cpu', data_cache=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new output directory')
    torch.manual_seed(config.seed)
    if str(device).startswith('cuda'):
        torch.cuda.manual_seed_all(config.seed)
    manifest, splits = load_target_manifest(manifest_path, config, include_test=config.evaluate_test,
                                           data_cache=data_cache)
    train_count = len(splits['train'])
    splits['train'], support = select_support(splits['train'], manifest['task'], config.k_shot)
    if config.selection_metric == 'accuracy' and manifest['task']['objective'] != 'multiclass':
        raise ValueError('accuracy selection requires a multiclass task')
    if config.selection_metric in ('roc_auc', 'average_precision') and manifest['task']['objective'] == 'multiclass':
        raise ValueError('Ranking selection requires a binary/multilabel task')
    started = time.monotonic()
    topology = precompute_topology(splits)
    if topology['enabled']:
        topology['elapsed_seconds'] = round(time.monotonic()-started, 3)
        print(json.dumps({'stage':'target_topology', **topology}), flush=True)
    shared, shared_config, meta = load_shared(checkpoint)
    model = TargetModel(shared, shared_config, manifest['feature_schema'], manifest['task'], config).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.lr, weight_decay=config.weight_decay)
    output.mkdir(parents=True)
    generator = torch.Generator().manual_seed(config.seed)
    minimize = config.selection_metric == 'loss'
    history, best_score, best_state, best_epoch = [], (float('inf') if minimize else -float('inf')), None, None
    stale_epochs, best_valid, optimizer_steps = 0, None, 0
    parameters = [p for p in model.parameters() if p.requires_grad]
    epochs_ran = 0
    for epoch in range(config.epochs):
        epochs_ran = epoch + 1
        started = time.monotonic()
        model.train()
        order = torch.randperm(len(splits['train']), generator=generator).tolist() # 打乱训练集样本顺序
        loss_sum, observed = torch.zeros((), device=device), 0
        logical_batch = config.batch_size or len(order)
        for start in range(0,len(order),logical_batch):
            indices = order[start:start+logical_batch]
            optimizer.zero_grad(set_to_none=True)
            batch_count = 0
            # Accumulate unnormalized loss sums, then divide once by the actual
            # number of observed labels. This is exactly BCE's logical-batch
            # reduction even with PCBA NaNs or a final incomplete microbatch.
            for offset in range(0,len(indices),config.microbatch_size):
                samples = [splits['train'][i] for i in indices[offset:offset+config.microbatch_size]]
                logits, labels = predict(model, samples, device)
                loss, count = classification_loss(logits,labels,manifest['task']['objective'])
                if not count:
                    continue
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                (loss * count).backward()
                loss_sum += loss.detach()*count
                observed += count
                batch_count += count
            if not batch_count:
                continue
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.div_(batch_count)
            torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            optimizer_steps += 1
        if not observed:
            raise ValueError('Training epoch has no observed labels')
        scheduled = epoch % config.eval_interval == 0 or epoch == config.epochs-1
        valid = evaluate(model,splits['valid'],device) if scheduled else None
        if scheduled:
            train_loss = float(loss_sum.cpu()) / observed
            history.append({'epoch':epoch+1,'train_loss':train_loss,'valid':valid})
            (output/'history.json').write_text(json.dumps(history,indent=2,allow_nan=False)+'\n')
            print(json.dumps({'stage':'target_epoch','adaptation':config.adaptation,'epoch':epoch+1,
                              'train_loss':train_loss,'valid':valid,
                              'elapsed_seconds':round(time.monotonic()-started,2)}),flush=True)
        if not scheduled:
            continue
        score = valid.get(config.selection_metric)
        if score is None or not math.isfinite(score):
            raise ValueError('Validation selection metric is undefined/nonfinite')
        if (score < best_score if minimize else score > best_score):
            best_score, best_epoch, stale_epochs = score, epoch+1, 0
            best_valid = valid
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale_epochs += 1
        if config.early_stopping_patience and stale_epochs >= config.early_stopping_patience:
            break
    if best_state is None:
        raise FloatingPointError('No finite validation checkpoint')
    model.load_state_dict(best_state, strict=True)
    if not model.frozen_base_unchanged():
        raise RuntimeError('A pretrained base parameter changed')
    # Test is evaluated ONLY after validation has selected a checkpoint.
    test = evaluate(model,splits['test'],device) if config.evaluate_test else None
    selection = ('minimum' if minimize else 'maximum') + ' validation ' + config.selection_metric + '; first tie'
    report = {'stage':'target_adaptation', 'best_epoch':best_epoch, 'selection':selection,
              'validation':best_valid, 'epochs_ran':epochs_ran,
              'validation_events':len(history), 'stopped_early':epochs_ran<config.epochs,
              'test_evaluated':config.evaluate_test,
              'history':history, 'test':test, 'frozen_base_unchanged':True,
              'local_alpha':float(torch.as_tensor(model.alpha()).detach().cpu()), 'parameters':model.parameter_report(),
              'split_sizes':{k:len(v) for k,v in splits.items()},
              'original_train_size':train_count, 'k_shot':config.k_shot,
              'support_indices_in_train':support,
              'support_sha256':hashlib.sha256(json.dumps([str(splits['train'].refs[i])
                                  for i in range(len(splits['train']))]).encode()).hexdigest(),
              'optimizer_steps':optimizer_steps, 'logical_batch_size':config.batch_size,
              'microbatch_size':config.microbatch_size, 'eval_batch_size':config.eval_batch_size,
              'crop_cache_mb':config.crop_cache_mb, 'eval_interval':config.eval_interval,
              'root_query_only':manifest['task']['level'] == 'node',
              'topology_precompute':topology, 'topology_cache_final':splits['train'].data_cache.stats(),
              'patience_unit':'validation events',
              'visibility':manifest['visibility'], 'protocol_id':manifest['protocol_id'],
              'smoke_only':manifest.get('smoke_only',False)}
    metadata = {'schema_version':1, 'stage':'target_adaptation', 'config':asdict(config),
                'shared_config':shared_config.to_dict(), 'task':manifest['task'],
                'feature_schema':manifest['feature_schema'], 'source_state_sha256':meta['state_sha256'],
                'source_protocol_id':meta['protocol_id'], 'source_round_id':meta['round_id'],
                'target_protocol_id':manifest['protocol_id'], 'visibility':manifest['visibility'],
                'manifest_sha256':hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
                'reference_code_manifest':'third_party_references/module_c/manifest.json'}
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
