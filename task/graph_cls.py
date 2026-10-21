"""Linear-head training and evaluation for graph classification."""

import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from model.finetuning import compute_multitask_loss
from utils.basic_utils import evaluate, get_device_from_model, task2metric


def _zero_loss_result():
    return {"linear_loss": 0.0, "loss": 0.0}


def _add_regularizers(loss, model, z_local, z_shared, lambda_prompt, lambda_align):
    """Add optional prompt and local/shared representation penalties."""
    if lambda_prompt > 0 and hasattr(model, "prompt_readout"):
        loss = loss + lambda_prompt * model.prompt_readout.prompt_tokens.pow(2).mean()
    if lambda_align > 0 and z_local is not None and z_shared is not None:
        loss = loss + lambda_align * F.mse_loss(
            F.normalize(z_local, p=2, dim=-1),
            F.normalize(z_shared, p=2, dim=-1),
        )
    return loss


def _step_optimizers(loss, optimizer, rm_optimizer, scheduler):
    optimizer.zero_grad()
    if rm_optimizer is not None:
        rm_optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if rm_optimizer is not None:
        rm_optimizer.step()
    if scheduler is not None:
        scheduler.step()


def _encode_graph_with_aux(model, batch):
    """Pool graph embeddings while retaining optional local/shared auxiliaries."""
    edge_attr = getattr(batch, "edge_attr", None)
    if hasattr(model, "encode_with_aux"):
        z_node, z_local_node, z_shared_node = model.encode_with_aux(
            batch.x, batch.edge_index, edge_attr
        )
        return (
            global_mean_pool(z_node, batch.batch),
            global_mean_pool(z_local_node, batch.batch),
            global_mean_pool(z_shared_node, batch.batch),
        )
    return (
        model.encode_graph(batch.x, batch.edge_index, edge_attr, batch.batch, pool="mean"),
        None,
        None,
    )


def ft_graph(
    model,
    dataset,
    loader,
    optimizer,
    split,
    labels,
    num_classes,
    scheduler=None,
    *,
    rm_optimizer=None,
    lambda_prompt=0.0,
    lambda_align=0.0,
):
    """Fine-tune graph representations with the linear multi-task head."""
    del dataset, labels, num_classes
    model.train()
    device = get_device_from_model(model)

    if split["train"].sum() == 0:
        return _zero_loss_result()

    total_linear_loss = 0.0
    total_loss = 0.0
    num_batches = 0
    for batch in loader:
        batch = batch.to(device)
        z, z_local, z_shared = _encode_graph_with_aux(model, batch)
        y = batch.y.to(dtype=torch.float64)
        linear_loss = compute_multitask_loss(model.get_lin_logits(z).mean(1), y)
        loss = _add_regularizers(
            linear_loss, model, z_local, z_shared, lambda_prompt, lambda_align
        )
        _step_optimizers(loss, optimizer, rm_optimizer, scheduler)

        total_linear_loss += linear_loss.item()
        total_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        return _zero_loss_result()
    return {
        "linear_loss": total_linear_loss / num_batches,
        "loss": total_loss / num_batches,
    }


def _align_graph_labels(labels, pred):
    """Match graph labels to the task-head matrix shape before AUC scoring."""
    if labels.ndim == 1:
        if pred.shape[1] == 1:
            return labels.unsqueeze(1)
        return labels.unsqueeze(1).expand(-1, pred.shape[1])
    if labels.ndim == 2 and labels.shape[1] != pred.shape[1]:
        if labels.shape[1] < pred.shape[1]:
            return labels.expand(-1, pred.shape[1])
        return labels[:, : pred.shape[1]]
    return labels


def eval_graph_single(model, loader, task):
    """Evaluate a graph loader using unblended linear-head logits."""
    model.eval()
    device = get_device_from_model(model)
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            z = model.encode_graph(
                batch.x,
                batch.edge_index,
                getattr(batch, "edge_attr", None),
                batch.batch,
                pool="mean",
            )
            pred = model.get_lin_logits(z).mean(1)
            predictions.append(pred.detach())
            labels.append(_align_graph_labels(batch.y, pred))

    if not predictions:
        return 0.0
    return evaluate(torch.cat(predictions, dim=0), torch.cat(labels, dim=0), task)


def eval_graph(model, dataset, loader, split, labels, num_classes, task):
    """Evaluate graph AUC from the linear classifier without prototype fusion."""
    del dataset, split, labels, num_classes
    train_loader, val_loader, test_loader = loader
    return {
        "train": 0.0,
        "val": eval_graph_single(model, val_loader, task),
        "test": eval_graph_single(model, test_loader, task),
        "metric": task2metric[task],
    }
