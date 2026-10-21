"""Linear-head training and evaluation for node classification."""

import torch
import torch.nn.functional as F

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


def _step_optimizers(
    loss, optimizer, rm_optimizer, scheduler, clip_riemannian_gradients=False
):
    optimizer.zero_grad()
    if rm_optimizer is not None:
        rm_optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if rm_optimizer is not None:
        if clip_riemannian_gradients:
            rm_parameters = [
                parameter
                for group in rm_optimizer.param_groups
                for parameter in group["params"]
                if parameter.grad is not None
            ]
            if rm_parameters:
                torch.nn.utils.clip_grad_norm_(rm_parameters, max_norm=1.0)
        rm_optimizer.step()
    if scheduler is not None:
        scheduler.step()


def ft_node(
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
    """Fine-tune node representations with the model's linear classifier."""
    del num_classes
    model.train()
    device = get_device_from_model(model)

    if loader is None:
        train_mask = split["train"]
        if train_mask.sum() == 0:
            return _zero_loss_result()

        x = dataset.x
        edge_index = dataset.edge_index
        edge_attr = getattr(dataset, "edge_attr", None)
        y = labels.to(device)
        if hasattr(model, "encode_with_aux"):
            z, z_local, z_shared = model.encode_with_aux(x, edge_index, edge_attr)
        else:
            z = model.encode(x, edge_index, edge_attr)
            z_local, z_shared = None, None

        linear_loss = F.cross_entropy(
            model.get_lin_logits(z[train_mask]).mean(1), y[train_mask].long()
        )
        loss = _add_regularizers(
            linear_loss,
            model,
            z_local[train_mask] if z_local is not None else None,
            z_shared[train_mask] if z_shared is not None else None,
            lambda_prompt,
            lambda_align,
        )
        _step_optimizers(
            loss,
            optimizer,
            rm_optimizer,
            scheduler,
            clip_riemannian_gradients=True,
        )
        return {"linear_loss": linear_loss.item(), "loss": loss.item()}

    if split["train"].sum() == 0:
        return _zero_loss_result()

    total_linear_loss = 0.0
    total_loss = 0.0
    num_batches = 0
    for batch in loader:
        batch = batch.to(device)
        batch_size = batch.batch_size
        edge_attr = getattr(batch, "edge_attr", None)
        y = batch.y[:batch_size]
        if hasattr(model, "encode_with_aux"):
            z, z_local, z_shared = model.encode_with_aux(
                batch.x, batch.edge_index, edge_attr
            )
            z = z[:batch_size]
            z_local = z_local[:batch_size]
            z_shared = z_shared[:batch_size]
        else:
            z = model.encode(batch.x, batch.edge_index, edge_attr)[:batch_size]
            z_local, z_shared = None, None

        linear_loss = F.cross_entropy(model.get_lin_logits(z).mean(1), y.long())
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


def eval_node(model, dataset, loader, split, labels, num_classes, task):
    """Evaluate node predictions generated solely by the linear head."""
    del num_classes
    model.eval()
    device = get_device_from_model(model)

    if loader is None:
        with torch.no_grad():
            z = model.encode(
                dataset.x, dataset.edge_index, getattr(dataset, "edge_attr", None)
            )
            pred = model.get_lin_logits(z).mean(1).softmax(dim=-1)
        y = labels.to(device)
        return {
            "train": evaluate(pred, y, task, split["train"]),
            "val": evaluate(pred, y, task, split["valid"]),
            "test": evaluate(pred, y, task, split["test"]),
            "metric": task2metric[task],
        }

    pred_by_node = {}
    labels_by_node = {}
    masks_by_node = {"train": {}, "valid": {}, "test": {}}
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch_size = batch.batch_size
            if hasattr(batch, "n_id"):
                node_ids = batch.n_id[:batch_size].cpu()
            else:
                node_ids = torch.arange(batch_size)

            z = model.encode(
                batch.x, batch.edge_index, getattr(batch, "edge_attr", None)
            )[:batch_size]
            pred = model.get_lin_logits(z).mean(1).softmax(dim=-1)
            y = labels[node_ids.to(labels.device)].to(device)

            for position, node_id in enumerate(node_ids.tolist()):
                if node_id in pred_by_node:
                    continue
                pred_by_node[node_id] = pred[position].detach().cpu()
                labels_by_node[node_id] = y[position].detach().cpu()
                for split_name in masks_by_node:
                    masks_by_node[split_name][node_id] = split[split_name][
                        node_id
                    ].item()

    if not pred_by_node:
        return {
            "train": 0.0,
            "val": 0.0,
            "test": 0.0,
            "metric": task2metric[task],
        }

    node_ids = sorted(pred_by_node)
    pred = torch.stack([pred_by_node[node_id] for node_id in node_ids])
    y = torch.stack([labels_by_node[node_id] for node_id in node_ids]).long()
    masks = {
        split_name: torch.tensor(
            [masks_by_node[split_name][node_id] for node_id in node_ids],
            dtype=torch.bool,
        )
        for split_name in masks_by_node
    }
    return {
        "train": evaluate(pred, y, task, masks["train"]),
        "val": evaluate(pred, y, task, masks["valid"]),
        "test": evaluate(pred, y, task, masks["test"]),
        "metric": task2metric[task],
    }
