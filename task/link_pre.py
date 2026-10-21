"""Linear-head training and evaluation for relation classification."""

import torch
import torch.nn.functional as F

from utils.basic_utils import evaluate, get_device_from_model, task2metric


def _edge_emb(z, edge_index):
    """Represent an edge by the mean of its endpoint embeddings."""
    return (z[edge_index[0]] + z[edge_index[1]]) / 2


def _zero_loss_result():
    return {"linear_loss": 0.0, "loss": 0.0}


def _add_regularizers(
    loss,
    model,
    z_local,
    z_shared,
    edge_index,
    lambda_prompt,
    lambda_align,
):
    """Add optional prompt and local/shared representation penalties."""
    if lambda_prompt > 0 and hasattr(model, "prompt_readout"):
        loss = loss + lambda_prompt * model.prompt_readout.prompt_tokens.pow(2).mean()
    if lambda_align > 0 and z_local is not None and z_shared is not None:
        loss = loss + lambda_align * F.mse_loss(
            F.normalize(_edge_emb(z_local, edge_index), p=2, dim=-1),
            F.normalize(_edge_emb(z_shared, edge_index), p=2, dim=-1),
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


def ft_link(
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
    """Fine-tune relation classification with the linear task head."""
    del num_classes
    model.train()
    device = get_device_from_model(model)

    if loader is None:
        train_mask = split["train"]
        if train_mask.sum() == 0:
            return _zero_loss_result()

        edge_index = dataset.edge_index
        edge_attr = getattr(dataset, "edge_attr", None)
        y = labels.to(device)
        if hasattr(model, "encode_with_aux"):
            z, z_local, z_shared = model.encode_with_aux(
                dataset.x, edge_index, edge_attr
            )
        else:
            z = model.encode(dataset.x, edge_index, edge_attr)
            z_local, z_shared = None, None

        train_edges = edge_index[:, train_mask]
        edge_z = _edge_emb(z, train_edges)
        linear_loss = F.cross_entropy(
            model.get_lin_logits(edge_z).mean(1), y[train_mask].long()
        )
        loss = _add_regularizers(
            linear_loss,
            model,
            z_local,
            z_shared,
            train_edges,
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
        edge_attr = getattr(batch, "edge_attr", None)
        if hasattr(model, "encode_with_aux"):
            z, z_local, z_shared = model.encode_with_aux(
                batch.x, batch.edge_index, edge_attr
            )
        else:
            z = model.encode(batch.x, batch.edge_index, edge_attr)
            z_local, z_shared = None, None

        edge_z = _edge_emb(z, batch.edge_label_index)
        linear_loss = F.cross_entropy(
            model.get_lin_logits(edge_z).mean(1), batch.edge_label.long()
        )
        loss = _add_regularizers(
            linear_loss,
            model,
            z_local,
            z_shared,
            batch.edge_label_index,
            lambda_prompt,
            lambda_align,
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


def eval_link(model, dataset, loader, split, labels, num_classes, task):
    """Evaluate relation predictions generated solely by the linear head."""
    del num_classes
    model.eval()
    device = get_device_from_model(model)

    if loader is None:
        with torch.no_grad():
            z = model.encode(
                dataset.x, dataset.edge_index, getattr(dataset, "edge_attr", None)
            )
            edge_z = _edge_emb(z, dataset.edge_index)
            pred = model.get_lin_logits(edge_z).mean(1).softmax(dim=-1)
        y = labels.to(device)
    else:
        pred_list = []
        labels_list = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                z = model.encode(
                    batch.x, batch.edge_index, getattr(batch, "edge_attr", None)
                )
                edge_z = _edge_emb(z, batch.edge_label_index)
                pred_list.append(
                    model.get_lin_logits(edge_z).mean(1).softmax(dim=-1).detach()
                )
                labels_list.append(batch.edge_label.detach())

        if not pred_list:
            return {
                "train": 0.0,
                "val": 0.0,
                "test": 0.0,
                "metric": task2metric[task],
            }
        pred = torch.cat(pred_list, dim=0)
        y = torch.cat(labels_list, dim=0)

    return {
        "train": evaluate(pred, y, task, split["train"]),
        "val": evaluate(pred, y, task, split["valid"]),
        "test": evaluate(pred, y, task, split["test"]),
        "metric": task2metric[task],
    }
