"""Checkpoint-directory helpers shared by public training entry points."""

import os.path as osp


def build_pretrain_checkpoint_dir(repository_root, datasets, client_counts, args):
    """Build the checkpoint directory for one pre-training configuration.

    The naming convention intentionally matches the historical public paths:
    source datasets and client counts define the base name; backbone, projector,
    ablation, and non-default top-k settings define its suffix. ``hidden_dim``
    and ``num_experts`` remain checkpoint compatibility requirements but are not
    directory components.
    """
    if len(datasets) != len(client_counts):
        raise ValueError("datasets and client_counts must have the same length.")

    base = "_".join(
        f"{dataset.lower()}_{client_count}"
        for dataset, client_count in zip(datasets, client_counts)
    )
    ablation_moe = getattr(args, "ablation_moe", False)
    ablation_no_geo = getattr(args, "ablation_no_geo_projector", False)
    ablation_fedavg = getattr(args, "ablation_fedavg", False)
    backbone = getattr(args, "gnn_backbone", "gcn").lower()

    if ablation_moe or ablation_no_geo or ablation_fedavg:
        suffix = "_ablation"
        if ablation_no_geo:
            suffix += "_no_geo_proj"
        if ablation_fedavg:
            suffix += "_fedavg"
        if ablation_moe:
            suffix += "_moe"
    elif backbone == "gat":
        suffix = "_gat"
    elif backbone == "sage":
        suffix = "_sage"
    else:
        suffix = "_new"

    if (
        getattr(args, "use_strict_projector", True)
        and not ablation_no_geo
        and not ablation_moe
    ):
        suffix += "_strict"

    checkpoint_dir = osp.join(repository_root, "ckpts", "pretrain_models", f"{base}{suffix}")
    topk = getattr(args, "topk", 2)
    if topk != 2:
        checkpoint_dir += f"_{topk}-exp_select"
    return checkpoint_dir
