"""Training-label perturbation for Gefra robustness experiments."""

import torch


def apply_finetune_label_noise(data, rate=0.3):
    """Replace a fraction of training labels in place and return its sample count.

    Validation and test labels remain unchanged, so noisy-supervision runs are
    evaluated on clean held-out labels. Graph collections store labels in the
    backing ``_data`` object used by PyG's reconstruction path.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"Label-noise rate must be in [0, 1], received {rate}.")
    if not hasattr(data, "train_mask") or not hasattr(data, "num_global_classes"):
        raise ValueError("Label noise requires train_mask and num_global_classes.")

    is_graph_collection = (
        getattr(data, "task", None) == "graph_cls" and hasattr(data, "_data")
    )
    target = data._data if is_graph_collection else data
    labels = getattr(target, "y", None)
    train_mask = data.train_mask
    if labels is None:
        raise ValueError("Label noise requires labels in data.y.")
    if labels.shape[0] != train_mask.numel():
        raise ValueError(
            "Label-noise labels and train_mask must have the same leading dimension."
        )

    train_indices = train_mask.to(labels.device).nonzero(as_tuple=False).view(-1)
    num_selected = int(train_indices.numel() * rate)
    if num_selected == 0:
        return 0

    permutation = torch.randperm(train_indices.numel(), device=labels.device)
    selected = train_indices[permutation[:num_selected]]
    num_classes = int(data.num_global_classes)
    if labels.dim() == 1:
        if num_classes < 2:
            raise ValueError("Single-label noise requires at least two classes.")
        noisy_labels = labels.clone()
        noisy_labels[selected] = torch.randint(
            0, num_classes, (num_selected,), device=labels.device
        ).to(labels.dtype)
        target.y = noisy_labels
        return num_selected

    # The supported multi-task graph datasets are binary. Preserve missing
    # targets such as NaNs in ogbg-molpcba instead of overwriting them.
    noisy_labels = labels.clone()
    for task_index in range(noisy_labels.shape[1]):
        column = noisy_labels[selected, task_index]
        if column.dtype.is_floating_point:
            valid = ~torch.isnan(column)
        else:
            valid = torch.ones_like(column, dtype=torch.bool)
        valid_count = int(valid.sum().item())
        if valid_count == 0:
            continue
        column[valid] = torch.randint(
            0, 2, (valid_count,), device=labels.device
        ).to(labels.dtype)
        noisy_labels[selected, task_index] = column
    target.y = noisy_labels
    return num_selected
