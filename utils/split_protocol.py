"""Deterministic dataset splits defined by the Gefra experimental protocol."""

import hashlib

import torch


PAPER_NODE_SPLITS = {
    "cora": (0.2, 0.4, 0.4),
    "citeseer": (0.2, 0.4, 0.4),
    "pubmed": (0.6, 0.2, 0.2),
    "photo": (0.2, 0.4, 0.4),
    "computers": (0.2, 0.4, 0.4),
    "wikics": (0.8, 0.1, 0.1),
    "arxiv": (0.8, 0.1, 0.1),
    "reddit": (0.8, 0.1, 0.1),
    "disease": (0.8, 0.1, 0.1),
    "airport": (0.8, 0.1, 0.1),
}


def dataset_seed(seed, dataset_name, client_id=None):
    """Derive a stable per-dataset (and optionally per-client) seed."""
    suffix = "" if client_id is None else f":{client_id}"
    digest = hashlib.sha256(
        f"{seed}:{dataset_name.lower()}{suffix}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2 ** 31)


def make_paper_node_split(labels, dataset_name, seed):
    """Create the protocol's class-stratified node masks deterministically."""
    name = dataset_name.lower()
    if name not in PAPER_NODE_SPLITS:
        raise ValueError(f"No paper node split is defined for {dataset_name!r}")

    train_ratio, val_ratio, _ = PAPER_NODE_SPLITS[name]
    labels_cpu = labels.detach().view(-1).cpu()
    num_nodes = labels_cpu.numel()
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    generator = torch.Generator().manual_seed(dataset_seed(seed, name))

    for label in torch.unique(labels_cpu, sorted=True):
        class_indices = torch.nonzero(labels_cpu == label, as_tuple=False).view(-1)
        class_indices = class_indices[torch.randperm(class_indices.numel(), generator=generator)]
        train_end = int(train_ratio * class_indices.numel())
        val_end = int((train_ratio + val_ratio) * class_indices.numel())
        train_mask[class_indices[:train_end]] = True
        val_mask[class_indices[train_end:val_end]] = True
        test_mask[class_indices[val_end:]] = True

    return (
        train_mask.to(labels.device),
        val_mask.to(labels.device),
        test_mask.to(labels.device),
    )
