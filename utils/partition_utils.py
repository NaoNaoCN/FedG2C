import json
import os

import numpy as np
import torch
from torch_geometric.data import Data

from utils.split_protocol import PAPER_NODE_SPLITS, dataset_seed


PARTITION_PROTOCOL_VERSION = "paper-splits-v1"


def extract_floats(s):
    """
    Extracts and converts three floats separated by hyphens from a string and ensures their sum is 1.

    Args:
        s (str): A string containing three float numbers separated by hyphens (e.g., "0.6-0.3-0.1").

    Returns:
        tuple: A tuple of three floats (train, val, test) extracted from the string.

    Raises:
        AssertionError: If the sum of the three numbers does not equal 1.
    """
    from decimal import Decimal
    parts = s.split('-')
    train = float(parts[0])
    val = float(parts[1])
    test = float(parts[2])
    assert Decimal(parts[0]) + Decimal(parts[1]) + Decimal(parts[2]) == Decimal(1)
    return train, val, test

def idx_to_mask_tensor(idx_list, length):
    """
    Converts a list of indices to a tensor mask of a specified length.

    Args:
        idx_list (list[int]): List of indices that should be marked as 1 in the mask.
        length (int): Total length of the mask tensor.

    Returns:
        torch.Tensor: A binary mask tensor where positions corresponding to indices in idx_list are set to 1.
    """
    mask = torch.zeros(length)
    mask[idx_list] = 1
    return mask



def mask_tensor_to_idx(tensor):
    """
    Converts a tensor mask to a list of indices where the tensor is non-zero.

    Args:
        tensor (torch.Tensor): A tensor containing binary values.

    Returns:
        list[int]: A list of indices corresponding to non-zero entries in the tensor.
    """
    result = tensor.nonzero().squeeze().tolist()
    if type(result) is not list:
        result = [result]
    return result


def local_subgraph_train_val_test_split_node(local_subgraph, split, num_classes, shuffle=True, seed=None):
    num_nodes = local_subgraph.x.shape[0]
    train_, val_, test_ = extract_floats(split)

    device = local_subgraph.x.device

    # Allocate masks once on the same device as the local subgraph.
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

    # Flatten labels for class-wise splitting.
    y = local_subgraph.y.squeeze()

    for class_i in range(num_classes):
        class_i_node_indices = torch.nonzero(y == class_i).squeeze(-1)
        num_class_i_nodes = class_i_node_indices.shape[0]

        if num_class_i_nodes == 0:
            continue

        # Generate on CPU so the same seed yields the same split across devices.
        if shuffle:
            generator = torch.Generator()
            if seed is not None:
                generator.manual_seed(seed + class_i)
            perm = torch.randperm(num_class_i_nodes, generator=generator).to(device)
            class_i_node_indices = class_i_node_indices[perm]

        train_end = int(train_ * num_class_i_nodes)
        val_end = int((train_ + val_) * num_class_i_nodes)

        train_mask[class_i_node_indices[:train_end]] = True
        val_mask[class_i_node_indices[train_end:val_end]] = True
        test_mask[class_i_node_indices[val_end:]] = True

    return train_mask, val_mask, test_mask


def local_subgraph_train_val_test_split_link(
    local_subgraph,
    split,
    num_classes,
    shuffle=True,
    seed=None,
):
    num_edges = local_subgraph.edge_index.shape[1]


    train_, val_, test_ = extract_floats(split)

    train_mask = idx_to_mask_tensor([], num_edges)
    val_mask = idx_to_mask_tensor([], num_edges)
    test_mask = idx_to_mask_tensor([], num_edges)


    for class_i in range(num_classes):
        class_i_edge_mask = local_subgraph.y == class_i
        num_class_i_edges = class_i_edge_mask.sum()
        # A relation class can be absent from an individual client partition.
        if num_class_i_edges == 0:
            continue

        class_i_node_list = mask_tensor_to_idx(class_i_edge_mask)
        if len(class_i_node_list) == 0:
            continue
        if shuffle:
            if seed is None:
                np.random.shuffle(class_i_node_list)
            else:
                rng = np.random.default_rng(seed + class_i)
                rng.shuffle(class_i_node_list)
        train_mask += idx_to_mask_tensor(class_i_node_list[:int(train_ * num_class_i_edges)], num_edges)
        val_mask += idx_to_mask_tensor(class_i_node_list[int(train_ * num_class_i_edges) : int((train_+val_) * num_class_i_edges)], num_edges)
        test_mask += idx_to_mask_tensor(class_i_node_list[int((train_+val_) * num_class_i_edges): min(num_class_i_edges, int((train_+val_+test_) * num_class_i_edges))], num_edges)


    train_mask = train_mask.bool()
    val_mask = val_mask.bool()
    test_mask = test_mask.bool()
    return train_mask, val_mask, test_mask

# Single-graph partitioning.
def get_subgraph_pyg_data_node(global_dataset, node_list, split_seed=None):
    num_classes = torch.unique(global_dataset.y).shape[0]
    global_edge_index = global_dataset.edge_index
    device = global_edge_index.device

    local_id_to_global_id = {local_id: global_id for local_id, global_id in enumerate(node_list)}

    # Vectorized edge-index remapping.
    max_global_id = global_edge_index.max().item()
    lookup_table = torch.full((max_global_id + 1,), -1, dtype=torch.long, device=device)
    valid_global_ids = torch.tensor(node_list, dtype=torch.long, device=device)
    lookup_table[valid_global_ids] = torch.arange(len(node_list), dtype=torch.long, device=device)

    local_edge_index_all = lookup_table[global_edge_index]
    valid_edge_mask = (local_edge_index_all[0] != -1) & (local_edge_index_all[1] != -1)
    local_edge_index = local_edge_index_all[:, valid_edge_mask]
    global_edge_map = torch.nonzero(valid_edge_mask).squeeze(-1)

    local_x = global_dataset.x[node_list]
    local_y = global_dataset.y[node_list]

    # Preserve edge attributes when the source graph provides them.
    local_edge_attr = None
    if global_dataset.edge_attr is not None:
        local_edge_attr = global_dataset.edge_attr[global_edge_map]

    local_subgraph = Data(x=local_x, edge_index=local_edge_index,
                          edge_attr=local_edge_attr, y=local_y)
    local_subgraph.global_map = local_id_to_global_id
    local_subgraph.name = global_dataset.name
    if hasattr(global_dataset, 'weight'):
        local_subgraph.weight = global_dataset.weight
    local_subgraph.num_global_classes = num_classes
    local_subgraph.task = getattr(global_dataset, 'task', None)
    local_subgraph.num_tasks = getattr(global_dataset, 'num_tasks', None)

    print(f"Constructed local subgraph for {local_subgraph.name}, "
          f"nodes: {local_x.shape[0]}, edges: {local_edge_index.shape[1]}")

    name = local_subgraph.name.lower()
    split_ratios = PAPER_NODE_SPLITS.get(name, (0.6, 0.2, 0.2))
    split = "-".join(str(value) for value in split_ratios)
    local_subgraph.train_mask, local_subgraph.val_mask, local_subgraph.test_mask = \
        local_subgraph_train_val_test_split_node(
            local_subgraph, split, num_classes=num_classes, seed=split_seed,
        )

    print(f"Assigned train/val/test masks for {local_subgraph.name}.")
    return local_subgraph


def get_subgraph_pyg_data_link(global_dataset, node_list, split_seed=None):
    global_edge_index = global_dataset.edge_index
    device = global_edge_index.device

    local_id_to_global_id = {local_id: global_id for local_id, global_id in enumerate(node_list)}

    # Vectorized edge-index remapping.
    max_global_id = global_edge_index.max().item()
    lookup_table = torch.full((max_global_id + 1,), -1, dtype=torch.long, device=device)
    valid_global_ids = torch.tensor(node_list, dtype=torch.long, device=device)
    lookup_table[valid_global_ids] = torch.arange(len(node_list), dtype=torch.long, device=device)

    local_edge_index_all = lookup_table[global_edge_index]
    valid_edge_mask = (local_edge_index_all[0] != -1) & (local_edge_index_all[1] != -1)
    local_edge_index = local_edge_index_all[:, valid_edge_mask]
    global_edge_map = torch.nonzero(valid_edge_mask).squeeze(-1)

    local_x = global_dataset.x[node_list]

    # Preserve edge attributes when the source graph provides them.
    local_edge_attr = None
    if global_dataset.edge_attr is not None:
        local_edge_attr = global_dataset.edge_attr[global_edge_map]

    # Link-prediction labels are defined per edge.
    if hasattr(global_dataset, 'edge_type'):
        local_y = global_dataset.edge_type[global_edge_map]
    elif global_dataset.y is not None and global_dataset.y.shape[0] == global_edge_index.shape[1]:
        local_y = global_dataset.y[global_edge_map]
    else:
        local_y = torch.zeros(local_edge_index.shape[1], dtype=torch.long)

    num_classes = int(local_y.max().item()) + 1

    local_subgraph = Data(x=local_x, edge_index=local_edge_index,
                          edge_attr=local_edge_attr, y=local_y)
    local_subgraph.global_map = local_id_to_global_id
    local_subgraph.name = global_dataset.name
    if hasattr(global_dataset, 'weight'):
        local_subgraph.weight = global_dataset.weight
    local_subgraph.num_global_classes = num_classes
    local_subgraph.task = getattr(global_dataset, 'task', None)
    local_subgraph.num_tasks = getattr(global_dataset, 'num_tasks', None)

    local_subgraph.train_mask, local_subgraph.val_mask, local_subgraph.test_mask = \
        local_subgraph_train_val_test_split_link(
            local_subgraph,
            "0.8-0.1-0.1",
            num_classes=num_classes,
            seed=split_seed,
        )

    return local_subgraph





def _partition_metadata_matches(path, expected):
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle) == expected
    except (OSError, json.JSONDecodeError):
        return False


def single_graph_partition(data_tag, num_partitions, task="node_cls", root=None, seed=2025):
    dataset_dir = os.path.join(root, f"{data_tag.name.lower()}", f"tot_clients_{num_partitions}")
    os.makedirs(dataset_dir, exist_ok=True)
    filename_list = [os.path.join(dataset_dir, f"client_{i}.pt") for i in range(num_partitions)]
    metadata_path = os.path.join(dataset_dir, "partition_metadata.json")
    node_split = PAPER_NODE_SPLITS.get(data_tag.name.lower())
    metadata = {
        "protocol_version": PARTITION_PROTOCOL_VERSION,
        "dataset": data_tag.name.lower(),
        "num_partitions": num_partitions,
        "task": task,
        "seed": seed,
        # JSON loads arrays as lists, so do not store a tuple here.
        "node_split": list(node_split) if node_split is not None else None,
    }

    all_partition_found = True
    for filename in filename_list:
        if not os.path.exists(filename):
            all_partition_found = False
            break
    if not _partition_metadata_matches(metadata_path, metadata):
        all_partition_found = False
    local_data = []

    # METIS minimizes edge cuts while keeping partitions roughly balanced.
    if not all_partition_found:
        # The single-client protocol needs no graph partitioning backend.
        if num_partitions == 1:
            client_indices = [list(range(data_tag.num_nodes))]
        else:
            # METIS is only needed for explicit multi-partition experiments.
            # Keep it optional so the paper's one-client-per-dataset protocol
            # does not require a native extension at import time.
            try:
                import pymetis as metis
            except ImportError as exc:
                raise ImportError(
                    "pymetis is required when num_partitions > 1; "
                    "install it with `python -m pip install pymetis`."
                ) from exc
            from torch_geometric.utils import to_networkx

            print("Conducting subgraph-fl metis simulation...")
            graph_nx = to_networkx(data_tag, to_undirected=True)
            n_cuts, membership = metis.part_graph(num_partitions, graph_nx)
            client_indices = [None] * num_partitions
            for client_id in range(num_partitions):
                client_indices[client_id] = np.where(np.array(membership) == client_id)[0].tolist()



        for client_id, filename in enumerate(filename_list):
            if task == "node_cls":
                if num_partitions == 1:
                    # A paper client is one whole dataset. Preserve the loader's
                    # deterministic dataset-level split instead of resampling it.
                    local_subgraph = data_tag.clone()
                else:
                    local_subgraph = get_subgraph_pyg_data_node(
                        data_tag,
                        client_indices[client_id],
                    split_seed=dataset_seed(seed, data_tag.name, client_id),
                    )
            elif task == "link_pre":
                local_subgraph = get_subgraph_pyg_data_link(
                    data_tag,
                    client_indices[client_id],
                    split_seed=dataset_seed(seed, data_tag.name, client_id),
                )

            torch.save(local_subgraph, filename)

        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)


    for filename in filename_list:
        local_data.append(torch.load(filename, weights_only=False))


    return local_data


def merge_raw(raw):
    print("Length of raw data:", len(raw))
    merge_x = torch.hstack([i.x for i in raw])
    merge_y = torch.hstack([i.y for i in raw])
    merge_xe = torch.hstack([i.xe for i in raw])
    node_graph_id_map = []
    edge_graph_id_map = []

    ptr = 0
    edge_index_list = []
    pretrain_edge_index_list = []


    for id, i in enumerate(raw):
        edge_index_list.append(i.edge_index + ptr)
        pretrain_edge_index_list.append(i.pretrain_edge_index + ptr)
        ptr += i.x.shape[0]
        node_graph_id_map += [id] * i.x.shape[0]
        edge_graph_id_map += [id] * i.edge_index.shape[1]

    merge_edge_index = torch.hstack(edge_index_list)
    merge_pretrain_edge_index = torch.hstack(pretrain_edge_index_list)



    data = Data(x=merge_x, y=merge_y, edge_index=merge_edge_index)
    data.name = raw.name
    data.num_tasks = raw.num_tasks
    data.xe = merge_xe
    data.pretrain_edge_index = merge_pretrain_edge_index
    data.node_graph_id_map = torch.tensor(node_graph_id_map)
    data.edge_graph_id_map = torch.tensor(edge_graph_id_map)
    data.node_text_feat = raw.data.node_embs
    data.edge_text_feat = raw.data.edge_embs


    return data




# Graph-set partitioning.
def graph_set_partition(data_tag, num_partitions, root=None, mode="pretrain", seed=2025):
    dataset_dir = os.path.join(root, f"{data_tag.name.lower()}", f"tot_clients_{num_partitions}")
    os.makedirs(dataset_dir, exist_ok=True)
    pt_filename_list = [os.path.join(dataset_dir, f"client_{i}_pt.pt") for i in range(num_partitions)]
    ft_filename_list = [os.path.join(dataset_dir, f"client_{i}_ft.pt") for i in range(num_partitions)]
    dataset_name = data_tag.name.lower()
    if dataset_name == "chemblpre":
        split = "1.0-0-0"
    elif dataset_name in {"chemhiv", "chempcba", "proteins", "imdb-binary"}:
        split = "0.8-0.1-0.1"
    else:
        raise ValueError(f"Unsupported graph-set dataset: {data_tag.name}")
    metadata_path = os.path.join(dataset_dir, "partition_metadata.json")
    metadata = {
        "protocol_version": PARTITION_PROTOCOL_VERSION,
        "dataset": dataset_name,
        "num_partitions": num_partitions,
        "task": "graph_cls",
        "seed": seed,
        "split": list(extract_floats(split)),
    }

    all_partition_found = True
    for filename in pt_filename_list:
        if not os.path.exists(filename):
            all_partition_found = False
            break

    for filename in ft_filename_list:
        if not os.path.exists(filename) and data_tag.name.lower() not in ["chemblpre"]:
            all_partition_found = False
            break

    if not _partition_metadata_matches(metadata_path, metadata):
        all_partition_found = False


    local_data = []


    if not all_partition_found:
        num_tot_graphs = len(data_tag)
        num_graphs_per_partition = int(num_tot_graphs / num_partitions)

        train_, val_, test_ = extract_floats(split)


        for partition_i, pt_filename in enumerate(pt_filename_list):
            print(f"processing {data_tag.name.lower()} client {partition_i}...")
            start_idx = partition_i * num_graphs_per_partition
            end_idx = (partition_i + 1) * num_graphs_per_partition if partition_i != num_partitions - 1 else num_tot_graphs


            data = data_tag[start_idx:end_idx]

            # Derive deterministic local train, validation, and test masks.
            num_graphs = end_idx-start_idx
            graph_list = np.arange(num_graphs)
            rng = np.random.default_rng(dataset_seed(seed, dataset_name, partition_i))
            rng.shuffle(graph_list)
            train_mask = idx_to_mask_tensor(graph_list[:int(train_ * num_graphs)], num_graphs).bool()
            val_mask = idx_to_mask_tensor(graph_list[int(train_ * num_graphs) : int((train_+val_) * num_graphs)], num_graphs).bool()
            test_mask = idx_to_mask_tensor(graph_list[int((train_+val_) * num_graphs): min(num_graphs, int((train_+val_+test_) * num_graphs))], num_graphs).bool()
            # Store the fine-tuning subset when the dataset has labels.
            if data_tag.name.lower() not in ["chemblpre"]:
                data.name = data_tag.name.lower()
                data.weight = data_tag.weight
                data.train_mask = train_mask
                data.val_mask = val_mask
                data.test_mask = test_mask
                torch.save(data, ft_filename_list[partition_i])

            # Preserve individual graphs for batched pre-training.
            data.name = data_tag.name.lower()
            data.num_tasks = data_tag.num_tasks
            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = test_mask
            torch.save(data, pt_filename)

        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)


    if mode == "pretrain":
        for filename in pt_filename_list:
            local_data.append(torch.load(filename, weights_only=False))
    else:
        for filename in ft_filename_list:
            local_data.append(torch.load(filename, weights_only=False))


    return local_data
