import json
import torch
import os.path as osp
import random
import numpy as np
from pathlib import Path
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader, DataLoader
from torchmetrics import Accuracy
from sklearn.metrics import roc_auc_score
from utils.split_protocol import make_paper_node_split

def get_k_shot(data, k=-1):
    """Apply k-shot sampling to node or graph classification training masks.

    ``k=-1`` preserves the full training split, ``k=0`` enables zero-shot
    evaluation, and positive values retain up to ``k`` examples per class.
    Relation-classification datasets retain their original edge masks.
    """
    if k == -1:
        return data

    # Preserve validation and test masks while clearing supervised examples.
    if k == 0:
        data.train_mask = torch.zeros_like(data.train_mask, dtype=torch.bool)
        print(f"[Zero-shot] Dataset: {data.name if hasattr(data, 'name') else 'Unknown'}, "
              f"train_mask cleared (all False)")
        return data

    # Relation classification uses edge masks and does not support k-shot sampling.
    if hasattr(data, 'task') and 'link' in str(data.task).lower():
        return data

    if hasattr(data, 'name') and data.name.lower() in ['wn18rr', 'fb15k237']:
        return data

    # Graph-level masks index individual graphs in a dataset slice.
    if hasattr(data, 'name') and data.name.lower() in ['chemhiv', 'chempcba', 'chemblpre', 'proteins', 'imdb-binary']:
        train_indices = data.train_mask.nonzero(as_tuple=True)[0].tolist()

        if hasattr(data, 'y'):
            train_labels = data.y[data.train_mask]
        else:
            train_labels = torch.tensor([data[i].y for i in train_indices])

        if train_labels.dim() > 1:
            num_tasks = train_labels.shape[1]
            selected_indices_set = set()

            for task_idx in range(num_tasks):
                task_labels = train_labels[:, task_idx]

                for class_label in [0, 1]:
                    class_mask = (task_labels == class_label)
                    class_indices = [train_indices[i] for i, m in enumerate(class_mask) if m]

                    if k == 0:
                        selected = []
                    else:
                        selected = class_indices[:min(k, len(class_indices))]
                    selected_indices_set.update(selected)

            new_train_mask = torch.zeros_like(data.train_mask)
            for idx in selected_indices_set:
                new_train_mask[idx] = True

        else:
            unique_labels = train_labels.unique().tolist()
            new_train_mask = torch.zeros_like(data.train_mask)

            for label in unique_labels:
                label_mask = (train_labels == label)
                cls_train_idx = [train_indices[i] for i, m in enumerate(label_mask) if m]

                if k == 0:
                    selected = []
                else:
                    selected = cls_train_idx[:min(k, len(cls_train_idx))]

                for idx in selected:
                    new_train_mask[idx] = True

        data.train_mask = new_train_mask
        print(f"[K-shot for Graph Classification] Dataset: {data.name}, k={k}, "
              f"Original training samples: {len(train_indices)}, "
              f"After k-shot: {new_train_mask.sum().item()}")
        return data

    # Node labels and masks must share their leading dimension.
    if data.y.shape[0] != data.train_mask.shape[0]:
        return data

    new_train_mask = torch.zeros_like(data.train_mask)
    train_labels = data.y[data.train_mask]
    unique_labels = train_labels.unique().tolist()

    for label in unique_labels:
        cls_train_idx = ((data.y == label) & data.train_mask).nonzero(as_tuple=True)[0].tolist()
        if k == 0:
            selected = []
        else:
            selected = cls_train_idx[:k]
        new_train_mask[selected] = True

    data.train_mask = new_train_mask
    print(f"[K-shot for Node Classification] Dataset: {data.name if hasattr(data, 'name') else 'Unknown'}, k={k}, "
          f"Selected samples: {new_train_mask.sum().item()}")
    return data



metric2order = {'loss': 'min', 'acc': 'max', 'f1': 'max', 'precision': 'max', 'recall': 'max', 'auc': 'max',
                'ap': 'max', 'mcc': 'max', 'hit': 'max', 'ndcg': 'max', 'map': 'max', 'mrr': 'max'}

class EarlyStopping:
    def __init__(self, patience=50):
        self.patience = patience
        self.counter = 0
        self.best_val = -np.inf
        self.best_dict = None
        self.early_stop = False

    def __call__(self, result):
        if result['val'] > self.best_val:
            self.best_val = result['val']
            self.best_dict = result
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

class Logger:
    def __init__(self):
        self.data = {}
        self.best = {}

    def check_result(self, result):
        if 'metric' not in result:
            raise ValueError('Result must contain metric key')
        if result['metric'] not in metric2order:
            raise ValueError('Metric not supported')
        if result['train'] is None:
            result['train'] = 0
        if result['val'] is None:
            result['val'] = 0

        return result

    def log(self, run, epoch, loss, result, **kwargs):
        result = self.check_result(result)

        train_value = result['train']
        val_value = result['val']
        test_value = result['test']

        if run not in self.data:
            self.data[run] = {'train': [], 'val': [], 'test': [], 'loss': [], 'epochs': []}

        if isinstance(loss, dict):
            loss_value = loss.get('loss', 0.0)
        else:
            loss_value = float(loss) if loss is not None else 0.0

        self.data[run]['loss_train'] = loss_value
        self.data[run]['train'].append(train_value)
        self.data[run]['val'].append(val_value)
        self.data[run]['test'].append(test_value)
        self.data[run]['loss'].append(loss_value)
        self.data[run]['epochs'].append(epoch)
        self.data[run]['epoch'] = epoch

        if run not in self.best:
            self.best[run] = {'train': None, 'val': None, 'test': None}

        if metric2order[result['metric']] == 'max':
            if self.best[run]['val'] is None or val_value >= self.best[run]['val']:
                self.best[run]['train'] = train_value
                self.best[run]['val'] = val_value
                self.best[run]['test'] = test_value
                self.best[run]['epoch'] = epoch
        else:
            if self.best[run]['val'] is None or val_value <= self.best[run]['val']:
                self.best[run]['train'] = train_value
                self.best[run]['val'] = val_value
                self.best[run]['test'] = test_value
                self.best[run]['epoch'] = epoch

    def set_run_efficiency(self, run, stop_epoch, runtime_sec, peak_gpu_mb):
        """Record stopping point, elapsed time, and peak GPU memory for one run."""
        if run not in self.best:
            self.best[run] = {}
        self.best[run]['stop_epoch'] = stop_epoch
        self.best[run]['runtime_sec'] = runtime_sec
        self.best[run]['peak_gpu_mb'] = peak_gpu_mb

    def get_run_raw(self):
        return self.data

    def get_best_raw(self):
        return self.best

    def get_single_run(self, run_idx):
        return self.data[run_idx]

    def get_single_best(self, run_idx):
        return self.best[run_idx]

    def get_run(self):
        train = np.mean([np.mean(self.data[run_idx]['train']) for run_idx in self.data])
        val = np.mean([np.mean(self.data[run_idx]['val']) for run_idx in self.data])
        test = np.mean([np.mean(self.data[run_idx]['test']) for run_idx in self.data])
        return {'train': train, 'val': val, 'test': test}

    def get_best(self):
        train = [self.best[run_idx]['train'] for run_idx in self.best]
        val = [self.best[run_idx]['val'] for run_idx in self.best]
        test = [self.best[run_idx]['test'] for run_idx in self.best]

        return {'train': {'mean': np.mean(train), 'std': np.std(train)},
                'val': {'mean': np.mean(val), 'std': np.std(val)},
                'test': {'mean': np.mean(test), 'std': np.std(test)}}

def get_loader(data, split, labels, task, batch_size):
    task = task
    setting = "standard"

    if task == "node_cls":
        if setting in ['zero_shot', 'in_context']:
            train_loader = None
        else:
            train_loader = NeighborLoader(
                data,
                num_neighbors=[10] * 2,
                input_nodes=mask2idx(split["train"]),
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                persistent_workers=True,
            )
        eval_nodes = torch.cat([mask2idx(split["valid"]), mask2idx(split["test"])])
        subgraph_loader = NeighborLoader(
            data,
            num_neighbors=[10] * 2,
            input_nodes=eval_nodes,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
        )
        return train_loader, subgraph_loader

    elif task == "link_pre":
        if setting in ['zero_shot', 'in_context']:
            train_loader = None
        else:
            train_loader = LinkNeighborLoader(
                data,
                num_neighbors=[30] * 2,
                edge_label_index=data.edge_index[:, split["train"]],
                edge_label=labels[split["train"]],
                batch_size=batch_size,
                shuffle=True,
            )
        subgraph_loader = LinkNeighborLoader(
            data,
            num_neighbors=[-1] * 2,
            edge_label_index=data.edge_index,
            edge_label=labels,
            batch_size=4096,
            shuffle=False,
        )
        return train_loader, subgraph_loader

    elif task == "graph_cls":
        if setting == 'standard':
            train_dataset = data[split["train"]]
            val_dataset = data[split["valid"]]
            test_dataset = data[split["test"]]

            # Indexing applies dataset transforms such as ``OneHotDegree``;
            # the lower-level ``get`` method bypasses them.
            def precompute_graph_list(dataset):
                return [dataset[i] for i in range(len(dataset))]

            train_list = precompute_graph_list(train_dataset)
            val_list   = precompute_graph_list(val_dataset)
            test_list  = precompute_graph_list(test_dataset)

            train_loader = DataLoader(
                train_list,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
            )
            val_loader = DataLoader(
                val_list,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
            )
            test_loader = DataLoader(
                test_list,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
            )
        elif setting in ['few_shot']:
            # As we only update the train_idx in sampling few-shot samples,
            # we can directly use the split["train"] as the train_idx
            # This enables the shuffle function in DataLoader.
            # The drawback is we should define the proto_loader in the finetune_graph_task function
            train_dataset = data[split["train"]]

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
            )
            val_loader = None
            test_loader = None

        elif setting in ['zero_shot', 'in_context']:
            train_loader = None
            val_loader = None
            test_loader = None

        return train_loader, val_loader, test_loader



def seed_everything(seed):
    """
    Sets the seed for multiple random number generators to ensure reproducibility across runs.
    It also configures the behavior of the CUDA backend for deterministic output.

    Args:
        seed (int): The seed number to use for seeding the random number generators.

    Details:
        - Sets the seed for Python's built-in `random` module, NumPy's random module, and PyTorch.
        - Configures PyTorch's CUDA-related seeds for all GPUs.
        - Sets CUDA's cuDNN backend to operate deterministically, which can impact performance
          due to the disabling of certain optimizations like `benchmark` and general `enabled` status.

    Note:
        Enabling determinism can lead to a performance trade-off but is necessary for reproducibility
        when exact outcomes are critical to maintain across different runs, especially during debugging
        or testing phases.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# PyG/OGB-native dataset loading. Node features are provided directly in
# ``data.x`` and edge attributes may be absent.
def _load_kg_via_pyg_random(dataset_name, args, data_root):
    """Load relation graphs through PyG with deterministic random node features."""
    from torch_geometric.datasets import WordNet18RR, RelLinkPredDataset
    from torch_geometric.data import Data as _PygData

    if dataset_name.lower() == "wn18rr":
        raw = WordNet18RR(root=osp.join(data_root, "wn18rr"))
    else:
        raw = RelLinkPredDataset(name="FB15k-237",
                                 root=osp.join(data_root, "fb15k237"))
    data = raw[0]

    # PyG exposes node IDs as integer features. Replace them with deterministic
    # floating-point features compatible with the graph encoders.
    in_dim = getattr(args, "hidden_dim", 128)
    num_nodes = int(data.num_nodes) if data.num_nodes is not None else int(data.edge_index.max().item()) + 1
    x_seed = getattr(args, "kg_feat_seed", 42)
    x_gen = torch.Generator().manual_seed(x_seed)
    data.x = torch.randn(num_nodes, in_dim, generator=x_gen) * 0.1
    data.num_nodes = num_nodes
    data.in_dim = in_dim
    data.edge_attr = None

    assert hasattr(data, "edge_type"), f"{dataset_name} is missing edge_type labels."

    # Keep only the forward triples; PyG appends inverse relation edges.
    if dataset_name.lower() == "fb15k237":
        num_relations_half = (int(data.edge_type.max().item()) + 1) // 2
        forward_mask = data.edge_type < num_relations_half
        data.edge_index = data.edge_index[:, forward_mask]
        data.edge_type = data.edge_type[forward_mask]

    data.y = data.edge_type.clone().long()
    num_relations = int(data.edge_type.max().item()) + 1

    # Initialize an 80/10/10 edge split; partitioning rebuilds it as needed.
    num_edges = data.edge_index.size(1)
    perm = torch.randperm(num_edges)
    t = int(0.8 * num_edges); v = int(0.9 * num_edges)
    train_mask = torch.zeros(num_edges, dtype=torch.bool)
    val_mask   = torch.zeros(num_edges, dtype=torch.bool)
    test_mask  = torch.zeros(num_edges, dtype=torch.bool)
    train_mask[perm[:t]] = True
    val_mask[perm[t:v]]  = True
    test_mask[perm[v:]]  = True
    data.train_mask = train_mask
    data.val_mask   = val_mask
    data.test_mask  = test_mask
    data.name = dataset_name
    data.task = "link_pre"
    data.num_tasks = 1
    data.num_global_classes = num_relations
    return data


def _load_arxiv_via_ogb(args, data_root):
    """Load OGB arXiv features with the paper-defined 80/10/10 node split."""
    from ogb.nodeproppred import PygNodePropPredDataset
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage, NodeStorage, EdgeStorage
    torch.serialization.add_safe_globals(
        [DataEdgeAttr, DataTensorAttr, GlobalStorage, NodeStorage, EdgeStorage])
    raw = PygNodePropPredDataset(name="ogbn-arxiv", root=osp.join(data_root, "arxiv"))
    data = raw[0]
    num_nodes = data.x.size(0)
    data.y = data.y.squeeze()
    data.train_mask, data.val_mask, data.test_mask = make_paper_node_split(
        data.y, "arxiv", getattr(args, "seed", 2025)
    )
    data.num_nodes = num_nodes
    data.in_dim = data.x.size(1)
    data.name = "arxiv"
    data.task = "node_cls"
    data.num_tasks = 1
    data.num_global_classes = int(data.y.max().item()) + 1
    return data


def load_fedgfm_datasets(args, mode="pretrain"):
    from torch_geometric.datasets import Planetoid, Amazon, WikiCS, Coauthor
    from torch_geometric.datasets import TUDataset, MoleculeNet
    from torch_geometric.transforms import NormalizeFeatures
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage, NodeStorage, EdgeStorage
    torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage, NodeStorage, EdgeStorage])
    with open(args.data_config, 'r', encoding='utf-8') as file:
        dataset_config = json.load(file)

    from utils.config_utils import select_dataset_config
    num_clients_each_dataset, datasets, weight_list = select_dataset_config(
        dataset_config, mode
    )

    data_root = osp.join(args.project_path, "cache_data")
    dataset_list = []

    for dataset_name in datasets:
        name_lower = dataset_name.lower()
        print(f"Loading raw data: {name_lower}")
        # Node classification datasets.
        if name_lower in ["cora", "pubmed", "citeseer"]:
            raw = Planetoid(root=data_root, name=dataset_name.capitalize(),
                            transform=NormalizeFeatures())
            data = raw[0]
            num_nodes = data.x.size(0)
            num_classes = int(data.y.max().item()) + 1
            data.name = name_lower
            data.task = "node_cls"
            data.num_tasks = 1
            data.num_global_classes = num_classes

            # The paper protocol uses one deterministic class-stratified 20/40/40 split.
            data.train_mask, data.val_mask, data.test_mask = make_paper_node_split(
                data.y, name_lower, getattr(args, "seed", 2025)
            )
            # dataset_list.append(data)

        elif name_lower in ["photo", "computers"]:
            cls = Amazon
            raw_name = "Photo" if name_lower == "photo" else "Computers"
            raw = cls(root=data_root, name=raw_name, transform=NormalizeFeatures())
            data = raw[0]
            num_nodes = data.x.size(0)
            num_classes = int(data.y.max().item()) + 1
            # The paper protocol uses one deterministic class-stratified 20/40/40 split.
            data.train_mask, data.val_mask, data.test_mask = make_paper_node_split(
                data.y, name_lower, getattr(args, "seed", 2025)
            )
            data.name = name_lower
            data.task = "node_cls"
            data.num_tasks = 1
            data.num_global_classes = num_classes
            # dataset_list.append(data)

        elif name_lower == "wikics":
            raw = WikiCS(root=osp.join(data_root, "wikics"))
            data = raw[0]
            num_nodes = data.x.size(0)
            num_classes = int(data.y.max().item()) + 1
            # The paper protocol uses one deterministic class-stratified 80/10/10
            # split rather than WikiCS's 20 official train masks.
            data.train_mask, data.val_mask, data.test_mask = make_paper_node_split(
                data.y, name_lower, getattr(args, "seed", 2025)
            )
            data.name = "wikics"
            data.task = "node_cls"
            data.num_tasks = 1
            data.num_global_classes = num_classes
            # dataset_list.append(data)

        # OGB node classification.
        elif name_lower == "arxiv":
            data = _load_arxiv_via_ogb(args, data_root)
            # dataset_list.append(data)

        elif name_lower == "reddit":
            from torch_geometric.datasets import Reddit
            raw = Reddit(root=osp.join(data_root, "reddit"))
            data = raw[0]
            num_classes = int(data.y.max().item()) + 1
            data.train_mask, data.val_mask, data.test_mask = make_paper_node_split(
                data.y, name_lower, getattr(args, "seed", 2025)
            )
            data.name = "reddit"
            data.task = "node_cls"
            data.num_tasks = 1
            data.num_global_classes = num_classes
            # dataset_list.append(data)

        # Domain-specific node classification datasets.
        elif name_lower in ["disease", "airport"]:
            import scipy.sparse as sp

            def _bin_feat(feat, bins):
                # Shift bin indices so the smallest class label is zero.
                digitized = np.digitize(feat, bins)
                return digitized - digitized.min()

            if name_lower == "airport":
                import pickle, networkx as nx
                root = osp.join(data_root, "airport")
                graph = pickle.load(open(osp.join(root, "airport.p"), "rb"))
                adj = nx.adjacency_matrix(graph)
                feats = np.array([graph.nodes[u]["feat"] for u in graph.nodes()])
                # The fifth column is the label; retain the first four features.
                label_idx = 4
                labels = feats[:, label_idx]
                feats = feats[:, :label_idx]
                labels = _bin_feat(labels, bins=[7.0 / 7, 8.0 / 7, 9.0 / 7])
            elif name_lower == "disease":
                ds = "disease_nc"
                root = osp.join(data_root, "disease_nc")
                obj2idx, idx_counter, edges = {}, 0, []
                with open(osp.join(root, f"{ds}.edges.csv"), "r") as f:
                    for line in f:
                        n1, n2 = line.rstrip().split(",")
                        for n in (n1, n2):
                            if n not in obj2idx:
                                obj2idx[n] = idx_counter; idx_counter += 1
                        edges.append((obj2idx[n1], obj2idx[n2]))
                N = idx_counter
                # Treat the source graph as undirected.
                row = [i for i, _ in edges] + [j for _, j in edges]
                col = [j for _, j in edges] + [i for i, _ in edges]
                adj = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(N, N))
                feats = sp.load_npz(osp.join(root, f"{ds}.feats.npz"))
                labels = np.load(osp.join(root, f"{ds}.labels.npy"))

            from torch_geometric.utils import from_scipy_sparse_matrix
            from torch_geometric.data import Data
            edge_index, _ = from_scipy_sparse_matrix(sp.csr_matrix(adj))
            x = torch.tensor(np.asarray(feats.todense()) if sp.issparse(feats) else np.asarray(feats),
                             dtype=torch.float)
            labels = np.asarray(labels)
            if labels.ndim > 1 and labels.shape[1] > 1:
                labels = labels.argmax(axis=1)
            labels = labels.reshape(-1)
            y = torch.tensor(labels, dtype=torch.long)

            data = Data(x=x, edge_index=edge_index, y=y)
            num_nodes = x.size(0)
            num_classes = int(y.max().item()) + 1

            data.train_mask, data.val_mask, data.test_mask = make_paper_node_split(
                data.y, name_lower, getattr(args, "seed", 2025)
            )
            data.name = name_lower
            data.task = "node_cls"
            data.num_tasks = 1
            data.num_global_classes = num_classes
            # dataset_list.append(data)

        # Relation classification datasets.
        elif dataset_name.lower() in ["wn18rr", "fb15k237"]:
            data = _load_kg_via_pyg_random(dataset_name, args, data_root)
            # dataset_list.append(data)

        # Graph classification datasets.
        elif name_lower in ["proteins", "imdb-binary"]:
            tu_name = "PROTEINS" if name_lower == "proteins" else "IMDB-BINARY"
            import torch_geometric.transforms as T
            # IMDB-BINARY has no node features, so encode node degree as one-hot.
            transform = T.OneHotDegree(max_degree=135) if name_lower == "imdb-binary" else None
            raw = TUDataset(root=osp.join(data_root), name=tu_name, transform=transform)
            num_graphs = len(raw)
            num_classes = raw.num_classes
            perm = torch.randperm(num_graphs)
            t = int(0.8 * num_graphs); v = int(0.9 * num_graphs)
            train_mask = torch.zeros(num_graphs, dtype=torch.bool)
            val_mask   = torch.zeros(num_graphs, dtype=torch.bool)
            test_mask  = torch.zeros(num_graphs, dtype=torch.bool)
            train_mask[perm[:t]] = True
            val_mask[perm[t:v]]  = True
            test_mask[perm[v:]]  = True
            raw.name = name_lower
            raw.task = "graph_cls"
            raw.num_tasks = 1
            raw.num_global_classes = num_classes
            raw.train_mask = train_mask
            raw.val_mask   = val_mask
            raw.test_mask  = test_mask
            raw.y = torch.cat([g.y for g in raw])
            data = raw
            # dataset_list.append(data)

        elif name_lower in ["chemhiv", "chempcba"]:
            from ogb.graphproppred import PygGraphPropPredDataset
            original_name = name_lower
            if name_lower == 'chempcba':
                ogb_name = 'ogbg-molpcba'
            else:
                ogb_name = 'ogbg-molhiv'
            raw = PygGraphPropPredDataset(name=ogb_name, root=osp.join(data_root))
            split_idx = raw.get_idx_split()
            num_graphs = len(raw)
            num_tasks = raw.num_tasks
            train_mask = torch.zeros(num_graphs, dtype=torch.bool)
            val_mask   = torch.zeros(num_graphs, dtype=torch.bool)
            test_mask  = torch.zeros(num_graphs, dtype=torch.bool)
            train_mask[split_idx["train"]] = True
            val_mask[split_idx["valid"]]   = True
            test_mask[split_idx["test"]]   = True
            raw.name = original_name
            raw.task = "graph_cls"
            raw.num_tasks = num_tasks
            raw.num_global_classes = num_tasks
            raw.train_mask = train_mask
            raw.val_mask   = val_mask
            raw.test_mask  = test_mask
            raw.y = raw.y
            data = raw
            # dataset_list.append(data)

        else:
            raise ValueError(f"Unknown dataset: {dataset_name}. Please check your configuration.")

        dataset_list.append(data)

    for idx, data_tag in enumerate(dataset_list):
        data_tag.weight = weight_list[idx]

    return num_clients_each_dataset, dataset_list


def pre_node(dataset):
    return dataset


def pre_link(dataset):
    return dataset


def pre_graph(dataset):
    return dataset

def get_preprocess(task):
    if task == 'node_cls':
        return pre_node
    elif task == 'link_pre':
        return pre_link
    elif task == 'graph_cls':
        return pre_graph
    else:
        raise NotImplementedError('The task is not implemented')

def check_path(path):
    if not osp.exists(path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
    return path

def mask2idx(mask):
    return torch.where(mask)[0]

def get_device_from_model(model):
    return next(model.parameters()).device


task2metric = {'node_cls': 'acc', 'link_pre': 'acc', 'graph_cls': 'auc'}
def evaluate(pred, y, task, mask=None, params=None):
    if mask is not None and mask.sum() == 0:
        return 0

    metric = task2metric[task]

    if metric == 'acc':
        return eval_acc(pred, y, mask) * 100
    elif metric == 'auc':
        return eval_auc(pred, y) * 100
    else:
        raise ValueError(f"Metric {metric} is not supported.")


def eval_acc(y_pred, y_true, mask):
    device = y_pred.device
    num_classes = y_pred.size(1)

    evaluator = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    if mask is not None:
        return evaluator(y_pred[mask], y_true[mask]).item()
    else:
        return evaluator(y_pred, y_true).item()


def eval_auc(y_pred, y_true):
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()

    # Promote one-dimensional predictions to a single-task matrix.
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # Align task dimensions before computing the metric.
    if y_pred.shape[1] != y_true.shape[1]:
        min_tasks = min(y_pred.shape[1], y_true.shape[1])
        y_pred = y_pred[:, :min_tasks]
        y_true = y_true[:, :min_tasks]

    roc_list = []
    y_true[y_true == -1] = 0
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_valid = y_true[:, i] == y_true[:, i]
            roc_list.append(roc_auc_score(y_true[is_valid, i], y_pred[is_valid, i]))

    if len(roc_list) == 0:
        # Use chance performance when no task has valid AUC labels.
        return 0.5

    return sum(roc_list) / len(roc_list)
