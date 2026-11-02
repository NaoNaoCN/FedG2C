# ===== FedG2C ADD BEGIN A/C: molecular graph manifests for M1 =====
"""Prepare the M1 molecular engineering protocol from trusted local caches.

Source clients:
  - 10,000 valid graphs sampled from the official OGB-MolPCBA train split;
  - all PROTEINS graphs;
  - all IMDB-BINARY graphs with normalized node degree as the only node feature.

Target:
  - 4,096/1,024/1,024 ChemHIV graphs selected inside the official scaffold
    train/valid/test partitions using fixed stratified sampling.

The source manifest intentionally contains no labels. Target labels use NaN for
missing OGB labels, never treating a missing label as a negative example.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import is_undirected

from .context import GraphView, validate_graph


SOURCE_PCBA_GRAPHS = 10_000
TARGET_COUNTS = {"train": 4_096, "valid": 1_024, "test": 1_024}
SOURCE_SHARD_SIZE = 256
PROTOCOL_VERSION = "fedg2c-molecular-chemhiv-scaffold-subset-v1"
INTEGER_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)


def molecular_schema():
    """Use OGB's published vocabulary, without fitting to validation/test graphs."""
    from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

    return {"input_dim": 9, "node_cardinalities": get_atom_feature_dims(),
            "edge_dim": 3, "edge_cardinalities": get_bond_feature_dims()}


def add_ogb_safe_globals():
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import EdgeStorage, GlobalStorage, NodeStorage

    torch.serialization.add_safe_globals(
        [DataEdgeAttr, DataTensorAttr, GlobalStorage, NodeStorage, EdgeStorage]
    )


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_degree(edge_index, num_nodes):
    degree = torch.bincount(edge_index[0], minlength=num_nodes).float()
    scale = degree.max().clamp_min(1.0)
    return (degree / scale).unsqueeze(1)


def graph_record(graph, graph_id, label=None, degree_features=False):
    x = graph.x
    if x is None:
        x = normalized_degree(graph.edge_index, int(graph.num_nodes)) if degree_features else None
    if x is None:
        raise ValueError(f"Graph {graph_id} has no node features and degree fallback is disabled")
    # PyG separates graphs using views of its complete dataset storage. A
    # contiguous view still shares that storage; torch.save would serialize the
    # entire backing allocation for every shard/file unless we clone here.
    x = x.detach().cpu().contiguous().clone()
    edge_index = graph.edge_index.detach().cpu().long().contiguous().clone()
    edge_attr = None if graph.edge_attr is None else graph.edge_attr.detach().cpu().contiguous().clone()
    original_node_id = torch.arange(int(graph.num_nodes), dtype=torch.long)
    record = {"x": x, "edge_index": edge_index, "original_node_id": original_node_id,
              "graph_id": str(graph_id)}
    if edge_attr is not None:
        record["edge_attr"] = edge_attr
    if label is not None:
        record["y"] = label.detach().cpu().reshape(-1).clone()
    validate_graph(GraphView(x, edge_index, edge_attr, original_node_id))
    if not is_undirected(edge_index, edge_attr=edge_attr, num_nodes=len(x)):
        raise ValueError(f"Expected paired reverse edges/attributes in {graph_id}")
    if x.dtype in INTEGER_DTYPES:
        if x.numel() and x.min() < 0:
            raise ValueError(f"Negative categorical node value in {graph_id}")
    if edge_attr is not None and edge_attr.dtype in INTEGER_DTYPES:
        if edge_attr.numel() and edge_attr.min() < 0:
            raise ValueError(f"Negative categorical edge value in {graph_id}")
    return record


class SchemaStats:
    def __init__(self, schema):
        self.feature_schema = schema
        self.graphs = 0
        self.min_nodes = None
        self.max_nodes = 0

    def update(self, record):
        x = record["x"]
        schema = self.feature_schema
        if x.shape[1] != schema["input_dim"]:
            raise ValueError("Node feature dimension differs from declared schema")
        for tensor, cardinalities in ((x, schema.get("node_cardinalities")),
                                      (record.get("edge_attr"), schema.get("edge_cardinalities"))):
            if cardinalities is not None:
                if tensor is None or tensor.dtype not in INTEGER_DTYPES or tensor.shape[1] != len(cardinalities):
                    raise ValueError("Categorical feature layout differs from OGB schema")
                if tensor.numel() and ((tensor < 0).any() or (tensor >= torch.tensor(cardinalities)).any()):
                    raise ValueError("Category ID exceeds the published OGB vocabulary")
        edge = record.get("edge_attr")
        if (edge is None) != (schema.get("edge_dim", 0) == 0):
            raise ValueError("Edge attributes differ from declared schema")
        if edge is not None and edge.shape[1] != schema["edge_dim"]:
            raise ValueError("Edge feature dimension differs from declared schema")
        self.graphs += 1
        self.min_nodes = int(x.shape[0]) if self.min_nodes is None else min(self.min_nodes, int(x.shape[0]))
        self.max_nodes = max(self.max_nodes, int(x.shape[0]))

    def schema(self):
        return dict(self.feature_schema)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_ogb(name, cache):
    add_ogb_safe_globals()
    from ogb.graphproppred import PygGraphPropPredDataset

    processed = cache / name.replace("-", "_") / "processed/geometric_data_processed.pt"
    if not processed.is_file():
        raise FileNotFoundError(f"Missing processed OGB cache: {processed}")
    return PygGraphPropPredDataset(name=name, root=str(cache))


def shuffled_valid_indices(dataset, indices, count, seed, prefix):
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(indices), generator=generator).tolist()
    selected, inspected = [], 0
    for position in order:
        inspected += 1
        index = int(indices[position])
        graph = dataset[index]
        if int(graph.num_nodes) < 2 or not (graph.edge_index[0] != graph.edge_index[1]).any():
            continue
        selected.append(index)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"{prefix}: only found {len(selected)} valid graphs after inspecting {inspected}")
    return selected, inspected


def stratified_indices(dataset, indices, count, seed, prefix):
    labels = []
    for index in indices:
        label = dataset[int(index)].y.reshape(-1)[0].item()
        labels.append(float(label))
    labels = torch.tensor(labels, dtype=torch.float32)
    observed = torch.isfinite(labels)
    if not bool(observed.all()) or not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError(f"{prefix}: binary target selection requires observed 0/1 labels")
    positive = torch.as_tensor(indices, dtype=torch.long)[labels == 1]
    negative = torch.as_tensor(indices, dtype=torch.long)[labels == 0]
    if count > len(positive) + len(negative):
        raise ValueError(f"{prefix}: requested {count} from {len(positive)+len(negative)} graphs")
    positive_count = round(count * len(positive) / (len(positive) + len(negative)))
    positive_count = min(max(positive_count, 1), len(positive))
    negative_count = count - positive_count
    if negative_count > len(negative):
        negative_count = len(negative)
        positive_count = count - negative_count
    if positive_count > len(positive):
        positive_count = len(positive)
        negative_count = count - positive_count
    generator = torch.Generator().manual_seed(seed)
    pos = positive[torch.randperm(len(positive), generator=generator)[:positive_count]]
    neg = negative[torch.randperm(len(negative), generator=generator)[:negative_count]]
    selected = torch.cat((pos, neg)).sort().values.tolist()
    return [int(index) for index in selected], {"positive": positive_count, "negative": negative_count}


def save_source_client(output, client_id, records, stats):
    names = []
    for shard_id in range(0, len(records), SOURCE_SHARD_SIZE):
        shard = records[shard_id:shard_id + SOURCE_SHARD_SIZE]
        path = output / f"source_{client_id}_{shard_id // SOURCE_SHARD_SIZE:04d}.pt"
        torch.save([{k: v for k, v in item.items() if k != "graph_id"} for item in shard], path)
        names.append(path.name)
    return {"id": client_id, "feature_schema": stats.schema(), "graphs": names}


def prepare_molecular_data(cache_root, output, seed=2026):
    cache, output = Path(cache_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new molecular data output directory")
    if not cache.is_dir() or output.is_relative_to(cache):
        raise ValueError("Cache must exist and output must be outside cache_data")
    output.mkdir(parents=True)
    print("Loading processed OGB and TU caches", flush=True)
    pcba = load_ogb("ogbg-molpcba", cache)
    hiv = load_ogb("ogbg-molhiv", cache)
    for name in ("PROTEINS", "IMDB-BINARY"):
        if not (cache / name / "processed/data.pt").is_file():
            raise FileNotFoundError(f"Missing processed TU cache: {name}")
    tud_proteins = TUDataset(str(cache), name="PROTEINS")
    tud_imdb = TUDataset(str(cache), name="IMDB-BINARY")
    pcba_split = pcba.get_idx_split()
    hiv_split = hiv.get_idx_split()
    pcba_indices, pcba_inspected = shuffled_valid_indices(pcba, pcba_split["train"], SOURCE_PCBA_GRAPHS, seed + 11, "PCBA")
    clients, source_reports = [], []

    def build_source(client_id, graphs, ids, schema, degree_features=False):
        records, stats = [], SchemaStats(schema)
        for graph, graph_id in zip(graphs, ids):
            record = graph_record(graph, graph_id, degree_features=degree_features)
            records.append(record); stats.update(record)
        client = save_source_client(output, client_id, records, stats)
        source_reports.append({"client_id": client_id, "graphs": len(records), "schema": client["feature_schema"],
                              "min_nodes": stats.min_nodes, "max_nodes": stats.max_nodes,
                              "shards": [{"path": name, "sha256": file_digest(output / name)} for name in client["graphs"]]})
        clients.append(client)
        print(json.dumps({"prepared_source": client_id, "graphs": len(records),
                          "shards": len(client["graphs"]), "schema": schema}), flush=True)

    build_source("molpcba", (pcba[index] for index in pcba_indices), pcba_indices, molecular_schema())
    build_source("proteins", (tud_proteins[index] for index in range(len(tud_proteins))), range(len(tud_proteins)),
                 {"input_dim": tud_proteins.num_node_features})
    build_source("imdb_binary", (tud_imdb[index] for index in range(len(tud_imdb))), range(len(tud_imdb)),
                 {"input_dim": 1}, degree_features=True)
    protocol_id = f"{PROTOCOL_VERSION}-seed{seed}"
    source_manifest = {"schema_version": 1, "protocol_id": protocol_id, "clients": clients}
    write_json(output / "source_manifest.json", source_manifest)

    target_splits = {}
    for split_id, (split, count) in enumerate(TARGET_COUNTS.items()):
        target_splits[split], _ = stratified_indices(hiv, hiv_split[split], count, seed + 100 + split_id, f"ChemHIV {split}")
    flat_indices = [index for indices in target_splits.values() for index in indices]
    if len(flat_indices) != len(set(flat_indices)):
        raise ValueError("Official target partitions overlap")
    target_entries = {split: [] for split in TARGET_COUNTS}
    target_stats = SchemaStats(molecular_schema())
    target_labels = {split: {"positive": 0, "negative": 0, "missing": 0} for split in TARGET_COUNTS}
    for split, indices in target_splits.items():
        for ordinal, index in enumerate(indices):
            graph = hiv[index]
            label = graph.y.reshape(-1).float()
            label[label == -1] = float("nan")
            if not ((label == 0) | (label == 1) | torch.isnan(label)).all():
                raise ValueError("Target labels must be 0/1/NaN")
            record = graph_record(graph, f"chemhiv_{split}_{index:06d}", label=label)
            target_stats.update(record)
            path = output / f"target_{split}_{ordinal:05d}.pt"
            torch.save(record, path)
            target_entries[split].append({"path": path.name})
            value = label[0].item()
            target_labels[split]["missing" if value != value else ("positive" if value == 1 else "negative")] += 1
        print(json.dumps({"prepared_target_split": split, "graphs": len(indices),
                          "class_counts": target_labels[split]}), flush=True)
    target_manifest = {"schema_version": 1, "protocol_id": protocol_id,
                       "task": {"level": "graph", "objective": "binary", "num_outputs": 1},
                       "visibility": "split_graphs", "feature_schema": target_stats.schema(),
                       "splits": target_entries}
    write_json(output / "target_manifest.json", target_manifest)
    write_json(output / "preparation_report.json", {
        "protocol_id": protocol_id, "seed": seed, "smoke_only": False,
        "purpose": "M1 real molecular engineering validation, not a compositional-generalization benchmark",
        "source": {"pcba_official_split": "train", "pcba_requested": SOURCE_PCBA_GRAPHS,
                   "pcba_inspected_until_valid": pcba_inspected, "pcba_selected_indices": pcba_indices,
                   "imdb_features": "node degree divided by graph maximum degree", "clients": source_reports},
        "target": {"dataset": "ogbg-molhiv", "official_split": "scaffold",
                   "requested_counts": TARGET_COUNTS, "selected_indices": target_splits,
                   "sampling": "fixed class-proportion sampling inside each official partition",
                   "class_counts": target_labels, "schema": target_manifest["feature_schema"],
                   "min_nodes": target_stats.min_nodes, "max_nodes": target_stats.max_nodes},
        "vocabulary": "published OGB atom/bond feature dimensions; no target-data fitting",
        "label_policy": "source labels omitted; existing NaN preserved; target -1 converted to NaN"})
    print(json.dumps({"protocol_id": protocol_id, "source_clients": {r["client_id"]: r["graphs"] for r in source_reports},
                      "target_counts": {key: len(value) for key, value in target_entries.items()},
                      "target_class_counts": target_labels}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path("cache_data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    torch.set_num_threads(1)
    prepare_molecular_data(args.cache_root, args.output, args.seed)


if __name__ == "__main__":
    main()
# ===== FedG2C ADD END A/C =====
