# ===== FedG2C ADD BEGIN A/C: cached node datasets to explicit tensor manifests =====
"""Prepare the approved five-source -> Cora transductive engineering protocol."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch_geometric.datasets import Amazon, Planetoid, WikiCS
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import is_undirected

from .context import GraphView, validate_graph
from utils.split_protocol import make_paper_node_split


SOURCE_DATASETS = ("citeseer", "pubmed", "photo", "computers", "wikics")
TARGET_DATASET = "cora"


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cached_graph(cache, name):
    """Refuse missing processed caches instead of downloading or reprocessing."""
    if name == "wikics":
        processed = cache / "wikics/processed/data_undirected.pt"
    else:
        processed = cache / name.capitalize() / "processed/data.pt"
    if not processed.is_file():
        raise FileNotFoundError(f"Missing processed cache: {processed}")
    original_digest = file_digest(processed)
    if name in ("cora", "citeseer", "pubmed"):
        dataset = Planetoid(str(cache), name.capitalize(), transform=NormalizeFeatures())
    elif name in ("photo", "computers"):
        dataset = Amazon(str(cache), name.capitalize(), transform=NormalizeFeatures())
    elif name == "wikics":
        dataset = WikiCS(str(cache / "wikics"), is_undirected=True)
    else:
        raise ValueError(f"Unsupported node dataset: {name}")
    graph = dataset[0]
    if file_digest(processed) != original_digest:
        raise RuntimeError(f"Processed source cache changed during loading: {name}")
    if graph.edge_attr is not None:
        raise ValueError(f"Unexpected edge attributes in topology-only protocol: {name}")
    record = {"x": graph.x.detach().cpu().float().contiguous(),
              "edge_index": graph.edge_index.detach().cpu().long().contiguous(),
              "original_node_id": torch.arange(graph.num_nodes)}
    validate_graph(GraphView(record["x"], record["edge_index"], None, record["original_node_id"]))
    if not is_undirected(record["edge_index"], num_nodes=graph.num_nodes):
        raise ValueError(f"Expected bidirectional edges for the configured masking protocol: {name}")
    audit = {"dataset": name, "nodes": graph.num_nodes,
             "edge_arcs": graph.num_edges, "input_dim": graph.num_node_features,
             "feature_preprocessing": "as_provided" if name == "wikics" else "NormalizeFeatures",
             "processed_cache": processed.relative_to(cache).as_posix(),
             "processed_cache_sha256": original_digest}
    return record, graph.y.detach().cpu().reshape(-1).long(), audit


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare_node_data(cache, output, seed=2025):
    cache, output = Path(cache).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new data output directory")
    if not cache.is_dir():
        raise FileNotFoundError(cache)
    if output.is_relative_to(cache):
        raise ValueError("Write prepared tensors outside the shared dataset cache")
    protocol = f"fedg2c-cora-transductive-20_40_40-seed{seed}-v1"
    output.mkdir(parents=True)
    clients, source_audit = [], []
    for name in SOURCE_DATASETS:
        record, _, audit = load_cached_graph(cache, name)
        tensor_path = output / f"{name}.pt"
        # No source labels, source split masks, or precomputed PE are exported.
        torch.save(record, tensor_path)
        audit["tensor_sha256"] = file_digest(tensor_path)
        source_audit.append(audit)
        clients.append({"id": name, "feature_schema": {"input_dim": record["x"].shape[1]},
                        "graphs": [tensor_path.name]})
        print(json.dumps({"prepared_source": name, "nodes": audit["nodes"],
                          "input_dim": audit["input_dim"]}), flush=True)
    source = {"schema_version": 1, "protocol_id": protocol, "clients": clients}

    record, labels, target_audit = load_cached_graph(cache, TARGET_DATASET)
    train, valid, test = make_paper_node_split(labels, TARGET_DATASET, seed)
    masks = {"train": train, "valid": valid, "test": test}
    if not torch.all(train.to(torch.int32) + valid.to(torch.int32) + test.to(torch.int32) == 1):
        raise RuntimeError("Target split must cover each node exactly once")
    record.update(graph_id="cora", y=labels)
    tensor_path = output / "cora.pt"
    torch.save(record, tensor_path)
    target = {"schema_version": 1, "protocol_id": protocol,
              "task": {"level": "node", "objective": "multiclass", "num_outputs": int(labels.max()) + 1},
              "visibility": "transductive", "feature_schema": {"input_dim": record["x"].shape[1], "edge_dim": 0},
              "splits": {split: [{"path": tensor_path.name, "indices": mask.nonzero().flatten().tolist()}]
                         for split, mask in masks.items()}}
    target_audit.update(tensor_sha256=file_digest(tensor_path),
                        split_sizes={split: int(mask.sum()) for split, mask in masks.items()},
                        class_counts={split: torch.bincount(labels[mask], minlength=target["task"]["num_outputs"]).tolist()
                                      for split, mask in masks.items()})
    write_json(output / "source_manifest.json", source)
    write_json(output / "target_manifest.json", target)
    write_json(output / "preparation_report.json", {
        "protocol_id": protocol, "seed": seed, "smoke_only": False,
        "purpose": "real-data engineering validation, not a compositional-generalization benchmark",
        "source_labels_exported": False, "target_in_source_pretraining": False,
        "target_structure_visibility": "all Cora features/edges available; only training-root labels fit parameters",
        "sources": source_audit, "target": target_audit})
    print(json.dumps({"target": TARGET_DATASET, "split_sizes": target_audit["split_sizes"],
                      "protocol_id": protocol}), flush=True)
    return output / "source_manifest.json", output / "target_manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path("cache_data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()
    torch.set_num_threads(1)
    prepare_node_data(args.cache_root, args.output, args.seed)


if __name__ == "__main__":
    main()
# ===== FedG2C ADD END A/C =====
