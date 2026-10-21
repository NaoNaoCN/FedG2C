# ===== FedG2C ADD BEGIN A： revision-two command line and portable artifacts =====
import argparse
import hashlib
import json
from pathlib import Path
import torch
from .edge_config import EdgePredictionConfig
from .edge_client import EdgePredictionClient
from .model import SharedBackbone
from .data import read_manifest, iter_graphs, write_smoke_manifest


def run(argv):
    """
    各客户端本地训练的总入口
    """
    
    parser = argparse.ArgumentParser(description="FedG2C: local nodewise joint-embedding prediction")
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True, help="New directory only")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--views-per-graph", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--edge-mask-rate", type=float)
    parser.add_argument("--feature-mask-rate", type=float)
    parser.add_argument("--position-encoding", choices=["random_walk", "none"])
    parser.add_argument("--composer", choices=["hypernetwork_relu", "hypernetwork_linear", "standard_attention", "linear_gate"])
    parser.add_argument("--relation-condition", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--undirected", action=argparse.BooleanOptionalAction, default=None)
    # ===== FedG2C ADD BEGIN A： predictor selection =====
    parser.add_argument("--predictor", choices=["graphsage", "transformer"])
    # ===== FedG2C ADD END A =====
    args = parser.parse_args(argv)
    if bool(args.source_manifest) == bool(args.smoke):
        parser.error("Choose exactly one of --source-manifest or --smoke")
    if args.epochs < 1 or args.views_per_graph < 1 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("Training counts must be positive")
    config_path = args.config
    if config_path is None and args.smoke:
        config_path = Path(__file__).resolve().parents[1] / "config/fedg2c_edge_prediction.json"
    if config_path is None:
        parser.error("Real-source training requires an explicit --config with masking rates")
    raw = json.loads(config_path.read_text())
    for key in ("edge_mask_rate", "feature_mask_rate", "position_encoding", "composer", "relation_condition", "undirected"):
        if getattr(args, key) is not None:
            raw[key] = getattr(args, key)
    # ===== FedG2C ADD BEGIN A： override JSON predictor selection =====
    if args.predictor is not None:
        raw["predictor_type"] = args.predictor
    # ===== FedG2C ADD END A =====
    config = EdgePredictionConfig(**raw)
    # Validate external input before creating output artifacts.
    if not args.smoke:
        manifest_path = args.source_manifest
        manifest = read_manifest(manifest_path)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    if args.smoke:
        torch.set_num_threads(1)
        manifest_path = write_smoke_manifest(args.output / "smoke_inputs")
        manifest = read_manifest(manifest_path)
    shared = SharedBackbone(config) # 共享模块
    initial_state = {k: v.detach().clone() for k, v in shared.state_dict().items()}
    # Do not serialize the loader's resolved host paths. This is provenance,
    # not a restart manifest; training artifacts are not optimizer checkpoints.
    provenance = json.loads(manifest_path.read_text())
    for record in provenance["clients"]:
        record["graphs"] = [Path(p).name if Path(p).is_absolute() else p for p in record["graphs"]]
    # 预训练元数据
    metadata = {"schema_version": 2, "stage": "local_pretraining_only", "local_design": "edge_prediction",
                "shared_schema_version": 1, "config": config.to_dict(), "seed": args.seed,
                "protocol_id": manifest["protocol_id"], "torch_version": str(torch.__version__),
                "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "source_manifest_provenance": provenance,
                "provenance_path_note": "Paths reference original manifest folder; absolute paths reduced to filenames; not a resume manifest",
                "shared_keys": list(initial_state), "relation_schema": ["self", "direct_edge", "two_hop"],
                "masking_note": "Continuous: raw feature channels. Categorical: embedding channels. Undirected: paired arcs.",
                "reference_code_manifest": "third_party_references/manifest.json"}
    # ===== FedG2C ADD BEGIN B：为服务器记录共同初始化指纹 =====
    from .server_io import state_digest
    metadata["initial_shared_sha256"] = state_digest(initial_state)
    metadata["round_id"] = 0
    # ===== FedG2C ADD END B =====
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    for index, record in enumerate(manifest["clients"]):
        client = EdgePredictionClient(record["id"], config, shared, record["feature_schema"], args.device)
        client.set_pretrain_model(initial_state)
        report = client.pretrain(lambda: iter_graphs(record["graphs"]), args.epochs,
                                 args.views_per_graph, args.seed+index, args.max_steps)
        folder = args.output / f"client_{index}"
        folder.mkdir()
        torch.save({"metadata": metadata, "client_id": record["id"], "state_dict": client.get_pretrain_model()},
                   folder / "shared_backbone.pt")
        torch.save({"feature_schema": record["feature_schema"], "config": config.to_dict(),
                    "local_design": "edge_prediction", "state_dict": client.model.state_dict()},
                   folder / "local_training_state.pt")
        (folder / "metrics.json").write_text(json.dumps(report, indent=2))
        print(f"client={record['id']} steps={len(report['steps'])} last_loss={report['steps'][-1]['loss']:.6f}")
    print("Local stage complete. Saved experts + composer per client; no server aggregation performed.")
# ===== FedG2C ADD END A =====
