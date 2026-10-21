# ===== FedG2C ADD BEGIN A： opt-in entry, preserve pretrain_gefra.py verbatim =====
"""Stage-one local training entry; without opt-in dispatch to original Gefra."""
import argparse
from dataclasses import fields
from pathlib import Path
import runpy
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    switch = argparse.ArgumentParser(add_help=False)
    switch.add_argument("--use-new-design", action=argparse.BooleanOptionalAction, default=False)
    switch_args, remaining = switch.parse_known_args(argv)
    # ===== FedG2C ADD BEGIN A： select nodewise design or preserve legacy pool =====
    if switch_args.use_new_design:
        route_parser = argparse.ArgumentParser(add_help=False)
        route_parser.add_argument("--local-design", choices=["edge_prediction", "legacy_pool"],
                                  default="edge_prediction")
        route, remaining = route_parser.parse_known_args(remaining)
        if route.local_design == "edge_prediction":
            from fedg2c.edge_entry import run
            run(remaining)
            return
    # ===== FedG2C ADD END A： old entry body below remains verbatim =====
    if not switch_args.use_new_design:
        # Integration point: old entry, old arguments, old algorithms unchanged.
        sys.argv = [str(Path(__file__).with_name("pretrain_gefra.py")), *remaining]
        runpy.run_path(sys.argv[0], run_name="__main__")
        return

    # Lazy imports keep geometry absent from the new execution path.
    import hashlib
    import json
    import torch
    from fedg2c.config import LocalConfig
    from fedg2c.model import SharedBackbone
    from fedg2c.client import LocalPretrainingClient
    from fedg2c.data import read_manifest, iter_graphs, write_smoke_manifest

    parser = argparse.ArgumentParser(description="FedG2C stage one: independent local updates, no server aggregation")
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Use tiny synthetic local training graphs")
    parser.add_argument("--output", type=Path, required=True, help="Must be a new directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--views-per-graph", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--config", type=Path, help="JSON LocalConfig fields, flags override")
    parser.add_argument("--composer", choices=["hypernetwork_relu", "hypernetwork_linear", "standard_attention", "linear_gate"])
    parser.add_argument("--relation-condition", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--objective", choices=["joint_embedding_prediction", "masked_feature_prediction"])
    args = parser.parse_args(remaining)
    if bool(args.source_manifest) == bool(args.smoke):
        parser.error("Choose exactly one of --source-manifest or --smoke")
    if args.epochs < 1 or args.views_per_graph < 1 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("Training counts must be positive")
    raw = json.loads(args.config.read_text()) if args.config else {}
    for name in ("composer", "relation_condition", "objective"):
        if getattr(args, name) is not None:
            raw[name] = getattr(args, name)
    config = LocalConfig(**raw)
    if config.composer == "linear_gate" and config.relation_condition:
        parser.error("linear_gate has no relation branch; explicitly pass --no-relation-condition")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1) if args.smoke else None
    manifest_path = write_smoke_manifest(args.output / "smoke_inputs") if args.smoke else args.source_manifest
    manifest = read_manifest(manifest_path)
    shared = SharedBackbone(config)
    initial_state = {k: v.detach().clone() for k, v in shared.state_dict().items()}
    metadata = {"schema_version": 1, "stage": "local_pretraining_only", "config": config.to_dict(),
                "protocol_id": manifest["protocol_id"], "seed": args.seed,
                "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "source_manifest": manifest, "torch_version": str(torch.__version__),
                "shared_keys": list(initial_state), "relation_schema": ["self", "direct_edge", "two_hop"]}
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    for index, record in enumerate(manifest["clients"]):
        client = LocalPretrainingClient(record["id"], config, shared, record["feature_schema"], args.device)
        client.set_pretrain_model(initial_state)  # same initialization across source feature spaces
        report = client.pretrain(lambda: iter_graphs(record["graphs"]), args.epochs,
                                 args.views_per_graph, args.seed + index, args.max_steps)
        folder = args.output / f"client_{index}"
        folder.mkdir()
        torch.save({"metadata": metadata, "client_id": record["id"], "state_dict": client.get_pretrain_model()},
                   folder / "shared_backbone.pt")
        torch.save({"feature_schema": record["feature_schema"], "config": config.to_dict(),
                    "state_dict": client.model.state_dict()}, folder / "local_training_state.pt")
        (folder / "metrics.json").write_text(json.dumps(report, indent=2))
        print(f"client={record['id']} steps={len(report['steps'])} last_loss={report['steps'][-1]['loss']:.6f}")
    print("Stage one complete: per-client shared states saved; server aggregation is not implemented in this entry.")


if __name__ == "__main__":
    main()
# ===== FedG2C ADD END A =====
