"""Trace one synthetic module-A update without modifying the model implementation."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fedg2c.context import GraphView
from fedg2c.edge_config import EdgePredictionConfig
from fedg2c.edge_client import EdgePredictionClient
from fedg2c.edge_views import NeighborhoodEdgeSampler
from fedg2c.model import SharedBackbone
import torch_geometric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new trace output directory")
    torch.set_num_threads(1)
    seed = 20261016
    torch.manual_seed(seed)
    config = EdgePredictionConfig(**json.loads(
        (ROOT / "config/fedg2c_edge_prediction.json").read_text(encoding="utf8")))
    # A nine-node undirected chain; IDs are teaching-only identifiers.
    n = 9
    left = torch.arange(n - 1)
    edges = torch.cat((torch.stack((left, left + 1)),
                       torch.stack((left + 1, left))), dim=1)
    x = torch.randn(n, 5)
    graph = GraphView(x, edges, None, torch.arange(100, 100 + n))
    raw_before = x.clone()
    shared = SharedBackbone(config)
    initial = {k: v.detach().clone() for k, v in shared.state_dict().items()}
    client = EdgePredictionClient("teaching_client", config, shared, {"input_dim": 5})
    client.set_pretrain_model(initial)
    model = client.model
    teacher_before = {k: v.detach().clone()
                      for k, v in model.momentum_branch.named_parameters()}
    trace = {
        "verified_on": "2026-10-16",
        "scope": "One synthetic CPU update; tensor/call/gradient trace, not effectiveness evidence",
        "seed": seed,
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "pyg": torch_geometric.__version__, "threads": 1, "device": "cpu"},
        "config": config.to_dict(),
        "source_graph": {"x": list(x.shape), "edge_index": list(edges.shape),
                         "original_node_id": graph.original_node_id.tolist()},
        "events": [],
    }
    sample_info = {}
    original_sample = NeighborhoodEdgeSampler.sample_edge_view

    def traced_sample(sampler, *sample_args, **sample_kwargs):
        pair = original_sample(sampler, *sample_args, **sample_kwargs)
        info = {
            "reference_x": list(pair.reference_view.x.shape),
            "visible_x": list(pair.visible_view.x.shape),
            "reference_edges": pair.reference_view.edge_index.t().tolist(),
            "visible_edges": pair.visible_view.edge_index.t().tolist(),
            "original_node_id": pair.visible_view.original_node_id.tolist(),
            "same_node_ids": torch.equal(pair.reference_view.original_node_id,
                                        pair.visible_view.original_node_id),
            "same_raw_features": torch.equal(pair.reference_view.x, pair.visible_view.x),
            "root_index": pair.root_index,
            "root_original_node_id": int(pair.visible_view.original_node_id[pair.root_index]),
            "prediction_indices": pair.prediction_indices.tolist(),
            "prediction_original_node_id": pair.visible_view.original_node_id[pair.prediction_indices].tolist(),
            "feature_keep": pair.feature_keep.tolist(),
            "removed_edge_arcs": pair.removed_edges,
        }
        sample_info.update(info)
        trace["events"].append({"stage": "sample_edge_view", **info})
        return pair

    handles = []

    def adapter_hook(stage):
        def hook(module, inputs, output):
            trace["events"].append({"stage": stage,
                                    "input_shape": list(inputs[0].shape),
                                    "output_shape": list(output.shape),
                                    "grad_enabled": torch.is_grad_enabled()})
        return hook

    def composer_hook(stage):
        def hook(module, inputs, output):
            trace["events"].append({"stage": stage,
                                    "expert_repr": list(inputs[0].shape),
                                    "relations": list(inputs[1].shape),
                                    "output_shape": list(output.shape),
                                    "requires_grad": output.requires_grad,
                                    "grad_enabled": torch.is_grad_enabled()})
        return hook

    def predictor_hook(module, inputs, output):
        trace["events"].append({"stage": "GraphSAGEPredictor",
                                "main_z": list(inputs[0].shape),
                                "visible_rwse": list(inputs[1].shape),
                                "prediction_indices": inputs[2].tolist(),
                                "visible_edge_index": list(inputs[3].shape),
                                "output_shape": list(output.shape),
                                "requires_grad": output.requires_grad})

    def model_hook(module, inputs, outputs):
        loss, detached = outputs
        trace["events"].append({"stage": "SmoothL1",
                                "loss": float(loss.detach()),
                                "loss_requires_grad": loss.requires_grad,
                                "prediction_shape": list(detached["prediction"].shape),
                                "reference_shape": list(detached["reference"].shape),
                                "returned_diagnostics_detached": not detached["prediction"].requires_grad
                                                                and not detached["reference"].requires_grad})

    for branch, tag in ((model.main_branch, "main"), (model.momentum_branch, "momentum")):
        handles.append(branch.node_adapter.register_forward_hook(adapter_hook(tag + "_node_adapter")))
        handles.append(branch.shared.composer.register_forward_hook(composer_hook(tag + "_composer")))
    handles.append(model.predictor.register_forward_hook(predictor_hook))
    handles.append(model.register_forward_hook(model_hook))
    NeighborhoodEdgeSampler.sample_edge_view = traced_sample
    try:
        report = client.pretrain(lambda: iter((graph,)), epochs=1, views_per_graph=1,
                                 seed=seed, max_steps=1)
    finally:
        NeighborhoodEdgeSampler.sample_edge_view = original_sample
        for handle in handles:
            handle.remove()
    assert sample_info["same_node_ids"] and sample_info["same_raw_features"]
    assert torch.equal(x, raw_before)
    assert all(p.grad is None and not p.requires_grad for p in model.momentum_branch.parameters())

    groups = {
        "node_adapter": list(model.main_branch.node_adapter.parameters()),
        "experts": list(model.main_branch.shared.experts.parameters()),
        "composer": list(model.main_branch.shared.composer.parameters()),
        "predictor": list(model.predictor.parameters()),
    }
    grad_info = {}
    for name, parameters in groups.items():
        values = [p.grad for p in parameters if p.grad is not None]
        assert values and all(torch.isfinite(value).all() for value in values)
        grad_info[name] = {"tensors_with_grad": len(values),
                           "grad_norm_after_clipping": float(
                               torch.sqrt(sum(value.square().sum() for value in values)))}
    source = dict(model.main_branch.named_parameters())
    ema_errors = [
        float((parameter - (
            teacher_before[name] * config.momentum + source[name].detach() * (1-config.momentum)
        )).abs().max())
        for name, parameter in model.momentum_branch.named_parameters()
    ]
    assert max(ema_errors) < 1e-6
    uploaded = client.get_pretrain_model()
    assert all(name.startswith(("experts.", "composer.")) for name in uploaded)
    assert all(value.device.type == "cpu" and not value.requires_grad for value in uploaded.values())
    # Fresh downlink preserves private weights and predictor but resets the EMA copy.
    private_before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
        if name.startswith(("main_branch.node_adapter.", "predictor."))
    }
    client.set_pretrain_model(initial)
    private_retained = all(torch.equal(value, model.state_dict()[name])
                           for name, value in private_before.items())
    momentum_reset = all(torch.equal(value, model.momentum_branch.state_dict()[name])
                         for name, value in model.main_branch.state_dict().items())
    assert private_retained and momentum_reset
    trace["checks"] = {
        "raw_source_features_unchanged": torch.equal(x, raw_before),
        "main_parameter_gradients": grad_info,
        "momentum_has_no_gradients": True,
        "momentum_eval_mode": not model.momentum_branch.training,
        "ema_max_abs_error": max(ema_errors),
        "upload_keys": len(uploaded),
        "upload_only_experts_and_composer": True,
        "upload_detached_cpu_snapshots": True,
        "some_shared_weights_changed": any(not torch.equal(v, initial[k]) for k, v in uploaded.items()),
        "private_weights_retained_on_downlink": private_retained,
        "momentum_matches_main_on_downlink": momentum_reset,
    }
    assert trace["checks"]["some_shared_weights_changed"]
    trace["local_report"] = report
    source_files = [
        "pretrain_fedg2c.py", "pretrain_federated_fedg2c.py", "fedg2c/edge_entry.py",
        "fedg2c/edge_client.py", "fedg2c/data.py", "fedg2c/edge_views.py", "fedg2c/context.py",
        "fedg2c/edge_model.py", "fedg2c/model.py", "fedg2c/composer.py",
        "fedg2c/structural_encoding.py", "fedg2c/graph_predictor.py", "fedg2c/edge_config.py",
        "config/fedg2c_edge_prediction.json",
    ]
    trace["source_sha256"] = {
        rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() for rel in source_files
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "trace.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps({
        "steps": len(report["steps"]),
        "view_nodes": sample_info["reference_x"][0],
        "root_original_node_id": sample_info["root_original_node_id"],
        "prediction_original_node_id": sample_info["prediction_original_node_id"],
        "feature_keep": sample_info["feature_keep"],
        "reference_edge_arcs": len(sample_info["reference_edges"]),
        "visible_edge_arcs": len(sample_info["visible_edges"]),
        "loss": report["steps"][0]["loss"],
        "upload_keys": len(uploaded),
        "ema_max_abs_error": max(ema_errors),
        "private_retained_on_downlink": private_retained,
        "momentum_reset_on_downlink": momentum_reset,
    }, indent=2))


if __name__ == "__main__":
    main()

