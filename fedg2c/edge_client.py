# ===== FedG2C ADD BEGIN A： local lifecycle for edge-corrupted node prediction =====
from copy import deepcopy
import torch
from .edge_views import NeighborhoodEdgeSampler
from .edge_model import NodewisePredictionModel


class EdgePredictionClient:
    """Reuse Gefra's set/pretrain/get boundary; no server algorithm in this class."""
    def __init__(self, client_id, config, shared, feature_schema, device="cpu"):
        self.client_id, self.config, self.device = client_id, config, torch.device(device)
        self.feature_schema = feature_schema
        self.feature_channels = config.hidden_dim if feature_schema.get("node_cardinalities") else feature_schema["input_dim"]
        self.model = NodewisePredictionModel(config, deepcopy(shared), **feature_schema).to(self.device)

    def set_pretrain_model(self, state):
        # ===== FedG2C ADD BEGIN B：接受 Gefra 命名参数下发，保留原状态字典接口 =====
        if isinstance(state, (list, tuple)):
            from .messages import state_from_named_weights
            state = state_from_named_weights(state)
        # ===== FedG2C ADD END B =====
        self.model.main_branch.shared.load_state_dict(state, strict=True)
        self.model.reset_momentum()

    def get_pretrain_model(self):
        return {k: v.detach().cpu().clone() for k, v in self.model.main_branch.shared.state_dict().items()}

    # ===== FedG2C ADD BEGIN B：共享参数快照封装为 Gefra 本地消息 =====
    def get_local_message(self):
        # get_pretrain_model remains the raw-state interface for checkpoints/tests.
        from .messages import client_message
        return client_message(self.get_pretrain_model())
    # ===== FedG2C ADD END B =====

    def pretrain(self, graphs_factory, epochs=1, views_per_graph=1, seed=0, max_steps=None):
        if epochs < 1 or views_per_graph < 1 or (max_steps is not None and max_steps < 1):
            raise ValueError("Training counts must be positive")
        generator = torch.Generator().manual_seed(seed)
        optimizer = torch.optim.AdamW((p for p in self.model.parameters() if p.requires_grad),
                                     lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.model.train()
        metrics, references = [], None
        for epoch in range(epochs):
            for graph in graphs_factory():
                sampler = NeighborhoodEdgeSampler(graph)
                for _ in range(views_per_graph):
                    pair = sampler.sample_edge_view(self.config, self.feature_channels, generator).to(self.device)
                    optimizer.zero_grad(set_to_none=True)
                    loss, outputs = self.model(pair)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite node prediction loss")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], 5.0, error_if_nonfinite=True)
                    optimizer.step()
                    self.model.update_momentum()  # after main-branch optimization
                    current = outputs["reference"].cpu()
                    references = current[-128:] if references is None else torch.cat([references, current])[-128:]
                    metrics.append({"step": len(metrics)+1, "epoch": epoch+1, "loss": loss.item(),
                                    "gradient_norm": float(norm), "context_std": outputs["context_std"].item(),
                                    "view_nodes": pair.visible_view.num_nodes,
                                    "prediction_nodes": pair.prediction_indices.numel(),
                                    "reference_edge_arcs": pair.reference_view.edge_index.shape[1],
                                    "visible_edge_arcs": pair.visible_view.edge_index.shape[1],
                                    "removed_edge_arcs": pair.removed_edges,
                                    "masked_feature_channels": int((~pair.feature_keep).sum()),
                                    "feature_channels": len(pair.feature_keep)})
                    if max_steps is not None and len(metrics) >= max_steps:
                        return self._summary(metrics, references)
        if not metrics:
            raise ValueError("Client has no permitted training graphs")
        return self._summary(metrics, references)

    @staticmethod
    def _summary(metrics, references):
        return {"steps": metrics, "reference_sample_count": len(references),
                "reference_std": references.std(dim=0, unbiased=False).mean().item(),
                "note": "Last <=128 node references; a diagnostic, not a non-collapse guarantee"}
# ===== FedG2C ADD END A =====
