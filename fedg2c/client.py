# ===== FedG2C ADD BEGIN A： local lifecycle compatible with Gefra message boundaries =====
from copy import deepcopy
import torch
from .context import ContextSampler
from .model import LocalPredictionModel


class LocalPretrainingClient:
    """Stage one only: same set/pretrain/get boundary as Gefra Client.

    Does not subclass the old Client: its imports pull in geometric modules.
    Optimizer is reset per local call, as in the Gefra pretraining lifecycle.
    """
    def __init__(self, client_id, config, shared, feature_schema, device="cpu"):
        self.client_id, self.config, self.device = client_id, config, torch.device(device)
        self.feature_schema = feature_schema
        self.model = LocalPredictionModel(config, deepcopy(shared), **feature_schema).to(self.device)

    def set_pretrain_model(self, shared_state):
        self.model.online.shared.load_state_dict(shared_state, strict=True)
        self.model.reset_momentum()

    def get_pretrain_model(self):
        return {k: v.detach().cpu().clone() for k, v in self.model.online.shared.state_dict().items()}

    def pretrain(self, graphs_factory, epochs=1, views_per_graph=1, seed=0, max_steps=None):
        if epochs < 1 or views_per_graph < 1 or (max_steps is not None and max_steps < 1):
            raise ValueError("Training counts must be positive")
        generator = torch.Generator().manual_seed(seed)
        optimizer = torch.optim.AdamW((p for p in self.model.parameters() if p.requires_grad),
                                     lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.model.train()
        metrics, references = [], []
        for epoch in range(epochs):
            for graph in graphs_factory():
                sampler = ContextSampler(graph)
                for _ in range(views_per_graph):
                    pair = sampler.sample(self.config.max_nodes, self.config.prediction_ratio, generator).to(self.device)
                    optimizer.zero_grad(set_to_none=True)
                    loss, outputs = self.model(pair)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite local prediction loss")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], 5.0, error_if_nonfinite=True)
                    optimizer.step()
                    self.model.update_momentum()  # after online optimizer, never before
                    references.append(outputs["reference"].cpu())
                    # Bounded diagnostic cache; report estimate, not proof of non-collapse.
                    references = references[-128:]
                    metrics.append({"step": len(metrics) + 1, "epoch": epoch + 1,
                                    "loss": loss.item(), "gradient_norm": float(norm),
                                    "context_std": outputs["context_std"].item(),
                                    "context_nodes": pair.context.num_nodes,
                                    "prediction_nodes": pair.prediction_indices.numel()})
                    if max_steps is not None and len(metrics) >= max_steps:
                        return self._summary(metrics, references)
        if not metrics:
            raise ValueError("Client has no permitted training graphs")
        return self._summary(metrics, references)

    @staticmethod
    def _summary(metrics, references):
        values = torch.cat(references)
        return {"steps": metrics, "reference_sample_count": len(values),
                "reference_std": values.std(dim=0, unbiased=False).mean().item(),
                "note": "std diagnostic is unreliable for very few samples; not a non-collapse guarantee"}
# ===== FedG2C ADD END A =====
