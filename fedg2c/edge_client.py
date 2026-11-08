# ===== FedG2C ADD BEGIN A： local lifecycle for edge-corrupted node prediction =====
from copy import deepcopy
import json
import time
import torch
from .edge_views import NeighborhoodEdgeSampler
from .edge_model import NodewisePredictionModel
from .batching import EdgePredictionBatch


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

    def pretrain(self, graphs_factory, epochs=1, views_per_graph=1, seed=0, max_steps=None,
                 log_every=0, batch_size=1, sampling_unit='graph', physical_batch_size=1):
        if epochs < 1 or views_per_graph < 1 or (max_steps is not None and max_steps < 1):
            raise ValueError("Training counts must be positive")
        if type(batch_size) is not int or batch_size < 1 or sampling_unit not in ('graph', 'node'):
            raise ValueError('Use positive batch_size and graph/node sampling_unit')
        if type(physical_batch_size) is not int or physical_batch_size < 1:
            raise ValueError('Use positive physical_batch_size')
        if min(physical_batch_size, batch_size) > 1 and self.config.dropout:
            raise ValueError('Use physical_batch_size=1 with dropout to preserve the scalar RNG stream')
        generator = torch.Generator().manual_seed(seed)
        optimizer = torch.optim.AdamW((p for p in self.model.parameters() if p.requires_grad),
                                     lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.model.train()
        metrics, references = [], None
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        optimizer.zero_grad(set_to_none=True)
        pending, diagnostics, buffered = 0, {}, []
        root_coverage, total_views = [], 0
        started = time.monotonic()

        def update(epoch):
            nonlocal pending, diagnostics
            # Each view loss already averages its prediction nodes. Mean by
            # independent view matches the logical batch; graphs never attend
            # to one another and the last partial batch uses its actual size.
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.div_(pending)
            norm = torch.nn.utils.clip_grad_norm_(parameters, self.config.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            self.model.update_momentum()
            optimizer.zero_grad(set_to_none=True)
            # Diagnostics stay on the device until an optimizer update. No
            # per-view .item(), reference D2H copy, or GPU validation readback.
            scalars = torch.stack((norm.detach(), diagnostics['loss'], diagnostics['context_std'])).cpu().tolist()
            item = {'step': len(metrics)+1, 'epoch': epoch+1, 'views': pending,
                    'gradient_norm': scalars[0], 'loss': scalars[1] / pending,
                    'context_std': scalars[2] / pending,
                    **{key: value / pending for key, value in diagnostics.items()
                       if key not in ('loss', 'context_std')}}
            metrics.append(item)
            if log_every and (len(metrics) == 1 or len(metrics) % log_every == 0):
                print(json.dumps({'stage':'local_pretraining','client':self.client_id,
                                  'steps':len(metrics),'views':total_views,'loss':item['loss'],
                                  'elapsed_seconds':round(time.monotonic()-started,2)}),flush=True)
            pending, diagnostics = 0, {}

        def summary():
            result = self._summary(metrics, references)
            result.update(batch_size=batch_size, physical_batch_size=physical_batch_size,
                          sampling_unit=sampling_unit, total_views=total_views,
                          loss_reduction='mean prediction-node loss per view, then mean independent views per update',
                          root_coverage=root_coverage)
            return result

        def consume():
            nonlocal pending, total_views, references, buffered
            count = len(buffered)
            if min(physical_batch_size, batch_size) == 1:
                # Keep the scalar forward as a reference and for dropout runs.
                loss, outputs = self.model(buffered[0].to(self.device))
            else:
                batch = EdgePredictionBatch.collate(buffered).to(self.device, non_blocking=True)
                loss, outputs = self.model.forward_batch(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite node prediction loss')
            loss.backward()
            pending += count
            total_views += count
            current = outputs['reference']
            references = current[-128:] if references is None else torch.cat((references, current))[-128:]
            for key, value in (('loss', loss.detach()), ('context_std', outputs['context_std'].sum())):
                diagnostics[key] = diagnostics.get(key, 0) + value
            for pair in buffered:
                values = {'view_nodes': pair.visible_view.num_nodes,
                          'prediction_nodes': pair.prediction_indices.numel(),
                          'reference_edge_arcs': pair.reference_view.edge_index.shape[1],
                          'visible_edge_arcs': pair.visible_view.edge_index.shape[1],
                          'removed_edge_arcs': pair.removed_edges,
                          'masked_feature_channels': int((~pair.feature_keep).sum()),
                          'feature_channels': len(pair.feature_keep)}
                for key, value in values.items():
                    diagnostics[key] = diagnostics.get(key, 0) + value
            buffered = []

        # Node clients have one large graph; retain its adjacency index across
        # epochs, while graph datasets retain only one transient small index.
        node_samplers = {}
        for epoch in range(epochs):
            for graph_id, graph in enumerate(graphs_factory()):
                if sampling_unit == 'node':
                    if not node_samplers:
                        node_samplers[0] = NeighborhoodEdgeSampler(graph)
                    sampler = node_samplers[0]
                    roots = sampler.eligible_roots
                    order = roots[torch.randperm(len(roots), generator=generator)].tolist()
                else:
                    sampler, order = NeighborhoodEdgeSampler(graph), [None]
                covered = []
                for root in order:
                  for _ in range(views_per_graph):
                    buffered.append(sampler.sample_edge_view(self.config, self.feature_channels, generator, root))
                    if root is not None:
                        covered.append(root)
                    if len(buffered) == min(physical_batch_size, batch_size - pending):
                        consume()
                    if pending == batch_size:
                        update(epoch)
                        if max_steps is not None and len(metrics) >= max_steps:
                            if sampling_unit == 'node':
                                root_coverage.append({'epoch':epoch+1, 'graph':graph_id,
                                    'eligible_roots':len(order), 'roots_visited':len(set(covered)),
                                    'complete':len(set(covered))==len(order)})
                            return summary()
                if sampling_unit == 'node':
                    root_coverage.append({'epoch':epoch+1, 'graph':graph_id,
                        'eligible_roots':len(order), 'roots_visited':len(set(covered)), 'complete':True})
            # Do not merge the final incomplete logical batch across epochs.
            if buffered:
                consume()
            if pending:
                update(epoch)
                if max_steps is not None and len(metrics) >= max_steps:
                    return summary()
        if not metrics:
            raise ValueError("Client has no permitted training graphs")
        return summary()

    @staticmethod
    def _summary(metrics, references):
        return {"steps": metrics, "reference_sample_count": len(references),
                "reference_std": references.std(dim=0, unbiased=False).mean().item(),
                "note": "Last <=128 node references; a diagnostic, not a non-collapse guarantee"}
# ===== FedG2C ADD END A =====
