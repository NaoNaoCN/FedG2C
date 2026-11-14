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
    def get_local_message(self, num_samples=1):
        # get_pretrain_model remains the raw-state interface for checkpoints/tests.
        from .messages import client_message
        return client_message(self.get_pretrain_model(), num_samples=num_samples)
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
        regularized = self.config.variance_weight > 0
        combined_variance = regularized and self.config.variance_representation == 'both'
        if regularized and (sampling_unit != 'node' or views_per_graph != 1):
            raise ValueError('Root variance requires sampling_unit=node and views_per_graph=1')
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
            scalar_keys = (('loss', 'context_std', 'prediction_loss', 'variance_loss', 'root_std', 'expert_root_std')
                           if combined_variance else
                           ('loss', 'context_std', 'prediction_loss', 'variance_loss', 'root_std')
                           if regularized else ('loss', 'context_std'))
            scalars = torch.stack((norm.detach(), *(diagnostics[key] for key in scalar_keys))).cpu().tolist()
            item = {'step': len(metrics)+1, 'epoch': epoch+1, 'views': pending,
                    'gradient_norm': scalars[0],
                    **{key: value / pending for key, value in zip(scalar_keys, scalars[1:])},
                    **{key: value / pending for key, value in diagnostics.items()
                       if key not in scalar_keys}}
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
            if regularized:
                result['variance_regularization'] = {
                    'weight': self.config.variance_weight,
                    'target': self.config.variance_target,
                    'eps': self.config.variance_eps,
                    'representation': self.config.variance_representation,
                    'view': self.config.variance_view,
                    'combination': 'sum of composed and expert floors' if combined_variance else 'single floor',
                    'statistic': ('online ' + self.config.variance_view + '-view roots across the complete logical batch; '
                                  'each expert/channel separately when representation is experts or both'),
                    'gradient': 'two passes over identical views; global root-loss derivative replayed by physical chunk',
                    'singleton_batch': 'zero auxiliary loss; no cross-batch statistics'}
            return result

        def consume(root_gradient=None, logical_count=None):
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
            backward_loss = loss
            clean_regularization = root_gradient is not None and self.config.variance_view == 'reference'
            if root_gradient is not None and not clean_regularization:
                if combined_variance:
                    composed_gradient, expert_gradient = root_gradient
                    replay = ((outputs['online_roots'] * composed_gradient).sum() +
                              (outputs['online_expert_roots'] * expert_gradient).sum())
                else:
                    replay = (outputs['online_roots'] * root_gradient).sum()
                backward_loss = loss + self.config.variance_weight * logical_count * replay
            backward_loss.backward()
            if clean_regularization:
                clean_roots = regularization_roots(buffered)
                if combined_variance:
                    clean_composed, clean_experts = clean_roots
                    composed_gradient, expert_gradient = root_gradient
                    replay = ((clean_composed * composed_gradient).sum() +
                              (clean_experts * expert_gradient).sum())
                else:
                    replay = (clean_roots * root_gradient).sum()
                (self.config.variance_weight * logical_count * replay).backward()
            pending += count
            total_views += count
            current = outputs['reference']
            references = current[-128:] if references is None else torch.cat((references, current))[-128:]
            diagnostics_values = [('loss', loss.detach()), ('context_std', outputs['context_std'].sum())]
            if regularized:
                diagnostics_values.append(('prediction_loss', loss.detach()))
            for key, value in diagnostics_values:
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

        def regularization_roots(group):
            track_experts = self.config.variance_representation in ('experts', 'both')
            clean = self.config.variance_view == 'reference'
            if min(physical_batch_size, batch_size) == 1:
                pair = group[0].to(self.device)
                view = pair.reference_view if clean else pair.visible_view
                keep = None if clean else pair.feature_keep
                result = self.model.main_branch(view, keep, return_experts=track_experts)
                composed, experts = result if track_experts else (result, None)
                if self.config.variance_representation == 'experts':
                    return experts[pair.root_index:pair.root_index + 1]
                if combined_variance:
                    return (composed[pair.root_index:pair.root_index + 1],
                            experts[pair.root_index:pair.root_index + 1])
                return composed[pair.root_index:pair.root_index + 1]
            pair = EdgePredictionBatch.collate(group).to(self.device)
            view = pair.reference_view if clean else pair.visible_view
            keep = None if clean else pair.feature_keep
            result = self.model.main_branch.forward_batch(view, keep, return_experts=track_experts)
            composed, experts = result if track_experts else (result, None)
            if self.config.variance_representation == 'experts':
                return experts[pair.root_indices]
            if combined_variance:
                return composed[pair.root_indices], experts[pair.root_indices]
            return composed[pair.root_indices]

        def consume_regularized():
            nonlocal buffered
            from .edge_model import variance_floor_loss
            logical_pairs, buffered = buffered, []
            count = len(logical_pairs)
            chunk_size = min(physical_batch_size, batch_size)
            composed_roots, expert_roots = [], []
            with torch.no_grad():
                for start in range(0, count, chunk_size):
                    group = logical_pairs[start:start + chunk_size]
                    result = regularization_roots(group)
                    if combined_variance:
                        composed_roots.append(result[0])
                        expert_roots.append(result[1])
                    else:
                        composed_roots.append(result)
            composed_values = torch.cat(composed_roots).detach().requires_grad_(True)
            if combined_variance:
                expert_values = torch.cat(expert_roots).detach().requires_grad_(True)
                composed_loss = variance_floor_loss(composed_values, self.config.variance_target,
                                                    self.config.variance_eps)
                expert_loss = variance_floor_loss(expert_values, self.config.variance_target,
                                                  self.config.variance_eps)
                variance_loss = composed_loss + expert_loss
                root_gradient = (torch.autograd.grad(variance_loss, (composed_values, expert_values))
                                 if count > 1 else (torch.zeros_like(composed_values),
                                                    torch.zeros_like(expert_values)))
            else:
                variance_loss = variance_floor_loss(composed_values, self.config.variance_target,
                                                    self.config.variance_eps)
                root_gradient = (torch.autograd.grad(variance_loss, composed_values)[0]
                                 if count > 1 else torch.zeros_like(composed_values))
            for start in range(0, count, chunk_size):
                buffered = logical_pairs[start:start + chunk_size]
                if combined_variance:
                    consume((root_gradient[0][start:start + len(buffered)],
                             root_gradient[1][start:start + len(buffered)]), count)
                else:
                    consume(root_gradient[start:start + len(buffered)], count)
            diagnostics['variance_loss'] = variance_loss.detach() * count
            diagnostics['root_std'] = composed_values.detach().std(0, unbiased=False).mean() * count
            if combined_variance:
                diagnostics['expert_root_std'] = expert_values.detach().std(0, unbiased=False).mean() * count
            diagnostics['loss'] += self.config.variance_weight * variance_loss.detach() * count

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
                    if regularized:
                        if len(buffered) == batch_size:
                            consume_regularized()
                    elif len(buffered) == min(physical_batch_size, batch_size - pending):
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
                consume_regularized() if regularized else consume()
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
