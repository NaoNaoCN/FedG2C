"""Server-side model construction and federated aggregation for Gefra."""

import time

import numpy as np
import ot  # Python Optimal Transport library
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from geoopt import ManifoldParameter
from geoopt.manifolds.stereographic.math import project as stereographic_project

from model.graph_encoders import InnerProductDecoder
from model.riemannian_encoders import KappaGCN
from model.experts import EuclideanExperts, LinearGating, RiemannianExperts
from model.pretraining import GefraPretrainingModel
from model.expert_relations import reconstruct_relation_matrix_from_factor
from model.gyro_projectors import (
    RiemannianGyroKernelProjector,
    RiemannianGyroKernelProjectorStrict,
)


class Server:
    """Aggregate client expert and projector updates into the global model."""

    def __init__(self, args, device, kmeans_init=False):
        self.per_global_message = None
        self.args = args
        self.device = device
        self.dim = getattr(args, 'hidden_dim', 768)
        self.use_fedavg = getattr(args, 'ablation_fedavg', False)
        self.ablation_no_geo_projector = getattr(args, 'ablation_no_geo_projector', False)

        self.num_experts = args.num_experts
        backbone = getattr(args, 'gnn_backbone', 'mysage')
        # Euclidean-MoE ablations replace curvature-aware experts.
        if getattr(args, 'ablation_moe', False):
            print("[Server] Ablation: Using Euclidean MoE instead of Riemannian MoE")
            self.moe_encoder = EuclideanExperts(
                num_experts=self.args.num_experts,
                input_dim=self.dim,
                hidden_dim=self.dim,
                activation=nn.ReLU,
                num_layers=2,
                backbone=backbone,
                normalize="batch",
                dropout=0.15
            ).to(self.device)
        else:
            self.moe_encoder = RiemannianExperts(
                num_experts=self.args.num_experts,
                input_dim=self.dim,
                hidden_dim=self.dim,
                activation=nn.ReLU,
                num_layers=2,
                backbone="mysage",
                normalize="batch",
                dropout=0.15
            ).to(self.device)

        self.gating = LinearGating(
            feature_dim=self.dim,
            hidden_dim=self.dim,
            dropout=0.15
        ).to(self.device)

        # The shared projector reads curvatures live from the MoE encoder.
        # Euclidean experts do not expose manifolds, so their projector is identity.
        if self.ablation_no_geo_projector or getattr(args, 'ablation_moe', False):
            self.geo_projector = nn.Identity()
        else:
            projector_class = (
                RiemannianGyroKernelProjectorStrict
                if getattr(args, 'use_strict_projector', True)
                else RiemannianGyroKernelProjector
            )
            self.geo_projector = projector_class(
                in_dim=self.dim,
                out_dim=self.dim,
                moe_encoder=self.moe_encoder,
                use_residual=True,
                dropout=0.0,
                norm_type="layernorm",
            ).to(self.device)

        self.feat_recon_decoder = nn.Linear(self.dim, self.dim)
        self.topo_recon_decoder = InnerProductDecoder(hidden_dim=self.dim, output_dim=self.dim)
        self.topo_sem_recon_decoder = nn.Linear(self.dim * 2, self.dim)

        # The server-side local encoder is a container for the global model.
        self.local_encoder = KappaGCN(
            k=-1e-2,
            in_dim=self.dim,
            hidden_dim=self.dim,
            out_dim=self.dim,
            learnable=True,
        )

        self.global_model = GefraPretrainingModel(
            local_encoder=self.local_encoder,
            moe_encoder=self.moe_encoder, gating=self.gating,
            feat_recon_decoder=self.feat_recon_decoder,
            topo_recon_decoder=self.topo_recon_decoder,
            topo_sem_recon_decoder=self.topo_sem_recon_decoder,
            geo_projector=self.geo_projector,
        ).to(device)


        self.round_id = 0

        # pretrain model params
        self.pretrain_epochs = 2
        self.pretrain_batch_size = 1024
        self.feat_p=0.2
        self.edge_p=0.2
        self.topo_recon_ratio=0.1
        self.feat_lambda=100
        self.topo_lambda=0.01
        self.topo_sem_lambda=100
        self.sem_lambda=1
        self.pretrain_lr=1e-4

    def execute(self, local_message_dict, local_codebooks=None, get_gfm=False):
        client_ids = list(local_message_dict.keys())
        client_param_dicts = [dict(local_message_dict[cid]['weight']) for cid in client_ids]
        global_named_params = dict(self.global_model.named_parameters())
        print("Total module num:", len(global_named_params))

        st = time.time()

        # [Ablation] FedAvg: uniform weights, skip GW and permutation alignment
        if self.use_fedavg:
            with torch.no_grad():
                for name, global_param in global_named_params.items():
                    if not self._is_federated_param(name):
                        continue
                    client_tensors = [params[name].data for params in client_param_dicts if name in params]
                    if not client_tensors:
                        continue
                    if not self._all_same_shape(client_tensors, global_param.data.shape):
                        continue
                    aggregated = self._aggregate_with_fedavg(client_tensors)
                    global_param.data.copy_(aggregated.to(device=global_param.device, dtype=global_param.dtype))
            self._project_geo_features_to_manifold()
            et = time.time()
            print(f"-------[FedAvg] Aggregating for {et - st} sec-------")
            self.round_id += 1
            return

        # Weight client updates by their GW distance to the shared barycenter.
        distributions = [
            local_message_dict[cid]['expert_distribution'].numpy()
            for cid in client_ids
        ] # List[Tensor], each shape [num_experts]
        relation_factors = [
            local_message_dict[cid]['expert_relation_factor'].numpy()
            for cid in client_ids
        ] # List[Tensor], each shape [num_experts, relation_factor_dim]
        relation_matrices = [
            reconstruct_relation_matrix_from_factor(L).numpy()
            for L in relation_factors
        ] # List[Tensor], each shape [num_experts, num_experts]
        agg_weights = self._compute_gw_aggregation_weights(distributions, relation_matrices) # shape [num_clients]

        # Align each client's expert indices with the global expert order.
        permutations = [
            self._compute_expert_permutation(params, global_named_params)
            for params in client_param_dicts
        ]
        print(f"[Server] Expert permutations: {permutations}")

        # Apply the alignment before averaging the client parameters.
        aligned_param_dicts = [
            self._apply_permutation(params, perm)
            for params, perm in zip(client_param_dicts, permutations)
        ]

        # Aggregate aligned client parameters with the GW-derived weights.
        with torch.no_grad():
            for name, global_param in global_named_params.items():
                if not self._is_federated_param(name):
                    continue

                client_tensors = [params[name].data for params in aligned_param_dicts if name in params]
                if not client_tensors:
                    continue

                if not self._all_same_shape(client_tensors, global_param.data.shape):
                    continue

                aggregated = self._aggregate_with_weights(client_tensors, agg_weights)
                global_param.data.copy_(aggregated.to(device=global_param.device, dtype=global_param.dtype))
        self._project_geo_features_to_manifold()
        et = time.time()
        print(f"-------Aggregating for {et - st} sec-------")
        self.round_id += 1

    def _compute_expert_permutation(self, client_params, global_named_params):
        """Match client experts to global experts by encoder-weight similarity.

        The returned mapping sends client expert ``i`` to global expert
        ``perm[i]`` and is computed with the Hungarian assignment algorithm.
        """
        num_experts = self.num_experts
        cost = np.zeros((num_experts, num_experts))
        key_template = "moe_encoder.experts.{}.encoder1.lin.weight"

        for i in range(num_experts):
            ck = key_template.format(i)
            if ck not in client_params:
                return list(range(num_experts))
            c_w = client_params[ck].data.to(self.device).float().flatten()
            for j in range(num_experts):
                gk = key_template.format(j)
                if gk not in global_named_params:
                    continue
                g_w = global_named_params[gk].data.to(self.device).float().flatten()
                sim = F.cosine_similarity(c_w.unsqueeze(0), g_w.unsqueeze(0)).item()
                cost[i, j] = -sim

        row_ind, col_ind = linear_sum_assignment(cost)
        perm = [0] * num_experts
        for r, c in zip(row_ind, col_ind):
            perm[r] = c
        return perm

    def _project_geo_features_to_manifold(self):
        """Project strict-projector anchors back onto their expert manifolds.

        Averaging can move manifold parameters close to a numerical boundary.
        The strict projector references each expert curvature directly, so this
        restores a stable point before the next Riemannian optimization step.
        Loose-projector and ablation paths have no manifold anchors to project.
        """
        if self.ablation_no_geo_projector or getattr(self.args, 'ablation_moe', False):
            return
        if not getattr(self.args, 'use_strict_projector', True):
            return
        if not hasattr(self.geo_projector, 'rm_features'):
            return

        with torch.no_grad():
            for i, feat in enumerate(self.geo_projector.rm_features):
                if not isinstance(feat, ManifoldParameter):
                    continue
                k = self.moe_encoder.experts[i].manifold.k
                feat.data = stereographic_project(feat.data, k=k)

    def _apply_permutation(self, client_params, perm):
        """Reindex client parameters according to an expert permutation.

        moe_encoder.experts.{i}.* -> moe_encoder.experts.{perm[i]}.*
        geo_projector.<per-expert ParameterList>.{i} ->
            geo_projector.<per-expert ParameterList>.{perm[i]}.*

        The strict projector has ``rm_features``, ``Ws``, and ``biases``;
        the loose projector additionally has ``lambdas``.  The client message
        is the source of truth here, so a missing block is simply ignored.
        This keeps the helper compatible with projector ablations and future
        projector implementations that omit one of these collections.
        Gating weights are shared across feature dimensions and do not have an
        expert index, so they are left unchanged.
        """
        projector_expert_blocks = {"rm_features", "Ws", "biases", "lambdas"}

        # A malformed assignment must never silently overwrite a different
        # expert entry.  The normal Hungarian result is a permutation, while
        # this fallback keeps the helper safe for hand-crafted messages/tests.
        try:
            is_valid_permutation = (
                len(perm) > 0
                and all(isinstance(index, (int, np.integer)) for index in perm)
                and set(int(index) for index in perm) == set(range(len(perm)))
            )
        except (TypeError, ValueError, AttributeError):
            is_valid_permutation = False
        if not is_valid_permutation:
            perm = []

        def remap_expert_index(parts):
            """Return a remapped parameter path when its expert index is valid."""
            if len(parts) < 3 or not parts[2].isdigit():
                return None

            source_index = int(parts[2])
            if source_index >= len(perm):
                return None

            target_index = perm[source_index]
            if not isinstance(target_index, (int, np.integer)):
                return None
            if not 0 <= int(target_index) < len(perm):
                return None

            parts[2] = str(int(target_index))
            return ".".join(parts)

        new_params = {}
        for name, tensor in client_params.items():
            if name.startswith("moe_encoder.experts."):
                parts = name.split(".")
                remapped_name = remap_expert_index(parts)
                new_params[remapped_name or name] = tensor
            elif name.startswith("geo_projector."):
                parts = name.split(".")
                if len(parts) >= 2 and parts[1] in projector_expert_blocks:
                    remapped_name = remap_expert_index(parts)
                    new_params[remapped_name or name] = tensor
                else:
                    new_params[name] = tensor
            else:
                new_params[name] = tensor
        return new_params

    def _is_federated_param(self, name):
        if name.startswith("moe_encoder.experts"):
            return True
        if name.startswith("geo_projector") and not self.ablation_no_geo_projector:
            return True
        return False

    def _all_same_shape(self, tensors, expected_shape):
        return all(tuple(t.shape) == tuple(expected_shape) for t in tensors)

    def _should_use_gw(self, name, tensor):
        return name.startswith("moe_encoder.experts") and name.endswith("weight") and tensor.ndim == 2

    def _aggregate_with_fedavg(self, tensors):
        stacked = [tensor.detach().to(self.device) for tensor in tensors]
        return torch.stack(stacked, dim=0).mean(dim=0)

    def _aggregate_with_weights(self, tensors, weights):
        """Return the weighted average of compatible client tensors."""
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.sum()
        result = torch.zeros_like(tensors[0], device=self.device, dtype=torch.float32)
        for w, t in zip(weights, tensors):
            result += float(w) * t.detach().to(self.device).float()
        return result

    def _compute_gw_aggregation_weights(
        self,
        distributions,
        relation_matrices,
        temperature=8.0,
        min_weight=1e-3,
    ):
        """Convert GW distances to normalized aggregation weights.

        The method builds a GW barycenter, measures each client relation matrix
        against it, and maps the distances through a temperature-scaled
        exponential.
        """
        S = len(distributions)

        def _sanitize(C):
            C = np.asarray(C, dtype=np.float64)
            C = 0.5 * (C + C.T)
            C = np.maximum(C, 0.0)
            np.fill_diagonal(C, 0.0)
            maxv = C.max()
            if maxv > 1e-12:
                C = C / maxv
            return C

        def _normalize(p):
            p = np.asarray(p, dtype=np.float64)
            p = np.clip(p, 1e-12, None)
            return p / p.sum()

        Cs = [_sanitize(C) for C in relation_matrices]
        ps = [_normalize(p) for p in distributions]

        bary_size = Cs[0].shape[0]
        p_bar = np.ones(bary_size, dtype=np.float64) / bary_size
        lambdas = np.ones(S, dtype=np.float64) / S

        try:
            C_bar = ot.gromov.gromov_barycenters(
                N=bary_size,
                Cs=Cs,
                ps=ps,
                p=p_bar,
                lambdas=lambdas,
                loss_fun="square_loss",
                max_iter=100,
                tol=1e-7,
                verbose=False,
            )
            C_bar = _sanitize(C_bar)
        except Exception as e:
            print(f"[Server] GW barycenter failed ({e}), falling back to uniform weights.")
            return np.ones(S) / S

        distances = []
        for p_c, C_c in zip(ps, Cs):
            try:
                d = ot.gromov.gromov_wasserstein2(
                    C1=C_c, C2=C_bar, p=p_c, q=p_bar,
                    loss_fun="square_loss",
                )
                distances.append(float(d))
            except Exception:
                distances.append(0.0)

        d = np.array(distances, dtype=np.float64)
        logits = -temperature * d
        logits -= logits.max()
        w = np.exp(logits)
        w = np.maximum(w, min_weight)
        w = w / w.sum()
        print(f"[Server] GW distances: {distances}")
        print(f"[Server] Aggregation weights: {w.tolist()}")
        return w

    def get_global_message(self):
        global_message = [
            (name, param) for name, param in self.global_model.named_parameters()
            if name.startswith("moe_encoder.experts")
            or (name.startswith("geo_projector") and not self.ablation_no_geo_projector)
        ]
        return global_message
