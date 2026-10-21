"""Federated client training and fine-tuning orchestration for Gefra."""

import os
import copy
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from model.graph_encoders import GraphEncoder, InnerProductDecoder
from model.finetuning import GefraFinetuningModel
from model.pretraining import GefraPretrainingModel
from model.experts import EuclideanExperts, LinearGating, RiemannianExperts
from model.gyro_projectors import (
    RiemannianGyroKernelProjector,
    RiemannianGyroKernelProjectorStrict,
)
from geoopt import ManifoldParameter
from model.expert_relations import (
    compute_expert_weight_distribution,
    compute_expert_relation_matrix,
    low_rank_factor,
)
from torch.optim import AdamW
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.utils import mask_feature, dropout_adj
from utils.basic_utils import EarlyStopping, Logger
from utils.basic_utils import get_loader, seed_everything, get_preprocess
from task.node_cls import ft_node, eval_node
from task.link_pre import ft_link, eval_link
from task.graph_cls import ft_graph, eval_graph
from geoopt.optim import RiemannianAdam


def split_optimization_parameters(module):
    """Separate Euclidean parameters from curvature and manifold parameters.

    Geoopt represents learnable curvatures as ordinary scalar Parameters under
    ``*.manifold.k``. Strict-projector anchors are explicit
    ``ManifoldParameter`` objects. Both belong to RiemannianAdam; every other
    trainable parameter belongs to AdamW.
    """
    euclidean_params = []
    riemannian_params = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if isinstance(parameter, ManifoldParameter) or name.endswith(".manifold.k"):
            riemannian_params.append(parameter)
        else:
            euclidean_params.append(parameter)
    return euclidean_params, riemannian_params


class Client:
    """Own one client partition and execute local Gefra updates."""

    def __init__(self, args, client_id, data_tag, device, finetune_params=None, mode="pretrain", backbone="mysage"):
        self.args = args
        self.client_id = client_id
        self.mode = mode
        self.topk = getattr(args, 'topk', 2)
        # PyG graph-dataset slices stay on CPU; the DataLoader moves each batch
        # to the selected device.
        if data_tag.name.lower() not in ["chemhiv", "chemblpre", "chempcba", "proteins", "imdb-binary"]:
            self.data_tag = data_tag.to(device)
        else:
            self.data_tag = data_tag
        self.device = device
        # Read features directly for a single graph and from the first item for
        # a graph-dataset slice.
        from torch_geometric.data import Data as _PyGData
        if isinstance(data_tag, _PyGData):
            if data_tag.x is not None and data_tag.x.dim() == 2:
                self.dim = data_tag.x.size(1)
            else:
                print("No fitting for particular dimension, falling back to 768.")
                self.dim = 768
        elif hasattr(data_tag, '__getitem__') and hasattr(data_tag, '__len__') and len(data_tag) > 0:
            # Indexing triggers dataset transforms such as ``OneHotDegree``.
            # ``InMemoryDataset`` forwards attributes to internal storage, so
            # attribute checks cannot reliably distinguish it from ``Data``.
            first_graph = data_tag[0]
            if hasattr(first_graph, 'x') and first_graph.x is not None and first_graph.x.dim() == 2:
                self.dim = first_graph.x.size(1)
            else:
                print("No fitting for particular dimension, falling back to 128.")
                self.dim = 128
        else:
            print("No fitting for particular dimension, falling back to 128.")
            self.dim = 128  # fallback
        print("dataset:", data_tag.name.lower(), "self.dim", self.dim)

        self.num_experts = args.num_experts # Default number of experts

        # ``input_proj`` is client-private and bridges raw features to hidden_dim.
        self.hidden_dim = getattr(args, 'hidden_dim', 128)
        self.ablation_no_geo_projector = getattr(args, 'ablation_no_geo_projector', False)

        # Dropout regularizes high-dimensional sparse inputs such as CiteSeer.
        proj_dropout = getattr(args, 'proj_dropout', 0.3)
        if self.dim != self.hidden_dim:
            self.input_proj = nn.Sequential(nn.Linear(self.dim, self.hidden_dim), nn.Dropout(proj_dropout))
        elif data_tag.name.lower() in ['fb15k237', 'wn18rr']:
            self.input_proj = nn.Sequential(nn.Linear(self.dim, self.hidden_dim), nn.Dropout(proj_dropout))
        else:
            self.input_proj = nn.Identity()

        if getattr(args, 'ablation_moe', False):
            print(f"[Client {self.client_id}] Initializing Euclidean MoE Components (Ablation)")
            self.moe_encoder = EuclideanExperts(
                num_experts=self.num_experts,
                input_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                activation=nn.ReLU,
                num_layers=2,
                backbone=backbone,
                normalize="batch",
                dropout=0.15,
            )
        else:
            print(f"[Client {self.client_id}] Initializing Riemannian MoE Components")
            self.moe_encoder = RiemannianExperts(
                num_experts=self.num_experts,
                input_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                activation=nn.ReLU,
                dropout=0.15
            )

        self.gating = LinearGating(
            feature_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            dropout=0.15,
        )

        # The geometric projector reads expert curvatures through a live
        # reference. Euclidean-MoE ablations do not expose manifolds.
        if self.ablation_no_geo_projector or getattr(args, 'ablation_moe', False):
            self.geo_projector = nn.Identity()
        else:
            projector_class = (
                RiemannianGyroKernelProjectorStrict
                if getattr(args, 'use_strict_projector', True)
                else RiemannianGyroKernelProjector
            )
            self.geo_projector = projector_class(
                in_dim=self.hidden_dim,
                out_dim=self.hidden_dim,
                moe_encoder=self.moe_encoder,
                use_residual=True,
                dropout=0.0,
                norm_type="layernorm",
            )

        self.local_encoder = GraphEncoder(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            activation=nn.ReLU,
            num_layers=2,
            backbone=backbone,
            normalize="batch",
            dropout=0.15,
        )

        print(f"[Client {self.client_id}] Initialized Local Encoder with Backbone: {backbone}")

        # Decode hidden representations back to the original feature dimension.
        self.feat_recon_decoder = nn.Linear(self.hidden_dim, self.dim)
        self.topo_recon_decoder = InnerProductDecoder(hidden_dim=self.hidden_dim, output_dim=self.hidden_dim)

        self.topo_sem_recon_decoder = nn.Linear(self.hidden_dim * 2, self.dim)
        self.pretrain_model = GefraPretrainingModel(
            local_encoder=self.local_encoder,
            moe_encoder=self.moe_encoder, gating=self.gating,
            feat_recon_decoder=self.feat_recon_decoder,
            topo_recon_decoder=self.topo_recon_decoder,
            topo_sem_recon_decoder=self.topo_sem_recon_decoder,
            topk=self.topk,
            input_proj=self.input_proj,
            geo_projector=self.geo_projector,
        ).to(device)

        # Experimental protocol defaults are supplied by the pre-training CLI.
        self.pretrain_epochs = getattr(args, "local_epochs", 5)
        self.pretrain_batch_size = getattr(args, "pretrain_batch_size", 1024)
        self.pretrain_lr = getattr(args, "pretrain_lr", 1e-3)
        self.riemannian_lr = getattr(args, "riemannian_lr", 1e-3)
        self.pretrain_weight_decay = getattr(args, "pretrain_weight_decay", 1e-5)
        self.max_train_batches = getattr(args, "max_train_batches", None)
        self.feat_p=0.2
        self.edge_p=0.2
        # self.topo_recon_ratio=0.1
        self.topo_recon_ratio=1.0
        self.topo_lambda=0.1
        self.struct_align_lambda=5 # Weight for structural alignment loss
        self.curvature_div_lambda = 0.3
        if finetune_params is None:
            self.finetune_batch_size = 0
            self.finetune_lr = 0
            self.early_stop = 0
            self.finetune_epochs = 0
        else:
            self.finetune_batch_size = finetune_params["batch_size"]
            self.finetune_lr = finetune_params["finetune_lr"]
            self.early_stop = finetune_params["early_stop"]
            self.finetune_epochs = finetune_params["finetune_epochs"]
        self.setting="standard"

        # Track expert-use statistics for federated aggregation.
        self.expert_usage = None
        self.routing_weights_list = []
        self.expert_relation_rank = 2
        self._loss_baseline = None

        # Curvature scalars and strict-projector manifold anchors use
        # RiemannianAdam. MoE linear weights remain Euclidean AdamW parameters.
        _, riemannian_params = split_optimization_parameters(self.pretrain_model)
        self.riemannian_optimizer = (
            RiemannianAdam(riemannian_params, lr=self.riemannian_lr)
            if riemannian_params else None
        )

    def pretrain(self, round_id=0, logger=None):
        time_start = time.time()
        print(f"[client {self.client_id}] pretraining...")
        self.pretrain_model.train()

        for e in self.pretrain_model.moe_encoder.experts:
            if hasattr(e, 'manifold'):
                e.manifold.k.requires_grad = True

        # AdamW is recreated for the local client update. Curvature and manifold
        # parameters were placed in self.riemannian_optimizer at initialization.
        other_params, _ = split_optimization_parameters(self.pretrain_model)

        optimizer = AdamW([
            {'params': other_params, 'lr': self.pretrain_lr},
        ], weight_decay=self.pretrain_weight_decay)

        # Graph classification uses graph batches; node and link tasks use
        # neighborhood sampling.
        task = getattr(self.data_tag, 'task', 'node_cls')
        is_graph_cls = (task == "graph_cls")

        # Relation labels are edge-level, whereas NeighborLoader slices ``y``
        # as a node-level tensor. Pre-training does not consume ``batch.y``,
        # so use a temporary dummy node label without altering fine-tuning.
        if task == "link_pre" and self.data_tag.y is not None and \
                self.data_tag.y.shape[0] == self.data_tag.edge_index.shape[1]:
            self.data_tag._edge_y_backup = self.data_tag.y
            self.data_tag.y = torch.zeros(self.data_tag.x.shape[0], dtype=torch.long)

        self.expert_usage = None
        total_batch_count = 0
        epoch_metrics = []

        for local_epoch in range(1, self.pretrain_epochs + 1):
            # The uploaded routing summary follows the final local epoch, as in
            # the previous implementation, while each epoch now covers its full loader.
            self.routing_weights_list = []
            batch_count = 0
            self._grad_accum = {}
            self._div_accum = {}

            if is_graph_cls:
                loader = DataLoader(self.data_tag, batch_size=64, shuffle=True)
            else:
                if not self.data_tag.edge_index.is_contiguous():
                    self.data_tag.edge_index = self.data_tag.edge_index.contiguous()
                total_idx = torch.arange(self.data_tag.x.shape[0]).long()
                loader = NeighborLoader(
                    self.data_tag,
                    input_nodes=total_idx,
                    num_neighbors=[10] * 2,
                    batch_size=self.pretrain_batch_size,
                    shuffle=True,
                )

            total_topo_recon_loss = 0.0
            total_struct_align_loss = 0.0
            total_curvature_div_loss = 0.0
            total_loss = 0.0
            total_raw_topo = 0.0
            total_raw_struct = 0.0
            total_raw_curvature_div_loss = 0.0

            for batch_data in loader:
                if self.max_train_batches is not None and batch_count >= self.max_train_batches:
                    break

                if is_graph_cls:
                    x = batch_data.x.to(self.device).float()
                    edge_index = batch_data.edge_index.to(self.device)
                    edge_attr = batch_data.edge_attr.to(self.device).float() if batch_data.edge_attr is not None else None
                else:
                    x = batch_data.x.to(self.device)
                    edge_index = batch_data.edge_index.to(self.device)
                    edge_attr = batch_data.edge_attr.to(self.device) if batch_data.edge_attr is not None else None

                graph = [x, edge_index, edge_attr]
                aug_x, _ = mask_feature(x, p=self.feat_p)
                aug_edge_index, aug_edge_attr = dropout_adj(
                    edge_index, edge_attr, p=self.edge_p,
                    force_undirected=True, num_nodes=x.size(0),
                )
                aug_graph = [aug_x, aug_edge_index, aug_edge_attr]
                bs = None if is_graph_cls else batch_data.batch_size
                _, gate_probs_all, mean_gate_weights, losses = self.pretrain_model(
                    aug_graph, graph, self.topo_recon_ratio, bs=bs
                )

                if self.expert_usage is None:
                    self.expert_usage = mean_gate_weights.detach()
                else:
                    self.expert_usage += mean_gate_weights.detach()
                self.routing_weights_list.append(mean_gate_weights.detach().unsqueeze(0).cpu())

                topo_recon_loss = self.topo_lambda * losses['topo_recon_loss']
                struct_align_loss = self.struct_align_lambda * losses['struct_align_loss']
                curvature_div_loss = self.curvature_div_lambda * losses['curvature_div_loss']
                loss = topo_recon_loss + struct_align_loss + curvature_div_loss

                optimizer.zero_grad()
                if self.riemannian_optimizer is not None:
                    self.riemannian_optimizer.zero_grad()
                loss.backward()

                if logger is not None:
                    from utils.metrics_logger import compute_grad_norms, compute_diversity_metrics
                    batch_grads = compute_grad_norms(self.pretrain_model)
                    for group, value in batch_grads.items():
                        self._grad_accum[group] = self._grad_accum.get(group, 0.0) + value
                    batch_div = compute_diversity_metrics(self.pretrain_model, gate_probs_all)
                    for key, value in batch_div.items():
                        self._div_accum[key] = self._div_accum.get(key, 0.0) + value

                nn.utils.clip_grad_norm_(self.pretrain_model.parameters(), 1.0)
                optimizer.step()
                if self.riemannian_optimizer is not None:
                    self.riemannian_optimizer.step()

                total_raw_topo += losses['topo_recon_loss'].item()
                total_raw_struct += losses['struct_align_loss'].item()
                total_raw_curvature_div_loss += losses['curvature_div_loss'].item()
                total_topo_recon_loss += topo_recon_loss.item()
                total_struct_align_loss += struct_align_loss.item()
                total_curvature_div_loss += curvature_div_loss.item()
                total_loss += loss.item()
                batch_count += 1
                total_batch_count += 1

            if batch_count == 0:
                raise RuntimeError(
                    f"Client {self.client_id} produced no batches in local epoch {local_epoch}."
                )

            avg_raw = {
                'topo_recon': total_raw_topo / batch_count,
                'struct_align': total_raw_struct / batch_count,
                'curvature_div_loss': total_raw_curvature_div_loss / batch_count,
            }
            if self._loss_baseline is None:
                self._loss_baseline = avg_raw.copy()

            avg_weighted = {
                'topo_recon': total_topo_recon_loss / batch_count,
                'struct_align': total_struct_align_loss / batch_count,
                'curvature_div_loss': total_curvature_div_loss / batch_count,
                'total': total_loss / batch_count,
            }
            avg_raw['total'] = avg_weighted['total']
            avg_grad = {
                group: value / batch_count for group, value in self._grad_accum.items()
            }
            avg_div = {
                key: value / batch_count for key, value in self._div_accum.items()
            }
            epoch_metrics.append((avg_raw, avg_weighted, avg_grad, avg_div))
            print(
                f"[client {self.client_id}] local epoch {local_epoch}/{self.pretrain_epochs}: "
                f"{batch_count} batches"
            )

        if self.expert_usage is not None:
            self.expert_usage = self.expert_usage / total_batch_count
            print(f"[client {self.client_id}] Expert Usage: {self.expert_usage}")

        if logger is not None:
            def mean_metrics(index):
                keys = set().union(*(metrics[index] for metrics in epoch_metrics))
                return {
                    key: sum(metrics[index].get(key, 0.0) for metrics in epoch_metrics) / len(epoch_metrics)
                    for key in keys
                }

            logger.log(
                round_id,
                self.client_id,
                mean_metrics(0),
                mean_metrics(1),
                mean_metrics(2),
                mean_metrics(3),
            )


        time_end = time.time()

        print(f"[client {self.client_id}]", "-------------", time_end-time_start)
        self.pretrain_model.eval()


    def finetune(
        self,
        standard=1,
        lambda_prompt=0.0,
        lambda_align=0.0,
        num_prompts=4,
        lambda_curv=1.0,
        ablation_learnable_weight=False,
        local_encoder_only=False,
        freeze_moe=True,
        emb_save_dir=None,
        emb_filename_prefix=None,
    ):
        num_tasks = self.data_tag.num_tasks
        task = self.data_tag.task

        train_loader = None
        val_loader = None
        test_loader = None
        subgraph_loader = None
        process = get_preprocess(task)
        dataset = self.data_tag
        dataset = process(dataset)


        labels = self.data_tag.y

        # Relation classification requires one label per edge.
        if task == "link_pre":
            assert labels is not None, f"link_pre requires dataset.y ({self.data_tag.name})."
            assert labels.shape[0] == dataset.edge_index.shape[1], (
                f"link_pre requires {dataset.edge_index.shape[1]} edge labels, "
                f"got {labels.shape[0]} ({self.data_tag.name})."
            )


        num_classes = num_tasks if task == "graph_cls" else self.data_tag.num_global_classes

        split = {"train": dataset.train_mask, "valid": dataset.val_mask, "test": dataset.test_mask}


        self.logger = Logger()
        # Save best-epoch representations for node-classification visualization.
        save_emb = (
            emb_save_dir is not None
            and task == "node_cls"
        )
        for idx in range(standard): # debug: only 3 split
            seed_everything(idx)
            best_state_dict = None

            # Capture per-run wall time and peak memory.
            t0 = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)
            last_eval_epoch = -1

            if self.setting == "standard":
                split = split
            else:
                raise ValueError("Invalid Setting")

            # Zero-shot evaluation keeps the shared modules frozen.
            is_zero_shot = hasattr(self.args, 'k_shot') and self.args.k_shot == 0
            if is_zero_shot:
                freeze_moe_override = True
                print(f"[Zero-shot] Client {self.client_id}: freeze_moe=True")
            else:
                freeze_moe_override = freeze_moe

            moe_enc_copy = copy.deepcopy(self.moe_encoder)
            geo_proj_copy = copy.deepcopy(self.geo_projector)
            if hasattr(geo_proj_copy, "bind_moe_encoder"):
                geo_proj_copy.bind_moe_encoder(moe_enc_copy)
            task_model = GefraFinetuningModel(
                local_encoder=copy.deepcopy(self.local_encoder),
                moe_encoder=moe_enc_copy,
                geo_projector=geo_proj_copy,
                num_classes=num_classes,
                input_proj=copy.deepcopy(self.input_proj),
                num_prompts=num_prompts,
                lambda_curv=lambda_curv,
                use_learnable_weight=ablation_learnable_weight,
                local_encoder_only=local_encoder_only,
            ).to(self.device)

            if not freeze_moe_override:
                for param in task_model.moe_encoder.parameters():
                    param.requires_grad = True
                for param in task_model.geo_projector.parameters():
                    param.requires_grad = True

            # The optional unfrozen-MoE path uses the same split as pre-training:
            # curvature/manifold parameters use RiemannianAdam, while GNN and
            # projector weights remain Euclidean AdamW parameters.
            euclidean_params, riemannian_params = split_optimization_parameters(task_model)
            task_opt = AdamW([
                {'params': [p for p in euclidean_params if id(p) != id(task_model.fusion_weight)]},
                {'params': [task_model.fusion_weight], 'lr': self.finetune_lr * 100},
            ], lr=self.finetune_lr)
            task_riem_opt = RiemannianAdam(riemannian_params, lr=self.finetune_lr) if riemannian_params else None
            # Evaluate periodically to limit full-graph inference overhead.
            eval_interval = 10
            # Interpret early-stop patience in evaluation events.
            effective_patience = max(1, self.early_stop // eval_interval)
            stopper = EarlyStopping(patience=effective_patience)

            # Force sampled mini-batches for graphs that cannot fit in memory.
            force_mini_batch = False
            if task in ["node_cls", "link_pre"]:
                num_nodes = dataset.x.shape[0] if hasattr(dataset, 'x') else 0
                num_edges = dataset.edge_index.shape[1] if hasattr(dataset, 'edge_index') else 0
                if num_nodes > 1000000 or num_edges > 50000000:
                    force_mini_batch = True
                    if self.finetune_batch_size == 0:
                        if dataset.name.lower() == "reddit":
                            effective_batch_size = 512
                        else:
                            effective_batch_size = 4096 if num_nodes < 5000000 else 512
                        print(
                            f"[Memory] {self.data_tag.name} is large "
                            f"(nodes: {num_nodes}, edges: {num_edges}); "
                            f"using sampled batches of {effective_batch_size}."
                        )
                    else:
                        effective_batch_size = self.finetune_batch_size
                else:
                    effective_batch_size = self.finetune_batch_size
            else:
                effective_batch_size = self.finetune_batch_size

            print(f"[Finetune] Dataset {self.data_tag.name} | Standard {idx} | Task: {task} | Batch Size: {effective_batch_size} | Force Mini-Batch: {force_mini_batch}")

            if (self.finetune_batch_size != 0 or force_mini_batch) and task in ["node_cls", "link_pre"]:
                train_loader, subgraph_loader = get_loader(dataset, split, labels, task, effective_batch_size)
            elif task == "graph_cls":
                # Graph tasks always require a positive PyG DataLoader batch size.
                graph_cls_batch_size = self.finetune_batch_size if self.finetune_batch_size != 0 else 256
                train_loader, val_loader, test_loader = get_loader(dataset, split, labels, task, graph_cls_batch_size)
            finetune = get_ft(task)
            evaluate = get_eval(task)

            pbar = tqdm(range(self.finetune_epochs), desc=f"Finetuning - Dataset {self.data_tag.name} - Standard {idx} - {self.data_tag.task}")
            metrics = {
                "curve": [],
                "curve_grad": []
            }
            for epoch in pbar:

                loss = finetune(
                    model=task_model,
                    dataset=dataset if task in ["node_cls", "link_pre"] else dataset,
                    loader=train_loader,
                    optimizer=task_opt,
                    rm_optimizer=task_riem_opt,
                    split=split,
                    labels=labels,
                    num_classes=num_classes,
                    lambda_prompt=lambda_prompt,
                    lambda_align=lambda_align,
                )
                if isinstance(loss, dict):
                    loss_str = " | ".join(f"{k}: {v:.6f}" for k, v in loss.items())
                    print(f"[Finetune] Dataset {self.data_tag.name} | Standard {idx} | Epoch {epoch} | {loss_str}")
                else:
                    print(f"[Finetune] Dataset {self.data_tag.name} | Standard {idx} | Epoch {epoch} | Loss: {float(loss):.6f}")

                should_eval = (epoch % eval_interval == 0) or (epoch == self.finetune_epochs - 1)
                if should_eval:
                    last_eval_epoch = epoch
                    result = evaluate(
                        model=task_model,
                        dataset= dataset if task in ["node_cls", "link_pre"] else dataset,
                        loader= subgraph_loader if task in ["node_cls", "link_pre"] else [train_loader, val_loader, test_loader],
                        split=split,
                        labels=labels,
                        num_classes=num_classes,
                        task=task,
                    )
                    self.logger.log(idx, epoch, loss, result)
                    # Cache the selected state only when embeddings are requested.
                    if save_emb and self.logger.best[idx].get('epoch', -1) == epoch:
                        best_state_dict = {k: v.detach().cpu().clone() for k, v in task_model.state_dict().items()}
                    if stopper(result):
                        print("Early Stopping at Epoch:", epoch)
                        break

            # Store the final evaluation point and resource measurements.
            runtime_sec = time.time() - t0
            peak_gpu_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2) if torch.cuda.is_available() else 0.0
            self.logger.set_run_efficiency(idx, last_eval_epoch, runtime_sec, peak_gpu_mb)

            if save_emb:
                self._save_best_embeddings(
                    task_model=task_model,
                    dataset=dataset,
                    labels=labels,
                    split=split,
                    best_state_dict=best_state_dict,
                    save_dir=emb_save_dir,
                    filename_prefix=emb_filename_prefix,
                    run_id=idx,
                    best_epoch=self.logger.best[idx].get('epoch', -1),
                )

            run_best = self.logger.best[idx]
            print(
                f"Standard {idx} Best -> "
                f"train: {run_best['train']:.2f}, "
                f"val: {run_best['val']:.2f}, "
                f"test: {run_best['test']:.2f} "
                f"(epoch {run_best.get('epoch', -1)})"
            )

        with torch.no_grad():
            print("fusion alpha:", torch.sigmoid(task_model.fusion_weight.detach()).data)
            print("fusion weight:", task_model.fusion_weight.detach().data)
        best = self.logger.get_best()
        print({
            "final/train": "{:.2f} ± {:.2f}".format(best['train']['mean'], best['train']['std']),
            "final/val": "{:.2f} ± {:.2f}".format(best['val']['mean'], best['val']['std']),
            "final/test": "{:.2f} ± {:.2f}".format(best['test']['mean'], best['test']['std']),
            "final/train_mean": best['train']['mean'],
            "final/val_mean": best['val']['mean'],
            "final/test_mean": best['test']['mean'],
            "final/train_std": best['train']['std'],
            "final/val_std": best['val']['std'],
            "final/test_std": best['test']['std'],
        })


    @torch.no_grad()
    def _save_best_embeddings(self, task_model, dataset, labels, split,
                              best_state_dict, save_dir,
                              filename_prefix, run_id, best_epoch):
        """Export best-epoch node representations for later visualization."""
        if best_state_dict is None:
            print(f"[Embedding] Run {run_id} has no selected state; skipping export.")
            return
        os.makedirs(save_dir, exist_ok=True)
        task_model.load_state_dict(best_state_dict)
        task_model.eval()

        x = dataset.x.float().to(self.device)
        edge_index = dataset.edge_index.to(self.device)
        edge_attr = dataset.edge_attr
        if edge_attr is not None:
            edge_attr = edge_attr.float().to(self.device)

        z_fusion, z_local, z_shared, expert_embeds, expert_curvatures, prompt_attn = \
            task_model.encode_with_aux_full(x, edge_index, edge_attr)

        # Match the linear-classifier evaluation path by averaging its heads.
        if hasattr(task_model, 'get_lin_logits'):
            logits = task_model.get_lin_logits(z_fusion).mean(1)
        else:
            logits = None

        run_best = self.logger.best.get(run_id, {})
        payload = {
            "raw_x": dataset.x.detach().cpu(),
            "z_fusion": z_fusion.detach().cpu(),
            "z_local": z_local.detach().cpu() if z_local is not None else None,
            "z_shared": z_shared.detach().cpu() if z_shared is not None else None,
            "logits": logits.detach().cpu() if logits is not None else None,
            "labels": labels.detach().cpu(),
            "train_mask": split["train"].detach().cpu(),
            "val_mask": split["valid"].detach().cpu(),
            "test_mask": split["test"].detach().cpu(),
            "dataset_name": self.data_tag.name,
            "client_id": self.client_id,
            "run_id": run_id,
            "best_epoch": best_epoch,
            "best_val": run_best.get('val', None),
            "best_test": run_best.get('test', None),
            "k_shot": getattr(self.args, 'k_shot', None),
            "hidden_dim": z_fusion.shape[-1],
            "fusion_alpha": float(torch.sigmoid(task_model.fusion_weight.detach()).item())
                if hasattr(task_model, 'fusion_weight') else None,
            "expert_embeds": expert_embeds.detach().cpu() if expert_embeds is not None else None,
            "expert_curvatures": expert_curvatures.detach().cpu() if expert_curvatures is not None else None,
            "prompt_attn": prompt_attn.detach().cpu() if prompt_attn is not None else None,
        }

        prefix = filename_prefix or f"client{self.client_id}_{self.data_tag.name.lower()}"
        save_path = os.path.join(save_dir, f"{prefix}_run{run_id}.pt")
        torch.save(payload, save_path)
        print(f"[Embedding] Saved run {run_id} best-epoch representations to: {save_path}")

    def get_pretrain_model(self):
        # Summarize the routed expert distribution.
        if self.routing_weights_list:
            _, p_smooth = compute_expert_weight_distribution(
                self.routing_weights_list, temperature=0.7
            )
        else:
            p_smooth = torch.ones(self.num_experts) / self.num_experts

        # Upload a low-rank factor of the expert relation matrix.
        experts = list(self.pretrain_model.moe_encoder.experts)
        if hasattr(experts[0], 'manifold'):
            curvatures = [e.manifold.k for e in experts]
        else:
            curvatures = [torch.tensor(0.0) for _ in experts]
        C = compute_expert_relation_matrix(experts, curvatures, alpha=0.5, normalize=True)
        L_upload, _ = low_rank_factor(C, rank=self.expert_relation_rank)

        local_message = {
            "num_samples": 1,
            "weight": [(name, param) for name, param in self.pretrain_model.named_parameters()
                       if name.startswith("moe_encoder.experts")
                       or (name.startswith("geo_projector") and not self.ablation_no_geo_projector)],
            "expert_distribution": p_smooth.cpu(),       # [M]
            "expert_relation_factor": L_upload.cpu(),    # [M, rank]
            "expert_usage": self.expert_usage.cpu() if self.expert_usage is not None else None,
        }

        return local_message

    def set_pretrain_model(self, global_message):
        # Only update shared expert and projector parameters.
        global_params = dict(global_message)
        with torch.no_grad():
            for local_name, local_param in self.pretrain_model.named_parameters():
                if local_name in global_params:
                    local_param.data.copy_(global_params[local_name].data.to(self.device))
def get_ft(c_task):
    task = c_task

    if task == "node_cls":
        return ft_node
    elif task == "link_pre":
        return ft_link
    elif task == "graph_cls":
        return ft_graph
    else:
        raise ValueError("Invalid Task")

def get_eval(c_task):
    task = c_task

    if task == "node_cls":
        return eval_node
    elif task == "link_pre":
        return eval_link
    elif task == "graph_cls":
        return eval_graph
    else:
        raise ValueError("Invalid Task")
