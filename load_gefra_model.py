"""Build Gefra fine-tuning models from a compatible pre-training checkpoint."""

import os
import os.path as osp
import argparse
import json
import copy

import torch
import yaml

from utils.basic_utils import load_fedgfm_datasets, seed_everything
from utils.config_utils import resolve_data_config_path
from utils.checkpoint_utils import build_pretrain_checkpoint_dir
from utils.partition_utils import graph_set_partition, single_graph_partition
from flcore.client import Client


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--tune_dataset", type=str, default="cora")
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--use_strict_projector", action=argparse.BooleanOptionalAction, default=True,
                        help="Use the strict projector; must match the pre-training checkpoint.")
    parser.add_argument("--ablation_fedavg", action="store_true", default=False)
    parser.add_argument("--ablation_no_geo_projector", action="store_true", default=False)
    parser.add_argument("--ablation_moe", action="store_true", default=False)
    parser.add_argument("--gnn-backbone", type=str, default="gcn", choices=["gcn", "gat", "sage"])
    parser.add_argument("--num_prompts", type=int, default=4)
    parser.add_argument("--lambda_curv", type=float, default=1.0)
    parser.add_argument("--ckpt_dir", type=str, default=None,
                        help="Pre-training checkpoint directory; inferred from the config by default.")
    parser.add_argument("--data_config", type=str,
                        default=osp.join(osp.dirname(__file__), "config", "pretrain_config_"))
    parser.add_argument("--partition_root", type=str,
                        default=osp.join(osp.dirname(__file__), "partitions"))
    parser.add_argument("--project_path", type=str, default=osp.join(osp.dirname(__file__)))
    return parser.parse_args()


def build_model_path(args):
    """Infer the pre-training checkpoint directory used by fine-tuning."""
    if args.ckpt_dir is not None:
        return args.ckpt_dir
    config_path = resolve_data_config_path(args.data_config, args.tune_dataset)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return build_pretrain_checkpoint_dir(
        osp.dirname(__file__),
        cfg["datasets"],
        cfg["pretrain_num_clients"],
        args,
    )


def load_clients(args, device):
    """Build fine-tuning clients using the same partitioning path as the driver."""
    args.data_config = resolve_data_config_path(args.data_config, args.tune_dataset)
    config_path = args.data_config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    args.num_experts = cfg.get("num_experts", 4)

    num_clients_each_dataset, dataset_list = load_fedgfm_datasets(args, mode="finetune")

    local_data_list = []
    for it, glb_data in enumerate(dataset_list):
        name = glb_data.name.lower()
        n = num_clients_each_dataset[it]
        if name in ["cora", "pubmed", "citeseer", "wikics", "arxiv", "photo", "computers", "reddit", "airport", "disease"]:
            local_data_list += single_graph_partition(
                glb_data, n, task="node_cls", root=args.partition_root, seed=args.seed,
            )
        elif name in ["wn18rr", "fb15k237"]:
            local_data_list += single_graph_partition(
                glb_data, n, task="link_pre", root=args.partition_root, seed=args.seed,
            )
        elif name in ["chemhiv", "chempcba", "chemblpre", "proteins", "imdb-binary"]:
            local_data_list += graph_set_partition(
                glb_data,
                n,
                root=args.partition_root,
                mode="finetune",
                seed=args.seed,
            )
    del dataset_list

    with open(osp.join(osp.dirname(__file__), "config", "finetune.yaml"), "r") as f:
        ft_params = yaml.safe_load(f)

    params_list = []
    for data_tag in local_data_list:
        name = data_tag.name.lower()
        if name in ["cora", "pubmed", "citeseer", "arxiv", "wikics", "photo", "computers", "reddit", "airport", "disease"]:
            params_list.append(ft_params["node"][name])
        elif name in ["fb15k237", "wn18rr"]:
            params_list.append(ft_params["link"][data_tag.name.upper()])
        else:
            params_list.append(ft_params["graph"][data_tag.name])

    clients = []
    for client_id, data_tag in enumerate(local_data_list):
        client = Client(args, client_id, data_tag, device,
                        params_list[client_id], mode="finetune",
                        backbone=getattr(args, "gnn_backbone", "gcn"))
        clients.append(client)

    return clients


def load_pretrained_weights(clients, model_path):
    """Load the shared pre-training weights into every fine-tuning client."""
    print(f"Loading pretrained weights from: {model_path}")
    if not clients:
        return
    device = clients[0].device
    for client in clients:
        client.moe_encoder.load_state_dict(
            torch.load(
                osp.join(model_path, "server/moe_encoder.pt"),
                map_location=device,
                weights_only=True,
            )
        )
        if not (
            client.ablation_no_geo_projector
            or getattr(client.args, "ablation_moe", False)
        ):
            geo_proj_path = osp.join(model_path, "server/geo_projector.pt")
            if osp.exists(geo_proj_path):
                client.geo_projector.load_state_dict(
                    torch.load(
                        geo_proj_path,
                        map_location=device,
                        weights_only=True,
                    ),
                    strict=True,
                )
            else:
                print(f"[Warning] geo_projector.pt not found at {geo_proj_path}")


def build_task_model(client, num_prompts=4, lambda_curv=1.0):
    """Build the supported Gefra fine-tuning model on ``client.device``."""
    from model.finetuning import GefraFinetuningModel

    num_tasks = client.data_tag.num_tasks
    task = client.data_tag.task
    num_classes = num_tasks if task == "graph_cls" else client.data_tag.num_global_classes

    moe_enc_copy = copy.deepcopy(client.moe_encoder)
    geo_proj_copy = copy.deepcopy(client.geo_projector)
    if hasattr(geo_proj_copy, "bind_moe_encoder"):
        geo_proj_copy.bind_moe_encoder(moe_enc_copy)
    task_model = GefraFinetuningModel(
        local_encoder=copy.deepcopy(client.local_encoder),
        moe_encoder=moe_enc_copy,
        geo_projector=geo_proj_copy,
        num_classes=num_classes,
        input_proj=copy.deepcopy(client.input_proj),
        num_prompts=num_prompts,
        lambda_curv=lambda_curv,
    ).to(client.device)

    return task_model


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)

    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if torch.cuda.is_available() and args.gpu_id is not None
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    # Build target-domain clients.
    clients = load_clients(args, device)
    print(f"Built {len(clients)} client(s) for dataset: {args.tune_dataset}")

    # Restore the shared pre-training weights.
    model_path = build_model_path(args)
    load_pretrained_weights(clients, model_path)

    # Construct an independent task model for each client.
    task_models = []
    for client in clients:
        tm = build_task_model(
            client,
            num_prompts=args.num_prompts,
            lambda_curv=args.lambda_curv,
        )
        task_models.append(tm)
        print(f"[Client {client.client_id}] Gefra fine-tuning model built, "
              f"trainable params: {sum(p.numel() for p in tm.parameters() if p.requires_grad):,}")

    print("\nAll task models ready.")
