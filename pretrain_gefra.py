"""Run federated pre-training for Gefra."""

import argparse
import json
import os
import os.path as osp
import sys
import time

import torch
from flcore.server import Server
from flcore.client import Client
from utils.metrics_logger import PretrainMetricsLogger
from utils.basic_utils import check_path, load_fedgfm_datasets, seed_everything
from utils.config_utils import resolve_data_config_path
from utils.checkpoint_utils import build_pretrain_checkpoint_dir
from utils.partition_utils import graph_set_partition, single_graph_partition

class Logger:
    """Mirror standard output to a UTF-8 log file."""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID for single-GPU training")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--tune_dataset", type=str, default="cora")
    parser.add_argument("--project_path", type=str, default=osp.join(osp.dirname(__file__)))
    parser.add_argument(
        "--data_config",
        type=str,
        default=osp.join(osp.dirname(__file__), "config", "pretrain_config_"),
        help="Complete .json configuration path, or legacy filename prefix.",
    )
    parser.add_argument("--partition_root", type=str, default=osp.join(osp.dirname(__file__), "partitions"))
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Directory for pre-training logs; defaults to <project_path>/logs/pretrain")
    parser.add_argument("--topk", type=int, default=2, help="Number of experts to select (Top-K)")
    parser.add_argument("--round", type=int, default=100, help="Total number of communication rounds")
    parser.add_argument("--local_epochs", type=int, default=5,
                        help="Complete local epochs per communication round")
    parser.add_argument("--pretrain_batch_size", type=int, default=1024)
    parser.add_argument("--pretrain_lr", type=float, default=1e-3,
                        help="AdamW learning rate for Euclidean parameters")
    parser.add_argument("--riemannian_lr", type=float, default=1e-3,
                        help="RiemannianAdam learning rate for manifold parameters")
    parser.add_argument("--pretrain_weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_train_batches", type=int, default=None,
                        help="Optional per-epoch batch cap for smoke tests only; omitted for full local epochs")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Unified hidden dimension for the MoE and projector")
    parser.add_argument("--gnn-backbone", type=str, default="gcn", choices=["gcn", "gat", "sage"],
                        help="GNN backbone to use for clients and server")
    parser.add_argument("--ablation_fedavg", action="store_true", default=False,
                        help="Ablation: replace GW aggregation with FedAvg (uniform weights, no permutation alignment)")
    parser.add_argument("--ablation_no_geo_projector", action="store_true", default=False,
                        help="Ablation: remove RiemannianGyroKernelProjector from pre-training and federation")
    parser.add_argument("--ablation_moe", action="store_true", default=False,
                        help="Ablation: Use Euclidean MoE instead of Riemannian MoE")
    parser.add_argument("--use_strict_projector", action=argparse.BooleanOptionalAction, default=True,
                        help="Use RiemannianGyroKernelProjectorStrict (GeoMancer-faithful, ManifoldParameter-based). "
                             "Disable with --no-use_strict_projector to fall back to the loose version.")
    args = parser.parse_args()
    if args.local_epochs < 1:
        parser.error("--local_epochs must be positive")
    if args.pretrain_batch_size < 1:
        parser.error("--pretrain_batch_size must be positive")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        parser.error("--max_train_batches must be positive when provided")
    args.data_config = resolve_data_config_path(args.data_config, args.tune_dataset)
    R1 = args.round

    print("All args:", args)

    print("project path:", args.project_path)

    seed_everything(args.seed)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        torch.cuda.set_device(args.gpu_id)
        print(f"Using device: {device}")
    else:
        device = torch.device("cpu")

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_timestamp = time.strftime('%Y%m%d')
    log_dir = args.log_dir or osp.join(args.project_path, "logs", "pretrain")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"pretrain_seed{args.seed}_{log_timestamp}_{args.tune_dataset}_{R1}rounds.log"
    log_file_path = osp.join(log_dir, log_filename)

    # Mirror console output to the run-specific log file.
    sys.stdout = Logger(log_file_path)
    sys.stderr = Logger(log_file_path)

    # load global dataset
    num_clients_each_dataset, dataset_list = load_fedgfm_datasets(args)
    print("dataset list:", [dataset.name for dataset in dataset_list])

    # Load num_experts from config
    with open(args.data_config, 'r', encoding='utf-8') as file:
        config_data = json.load(file)
        args.num_experts = config_data.get("num_experts", 4)


    local_data_list = []

    # Partition each dataset with the task-specific helper.
    for it, glb_data in enumerate(dataset_list):
        print(f"loading partition for {glb_data.name.lower()}...")
        if glb_data.name.lower() in ["cora","pubmed","citeseer","wikics","arxiv","photo","computers","reddit","disease","airport"]:
            local_data_list += single_graph_partition(
                glb_data, num_partitions=num_clients_each_dataset[it], task="node_cls",
                root=args.partition_root, seed=args.seed,
            )
        elif glb_data.name.lower() in ["wn18rr", "fb15k237"]:
            local_data_list += single_graph_partition(
                glb_data, num_partitions=num_clients_each_dataset[it], task="link_pre",
                root=args.partition_root, seed=args.seed,
            )
        elif glb_data.name.lower() in ["chemhiv","chempcba", "chemblpre", "proteins", "imdb-binary"]:
            local_data_list += graph_set_partition(
                glb_data,
                num_partitions=num_clients_each_dataset[it],
                root=args.partition_root,
                mode="pretrain",
                seed=args.seed,
            )

    # free memory for g/lobal datasets
    dataset_name_list = [i.name.lower() for i in dataset_list]
    del dataset_list

    print("Length of local_data_list:", len(local_data_list))
    num_clients = sum(num_clients_each_dataset)
    print("Total number of clients:", num_clients)
    clients = []
    for client_id in range(num_clients):
        client = Client(args, client_id, local_data_list[client_id], device, backbone=args.gnn_backbone)
        clients.append(client)

    server = Server(args, device)


    # pre-training
    local_messages = {}
    metrics_logger = PretrainMetricsLogger(num_clients)

    for round_id in range(R1):
        print(f"Fed round {round_id}")
        global_msg = server.get_global_message()
        for client in clients:
            client.set_pretrain_model(global_msg)

        local_messages = {}
        for client_id, client in enumerate(clients):
            print(f"Round {round_id}, Client {client_id}")
            client.pretrain(round_id=round_id, logger=metrics_logger)
            local_messages[f"client_{client_id}"] = client.get_pretrain_model()

        server.execute(local_messages, get_gfm=False)


    # save metrics
    metrics_path = osp.join(log_dir, f"metrics_{timestamp}_{args.tune_dataset}.json")
    metrics_logger.save(metrics_path)

    model_path = build_pretrain_checkpoint_dir(
        osp.dirname(__file__), dataset_name_list, num_clients_each_dataset, args
    )
    for client_id in range(num_clients):
        client_save_path = osp.join(model_path, f"client_{client_id}")
        check_path(client_save_path)
        clients[client_id].pretrain_model.save_encoder(osp.join(client_save_path, f"moe_encoder.pt"))
        print("Client Save the model at", client_save_path)
    server_save_path = osp.join(model_path, "server")
    check_path(server_save_path)
    server.global_model.save_encoder(osp.join(server_save_path, f"moe_encoder.pt"))
    if not (args.ablation_no_geo_projector or args.ablation_moe):
        server.global_model.save_geo_projector(
            osp.join(server_save_path, "geo_projector.pt")
        )
    print("Server Save the model at", server_save_path)
