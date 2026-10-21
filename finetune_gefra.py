"""Fine-tune a pre-trained Gefra model on a target graph dataset."""

import torch
import json
import os
import os.path as osp
import argparse
import numpy as np

from utils.basic_utils import get_k_shot, load_fedgfm_datasets, seed_everything
from flcore.client import Client
import yaml
from utils.config_utils import resolve_data_config_path
from utils.checkpoint_utils import build_pretrain_checkpoint_dir
from utils.label_noise import apply_finetune_label_noise
from utils.partition_utils import graph_set_partition, single_graph_partition
import sys
import time

# Default fine-tuning output locations.
LOG_DIR = osp.join(osp.dirname(__file__), "logs", "finetune")
os.makedirs(LOG_DIR, exist_ok=True)

class Logger:
    """Mirror a text stream to a UTF-8 log file."""
    def __init__(self, log_file, terminal=None):
        self.terminal = sys.stdout if terminal is None else terminal
        self.log = open(log_file, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        if not self.log.closed:
            self.log.close()


def redirect_standard_streams(log_file):
    """Mirror standard output and error to a single run log."""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    logger = Logger(log_file, terminal=original_stdout)
    sys.stdout = logger
    sys.stderr = logger
    return original_stdout, original_stderr, logger


def restore_standard_streams(original_stdout, original_stderr, logger):
    """Restore standard streams before closing the run log."""
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID for single-GPU finetuning")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--project_path", type=str, default=osp.join(osp.dirname(__file__)))
    parser.add_argument(
        "--data_config",
        type=str,
        default=osp.join(osp.dirname(__file__), "config", "pretrain_config_"),
        help="Complete .json configuration path, or legacy filename prefix.",
    )
    parser.add_argument("--tune_dataset", type=str, default="cora")
    parser.add_argument("--partition_root", type=str, default=osp.join(osp.dirname(__file__), 'partitions'))
    parser.add_argument("--k_shot", type=int, default=-1)
    parser.add_argument("--standard", type=int, default=1,help="Number of runs for statistics (Mean & Std)")
    parser.add_argument("--topk", type=int, default=2, help="Number of experts to select (Top-K)")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Directory for fine-tuning logs; defaults to the repository log directory.")
    parser.add_argument("--freeze_moe", action=argparse.BooleanOptionalAction, default=True, help="Whether to freeze moe_encoder parameters during finetuning (use --no-freeze_moe to disable)")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Unified hidden dimension for all MoE encoders and readouts")
    parser.add_argument("--lambda_prompt", type=float, default=1e-3, help="Weight for prompt token regularization loss")
    parser.add_argument("--lambda_align", type=float, default=1e-1, help="Weight for local-shared alignment loss")
    parser.add_argument("--num_prompts", type=int, default=4, help="Number of prompt tokens in RiemannianPromptReadout")
    parser.add_argument("--lambda_curv", type=float, default=1.0, help="Curvature penalty weight in RiemannianPromptReadout")
    parser.add_argument("--ablation_learnable_weight", action="store_true", default=False,
                        help="Ablation: replace RiemannianPromptReadout with learnable scalar weights over experts")
    parser.add_argument("--ablation_fedavg", action="store_true", default=False,
                        help="Ablation: replace GW aggregation with FedAvg (uniform weights, no permutation alignment)")
    parser.add_argument("--ablation_no_geo_projector", action="store_true", default=False,
                        help="Ablation: remove RiemannianGyroKernelProjector (must match pre-training flag)")
    parser.add_argument("--ablation_moe", action="store_true", default=False,
                        help="Ablation: Use Euclidean MoE instead of Riemannian MoE")
    parser.add_argument("--use_strict_projector", action=argparse.BooleanOptionalAction, default=True,
                        help="Use RiemannianGyroKernelProjectorStrict (geometry-faithful). "
                             "Must match the flag used at pre-training time.")
    parser.add_argument("--local_encoder_only", action="store_true", default=False,
                        help="Ablation: skip MoE fusion; use only the local encoder")
    parser.add_argument("--gnn-backbone", type=str, default="gcn", choices=["gcn", "gat", "sage"],
                        help="GNN backbone to use for clients and server (default: gcn)")
    parser.add_argument("--save_embeddings", action="store_true", default=False,
                        help="Save best-epoch node embeddings for t-SNE visualization.")
    parser.add_argument("--emb_dir", type=str, default=None,
                        help="Embedding output directory; defaults to <log_dir>/embeddings/.")
    parser.add_argument(
        "--ft_label_noise", "--ft_label_attack", dest="ft_label_noise",
        action="store_true", default=False,
        help="Replace labels sampled from each local training mask before fine-tuning."
    )
    parser.add_argument(
        "--ft_label_noise_rate", "--ft_attack_label_rate", dest="ft_label_noise_rate",
        type=float, default=0.3,
        help="Fraction of local training examples selected for label replacement (default: 0.3)."
    )
    args = parser.parse_args()
    args.data_config = resolve_data_config_path(args.data_config, args.tune_dataset)
    # Resolve the output directory before redirecting standard streams.
    if args.log_dir is not None:
        log_base_dir = args.log_dir
    else:
        log_base_dir = LOG_DIR

    os.makedirs(log_base_dir, exist_ok=True)

    # Create a timestamped log file for this fine-tuning invocation.
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_filename = f"kshot{args.k_shot}_standard{args.standard}_{args.tune_dataset}_{timestamp}.log"
    log_file_path = osp.join(log_base_dir, log_filename)

    # Mirror console output to the run-specific log file.
    original_stdout, original_stderr, run_logger = redirect_standard_streams(
        log_file_path
    )

    print(f"=" * 60)
    print(f"Finetune started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_file_path}")
    print(f"Arguments: {vars(args)}")
    print(f"=" * 60)

    seed_everything(args.seed)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        torch.cuda.set_device(args.gpu_id)
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Keep the aggregate summary next to individual run logs.
    summary_file = osp.join(log_base_dir, "results_summary.log")

    # Create an embedding directory only when the optional export is enabled.
    if args.save_embeddings:
        emb_dir = args.emb_dir if args.emb_dir is not None else osp.join(log_base_dir, "embeddings", timestamp)
        os.makedirs(emb_dir, exist_ok=True)
        print(f"[Embedding] Saving representations to: {emb_dir}")
    else:
        emb_dir = None


    # Keep the legacy expert-count fallback for custom external configurations.
    with open(args.data_config, 'r', encoding='utf-8') as file:
        config_data = json.load(file)
        args.num_experts = config_data.get("num_experts", 4)

    num_clients_each_dataset, dataset_list = load_fedgfm_datasets(args, mode="finetune")


    local_data_list = []

    for it, glb_data in enumerate(dataset_list):
        print(f"loading partition for {glb_data.name.lower()}...")
        if glb_data.name.lower() in ["cora","pubmed","citeseer","wikics","arxiv","photo","computers","reddit","airport","disease"]:
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
                mode="finetune",
                seed=args.seed,
            )



    # free memory for global datasets
    del dataset_list

    if args.ft_label_noise:
        print(f"Applying training-label noise with rate={args.ft_label_noise_rate}.")
        for client_id, local_data in enumerate(local_data_list):
            selected_count = apply_finetune_label_noise(
                local_data, rate=args.ft_label_noise_rate
            )
            print(
                f"[Label noise] Client {client_id} ({local_data.name}): "
                f"selected {selected_count} training examples."
            )

    # load finetuning params for each dataset
    with open(osp.join(osp.dirname(__file__), 'config', 'finetune.yaml'), 'r') as file:
        params = yaml.safe_load(file)

    params_list = []
    for idx, data_tag in enumerate(local_data_list):
        # Normalize names before indexing the per-task fine-tuning settings.
        dataset_name_lower = data_tag.name.lower()
        if dataset_name_lower in ["cora", "pubmed", "citeseer", "arxiv", "wikics", "photo", "computers", "reddit","airport","disease"]:
            client_params = params["node"][dataset_name_lower]
        elif dataset_name_lower in ["fb15k237", "wn18rr"]:
            client_params = params["link"][data_tag.name.upper()]
        else:
            client_params = params["graph"][data_tag.name]
        params_list.append(client_params)


    # load partitions
    num_clients = sum(num_clients_each_dataset)
    print(f"Total clients for finetuning: {num_clients}")
    clients = []
    for client_id in range(num_clients):
        client = Client(args, client_id, local_data_list[client_id], device, params_list[client_id],
                        mode="finetune", backbone=args.gnn_backbone)
        clients.append(client)
    # load pretrained model
    # Reuse the source-domain configuration to locate the shared checkpoint.
    with open(args.data_config, 'r', encoding='utf-8') as file:
        dataset_config_pretrain = json.load(file)

    num_clients_each_dataset_pretrain = dataset_config_pretrain["pretrain_num_clients"]
    dataset_name_list_pretrain = dataset_config_pretrain["datasets"]
    model_path = build_pretrain_checkpoint_dir(
        osp.dirname(__file__),
        dataset_name_list_pretrain,
        num_clients_each_dataset_pretrain,
        args,
    )
    # Every target client starts from the aggregated server model. This keeps
    # fine-tuning valid when the source and target client counts differ.
    print(f"Loading pretrained model from: {model_path}")
    print(f"Pretrain: {len(dataset_name_list_pretrain)} datasets, Finetune: {num_clients} clients")

    for client_id in range(num_clients):
        clients[client_id].moe_encoder.load_state_dict(
            torch.load(
                os.path.join(model_path, "server/moe_encoder.pt"),
                map_location=device,
                weights_only=True,
            )
        )
        if not (args.ablation_no_geo_projector or args.ablation_moe):
            geo_proj_path = os.path.join(model_path, 'server/geo_projector.pt')
            if os.path.exists(geo_proj_path):
                clients[client_id].geo_projector.load_state_dict(
                    torch.load(
                        geo_proj_path,
                        map_location=device,
                        weights_only=True,
                    ),
                    strict=True,
                )
            else:
                print(f"[Warning] geo_projector.pt not found at {geo_proj_path}. Re-run pre-training to generate it.")

    # final results
    result = {}
    num_runs = args.standard

    # isolated finetune
    for client_id in range(num_clients):
        clients[client_id].data_tag = get_k_shot(clients[client_id].data_tag, k=args.k_shot)

        name = clients[client_id].data_tag.name.lower()

        clients[client_id].finetune(standard=num_runs, lambda_prompt=args.lambda_prompt,
                                    lambda_align=args.lambda_align, num_prompts=args.num_prompts, lambda_curv=args.lambda_curv,
                                    ablation_learnable_weight=args.ablation_learnable_weight,
                                    local_encoder_only=args.local_encoder_only,
                                    freeze_moe=args.freeze_moe,
                                    emb_save_dir=emb_dir,
                                    emb_filename_prefix=f"client{client_id}_{clients[client_id].data_tag.name.lower()}_{timestamp}")

        if name not in result:
            result[name] = {}

        for standard_id in range(num_runs):
            if standard_id not in result[name]:
                result[name][standard_id] = []

            if standard_id in clients[client_id].logger.best:
                single_best = clients[client_id].logger.best[standard_id]
                num_samples = clients[client_id].data_tag.test_mask.nonzero().shape[0]
                result[name][standard_id].append((single_best, num_samples))

        # Report the weighted aggregate after each target client completes.
        standard_dict = result[name]
        standard_result = []

        for values in standard_dict.values():
            if len(values) > 0:
                # Weight each target-client score by its test-set size.
                result_test = sum([pair[0]['test'] * pair[1] for pair in values]) / sum([pair[1] for pair in values])
                standard_result.append(result_test)

        if len(standard_result) > 0:
            curr_mean = np.mean(standard_result)
            curr_std = np.std(standard_result)
            _line = f"  [{name}] Client {client_id} done, current: {curr_mean:4f} ± {curr_std:.4f}%"
            print(_line)

        # Persist epoch-level metrics for the visualization utilities.
        metrics_json = {}
        for run_id, run_data in clients[client_id].logger.data.items():
            metrics_json[str(run_id)] = {
                "epochs": run_data.get("epochs", []),
                "train": run_data.get("train", []),
                "val": run_data.get("val", []),
                "test": run_data.get("test", []),
                "loss": run_data.get("loss", []),
                "best_epoch": clients[client_id].logger.best.get(run_id, {}).get("epoch", None),
                "best_val": clients[client_id].logger.best.get(run_id, {}).get("val", None),
                "best_test": clients[client_id].logger.best.get(run_id, {}).get("test", None),
                # Record timing and memory measurements when they are available.
                "stop_epoch": clients[client_id].logger.best.get(run_id, {}).get("stop_epoch", None),
                "runtime_sec": clients[client_id].logger.best.get(run_id, {}).get("runtime_sec", None),
                "peak_gpu_mb": clients[client_id].logger.best.get(run_id, {}).get("peak_gpu_mb", None),
            }
        metrics_json_path = osp.join(log_base_dir, f"metrics_client{client_id}_{name}_{timestamp}.json")
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(metrics_json, f, indent=2)
        print(f"[Metrics] Saved epoch metrics to: {metrics_json_path}")

    # Aggregate results across repeated fine-tuning runs.
    final_results = {}
    for name, standards in result.items():
        dataset_scores = []

        for std_id, values in standards.items():
            if values:
                weighted_scores = [pair[0]['test'] * pair[1] for pair in values]
                weights = [pair[1] for pair in values]
                mean_score = sum(weighted_scores) / sum(weights)
                dataset_scores.append(mean_score)

        if dataset_scores:
            final_mean = np.mean(dataset_scores)
            final_std = np.std(dataset_scores)
            final_results[name] = {"mean": final_mean, "std": final_std}

    # Print and persist the final dataset-level summary.
    print(f"\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    for name, stats in final_results.items():
        print(f"{name.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    with open(summary_file, "a", encoding="utf-8") as f:
        f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write(f"Seed: {args.seed}, K-shot: {args.k_shot}\n")
        for name, stats in final_results.items():
            f.write(f"{name.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
        f.write("-" * 50 + "\n")

    print(f"=" * 60)
    print(f"Finetune completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results summary saved to: {summary_file}")
    print(f"Full log saved to: {log_file_path}")
    print(f"=" * 60)

    restore_standard_streams(original_stdout, original_stderr, run_logger)

    sys.exit(0)
