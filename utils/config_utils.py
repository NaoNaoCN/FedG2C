"""Small, dependency-free helpers for experiment configuration files."""

import os.path as osp


def resolve_data_config_path(config_path_or_prefix: str, dataset_name: str) -> str:
    """Resolve a JSON configuration path while preserving the legacy prefix CLI.

    ``--data_config`` originally accepted only a filename prefix and always
    appended ``<dataset_name>.json``.  Public experiment commands may instead
    provide a complete JSON file path, which must be used unchanged.
    """
    if osp.splitext(config_path_or_prefix)[1].lower() == ".json":
        return config_path_or_prefix
    return f"{config_path_or_prefix}{dataset_name}.json"


def select_dataset_config(dataset_config: dict, mode: str):
    """Return the active dataset/client/weight lists from a public JSON config.

    Earlier development schemas also carried task labels, runtime roots,
    and per-dataset weights.  The native PyG loader determines task labels
    itself, derives runtime roots from command-line arguments, and uses equal
    weights when a legacy ``weights`` list is not present.
    """
    if mode == "pretrain":
        datasets_key = "datasets"
        clients_key = "pretrain_num_clients"
    elif mode == "finetune":
        datasets_key = "finetune_datasets"
        clients_key = "finetune_num_clients"
    else:
        raise ValueError(f"Unknown dataset loading mode: {mode!r}.")

    try:
        datasets = dataset_config[datasets_key]
        num_clients_each_dataset = dataset_config[clients_key]
    except KeyError as error:
        raise ValueError(
            f"Configuration is missing required field {error.args[0]!r} for {mode}."
        ) from error

    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"Configuration field {datasets_key!r} must be a non-empty list.")
    if not isinstance(num_clients_each_dataset, list):
        raise ValueError(f"Configuration field {clients_key!r} must be a list.")
    if len(num_clients_each_dataset) != len(datasets):
        raise ValueError(
            f"Configuration field {clients_key!r} must have one value per "
            f"dataset ({len(datasets)} expected, got {len(num_clients_each_dataset)})."
        )
    if any(not isinstance(count, int) or count < 1 for count in num_clients_each_dataset):
        raise ValueError(f"Configuration field {clients_key!r} must contain positive integers.")

    weights = dataset_config.get("weights", [1] * len(datasets))
    if not isinstance(weights, list) or len(weights) != len(datasets):
        actual = len(weights) if isinstance(weights, list) else type(weights).__name__
        raise ValueError(
            "Configuration field 'weights' must have one value per dataset "
            f"({len(datasets)} expected, got {actual})."
        )

    return num_clients_each_dataset, datasets, weights
