#!/usr/bin/env bash

# Fine-tune an existing compatible checkpoint for a target selected at launch time.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

usage() {
    printf '%s\n' "Usage: bash scripts/finetune/run.sh <target> [k_shot ...]" >&2
    printf '%s\n' "Node and graph targets default to: -1 5 1. Link targets default to: -1." >&2
    print_supported_targets
}

if [[ "$#" -lt 1 ]]; then
    usage
    exit 2
fi

set_target_from_argument "$1"
shift
set_default_standard
set_k_shots_from_arguments "$@"
prepare_run
for k_shot in "${K_SHOT_VALUES[@]}"; do
    FINETUNE_LABEL="kshot${k_shot}"
    run_finetune "${k_shot}"
done
