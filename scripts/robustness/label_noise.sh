#!/usr/bin/env bash

# Pre-train once, then sweep training-label noise for a selected target.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

NOISE_RATES="${NOISE_RATES:-0 0.1 0.2 0.3}"

usage() {
    printf '%s\n' "Usage: bash scripts/robustness/label_noise.sh <target> [k_shot]" >&2
    printf '%s\n' "The default is 5-shot for node and graph targets, or full supervision (-1) for link targets." >&2
    print_supported_targets
}

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
    usage
    exit 2
fi

set_target_from_argument "$1"
shift
set_default_standard
set_robustness_k_shot_from_arguments "$@"
prepare_run
run_pretrain
for noise_rate in ${NOISE_RATES}; do
    FINETUNE_LABEL="kshot${K_SHOT}_noise${noise_rate}"
    run_finetune "${K_SHOT}" --ft_label_noise --ft_label_noise_rate "${noise_rate}"
done
