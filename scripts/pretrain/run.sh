#!/usr/bin/env bash

# Full federated pre-training for a target selected at launch time.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

usage() {
    printf '%s\n' "Usage: bash scripts/pretrain/run.sh <target>" >&2
    print_supported_targets
}

if [[ "$#" -ne 1 ]]; then
    usage
    exit 2
fi

set_target_from_argument "$1"
prepare_run
run_pretrain
