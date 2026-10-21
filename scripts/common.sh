#!/usr/bin/env bash

# Shared setup for the checked-in experiment launchers. Source this file from a
# task-specific script so pre-training and fine-tuning use identical model flags.
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    printf '%s\n' "Source scripts/common.sh from an experiment launcher instead of running it directly." >&2
    exit 2
fi

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-2025}"
TARGET="${TARGET:-}"
TARGET_TASK=""
ROUND="${ROUND:-100}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
STANDARD="${STANDARD:-}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
TOPK="${TOPK:-2}"
BACKBONE="${BACKBONE:-gcn}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-1024}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-}"
PROJECT_PATH="${PROJECT_PATH:-${REPO_ROOT}}"
PARTITION_ROOT="${PARTITION_ROOT:-${REPO_ROOT}/partitions}"
DATA_CONFIG="${DATA_CONFIG:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-}"
STRICT_PROJECTOR="${STRICT_PROJECTOR:-1}"
ABLATION_FEDAVG="${ABLATION_FEDAVG:-0}"
ABLATION_NO_GEO_PROJECTOR="${ABLATION_NO_GEO_PROJECTOR:-0}"
ABLATION_MOE="${ABLATION_MOE:-0}"

MODEL_FLAGS=()
DATA_CONFIG_FLAGS=()
K_SHOT_VALUES=()


print_command() {
    printf 'Running:'
    printf ' %q' "$@"
    printf '\n'
}


die() {
    printf '%s\n' "$1" >&2
    exit 2
}


normalize_target() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}


print_supported_targets() {
    local config_path filename target_name
    printf '%s' "Supported targets:" >&2
    for config_path in "${REPO_ROOT}/config/pretrain_config_"*.json; do
        [[ -f "${config_path}" ]] || continue
        filename="${config_path##*/}"
        target_name="${filename#pretrain_config_}"
        target_name="${target_name%.json}"
        printf ' %s' "${target_name}" >&2
    done
    printf '\n' >&2
}


infer_target_task() {
    local line section="" candidate
    local config_path="${REPO_ROOT}/config/finetune.yaml"

    while IFS= read -r line || [[ -n "${line}" ]]; do
        if [[ "${line}" =~ ^([a-z]+):[[:space:]]*$ ]]; then
            section="${BASH_REMATCH[1]}"
            continue
        fi
        if [[ "${line}" =~ ^[[:space:]][[:space:]]([^:]+):[[:space:]]*$ ]]; then
            candidate="$(normalize_target "${BASH_REMATCH[1]}")"
            if [[ "${candidate}" != "${TARGET}" ]]; then
                continue
            fi
            case "${section}" in
                node) TARGET_TASK="node_cls" ;;
                link) TARGET_TASK="link_pre" ;;
                graph) TARGET_TASK="graph_cls" ;;
                *) die "Could not determine the task type for target '${TARGET}'." ;;
            esac
            return
        fi
    done < "${config_path}"

    die "Target '${TARGET}' is not listed in ${config_path}."
}


set_target_from_argument() {
    if [[ "$#" -ne 1 || -z "$1" ]]; then
        die "A target dataset is required."
    fi

    TARGET="$(normalize_target "$1")"
    if [[ ! "${TARGET}" =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
        print_supported_targets
        die "Invalid target dataset '${TARGET}'."
    fi

    local default_config="${REPO_ROOT}/config/pretrain_config_${TARGET}.json"
    if [[ ! -f "${default_config}" ]]; then
        print_supported_targets
        die "No checked-in experiment configuration exists for target '${TARGET}'."
    fi
    if [[ -z "${DATA_CONFIG}" ]]; then
        DATA_CONFIG="${default_config}"
    elif [[ ! -f "${DATA_CONFIG}" ]]; then
        die "DATA_CONFIG does not exist: ${DATA_CONFIG}"
    fi

    infer_target_task
}


set_default_standard() {
    if [[ -n "${STANDARD}" ]]; then
        return
    fi
    if [[ "${TARGET_TASK}" == "link_pre" ]]; then
        STANDARD=1
    else
        STANDARD=10
    fi
}


default_k_shots() {
    if [[ "${TARGET_TASK}" == "link_pre" ]]; then
        printf '%s\n' "-1"
    else
        printf '%s\n' "-1 5 1"
    fi
}


validate_k_shot() {
    local k_shot="$1"
    if [[ ! "${k_shot}" =~ ^-?[0-9]+$ ]] || (( k_shot < -1 )); then
        die "k-shot must be -1, 0, or a positive integer; got '${k_shot}'."
    fi
    if [[ "${TARGET_TASK}" == "link_pre" && "${k_shot}" != "-1" && "${k_shot}" != "0" ]]; then
        die "Link prediction target '${TARGET}' only supports k-shot values -1 or 0."
    fi
}


set_k_shots_from_arguments() {
    if [[ "$#" -gt 0 ]]; then
        K_SHOTS="$*"
    elif [[ -z "${K_SHOTS:-}" ]]; then
        K_SHOTS="$(default_k_shots)"
    fi

    read -r -a K_SHOT_VALUES <<< "${K_SHOTS}"
    if [[ "${#K_SHOT_VALUES[@]}" -eq 0 ]]; then
        die "At least one k-shot value is required."
    fi
    local k_shot
    for k_shot in "${K_SHOT_VALUES[@]}"; do
        validate_k_shot "${k_shot}"
    done
}


set_robustness_k_shot_from_arguments() {
    if [[ "$#" -gt 1 ]]; then
        die "Label-noise launcher accepts at most one k-shot value."
    fi
    if [[ "$#" -eq 1 ]]; then
        K_SHOT="$1"
    elif [[ -z "${K_SHOT:-}" ]]; then
        if [[ "${TARGET_TASK}" == "link_pre" ]]; then
            K_SHOT=-1
        else
            K_SHOT=5
        fi
    fi
    validate_k_shot "${K_SHOT}"
}


require_toggle() {
    local name="$1"
    local value="${!name}"
    case "${value}" in
        0|1) ;;
        *)
            printf '%s\n' "${name} must be 0 or 1, got: ${value}" >&2
            exit 2
            ;;
    esac
}


build_model_flags() {
    case "${BACKBONE}" in
        gcn|gat|sage) ;;
        *)
            printf '%s\n' "BACKBONE must be one of: gcn, gat, sage." >&2
            exit 2
            ;;
    esac

    require_toggle STRICT_PROJECTOR
    require_toggle ABLATION_FEDAVG
    require_toggle ABLATION_NO_GEO_PROJECTOR
    require_toggle ABLATION_MOE

    MODEL_FLAGS=(
        --hidden_dim "${HIDDEN_DIM}"
        --topk "${TOPK}"
        --gnn-backbone "${BACKBONE}"
    )

    if [[ "${STRICT_PROJECTOR}" == "1" ]]; then
        MODEL_FLAGS+=(--use_strict_projector)
    else
        MODEL_FLAGS+=(--no-use_strict_projector)
    fi
    if [[ "${ABLATION_FEDAVG}" == "1" ]]; then
        MODEL_FLAGS+=(--ablation_fedavg)
    fi
    if [[ "${ABLATION_NO_GEO_PROJECTOR}" == "1" ]]; then
        MODEL_FLAGS+=(--ablation_no_geo_projector)
    fi
    if [[ "${ABLATION_MOE}" == "1" ]]; then
        MODEL_FLAGS+=(--ablation_moe)
    fi
}


build_data_config_flags() {
    DATA_CONFIG_FLAGS=()
    if [[ -n "${DATA_CONFIG}" ]]; then
        DATA_CONFIG_FLAGS=(--data_config "${DATA_CONFIG}")
    fi
}


prepare_run() {
    if [[ -z "${TARGET}" || -z "${TARGET_TASK}" ]]; then
        die "Call set_target_from_argument before prepare_run."
    fi
    build_model_flags
    build_data_config_flags
    if [[ -z "${RUN_ROOT}" ]]; then
        RUN_ROOT="${REPO_ROOT}/logs/experiments/${TARGET}/${RUN_ID}"
    fi
    mkdir -p "${RUN_ROOT}/pretrain" "${RUN_ROOT}/finetune"
    printf '%s\n' "Experiment logs: ${RUN_ROOT}"
    printf '%s\n' "Target=${TARGET}, task=${TARGET_TASK}, backbone=${BACKBONE}, topk=${TOPK}, hidden_dim=${HIDDEN_DIM}, seed=${SEED}"
}


run_pretrain() {
    local -a command=(
        "${PYTHON_BIN}" "${REPO_ROOT}/pretrain_gefra.py"
        --tune_dataset "${TARGET}"
        --gpu_id "${GPU_ID}"
        --seed "${SEED}"
        --project_path "${PROJECT_PATH}"
        --partition_root "${PARTITION_ROOT}"
        --log_dir "${RUN_ROOT}/pretrain"
        --round "${ROUND}"
        --local_epochs "${LOCAL_EPOCHS}"
        --pretrain_batch_size "${PRETRAIN_BATCH_SIZE}"
        "${MODEL_FLAGS[@]}"
        "${DATA_CONFIG_FLAGS[@]}"
    )
    if [[ -n "${MAX_TRAIN_BATCHES}" ]]; then
        command+=(--max_train_batches "${MAX_TRAIN_BATCHES}")
    fi
    print_command "${command[@]}"
    "${command[@]}"
}


run_finetune() {
    local k_shot="$1"
    shift
    local label="${FINETUNE_LABEL:-kshot${k_shot}}"
    local -a command=(
        "${PYTHON_BIN}" "${REPO_ROOT}/finetune_gefra.py"
        --tune_dataset "${TARGET}"
        --gpu_id "${GPU_ID}"
        --seed "${SEED}"
        --project_path "${PROJECT_PATH}"
        --partition_root "${PARTITION_ROOT}"
        --log_dir "${RUN_ROOT}/finetune/${label}"
        --k_shot "${k_shot}"
        --standard "${STANDARD}"
        "${MODEL_FLAGS[@]}"
        "${DATA_CONFIG_FLAGS[@]}"
        "$@"
    )
    print_command "${command[@]}"
    "${command[@]}"
}
