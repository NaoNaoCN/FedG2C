#!/usr/bin/env bash
# ===== FedG2C ADD BEGIN A/B/C: real-data CPU engineering run =====
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-2025}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/cora_cpu_$(date +%Y%m%d_%H%M%S)}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/cache_data}"
ROUNDS="${ROUNDS:-2}"
LOCAL_STEPS="${LOCAL_STEPS:-3}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-2}"

# Set library thread counts before Python imports; every training device is CPU.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ -e "${RUN_ROOT}" ]]; then
    printf 'Use a new RUN_ROOT: %s\n' "${RUN_ROOT}" >&2
    exit 2
fi
mkdir -p "${RUN_ROOT}"
exec > >(tee "${RUN_ROOT}/console.log") 2>&1
printf 'FedG2C real Cora CPU run: %s\n' "${RUN_ROOT}"
"${PYTHON_BIN}" -u -m fedg2c.prepare_node_data \
    --cache-root "${CACHE_ROOT}" --output "${RUN_ROOT}/data" --seed "${SEED}"
"${PYTHON_BIN}" -u pretrain_federated_fedg2c.py \
    --source-manifest "${RUN_ROOT}/data/source_manifest.json" \
    --config config/fedg2c_edge_prediction.json \
    --server-config config/fedg2c_server_response.json --aggregation response \
    --rounds "${ROUNDS}" --local-epochs 1 --views-per-graph "${LOCAL_STEPS}" \
    --local-steps "${LOCAL_STEPS}" --seed "${SEED}" --threads 1 --device cpu \
    --output "${RUN_ROOT}/federated"
printf -v LAST_ROUND 'round_%03d' "$((ROUNDS - 1))"
"${PYTHON_BIN}" -u finetune_fedg2c.py --use-new-design \
    --checkpoint "${RUN_ROOT}/federated/${LAST_ROUND}/server/shared_backbone.pt" \
    --target-manifest "${RUN_ROOT}/data/target_manifest.json" \
    --config config/fedg2c_target.json --adaptation lora \
    --epochs "${FINETUNE_EPOCHS}" --seed "${SEED}" --device cpu \
    --output "${RUN_ROOT}/target"
printf 'Completed: %s\n' "${RUN_ROOT}/target/metrics.json"
# ===== FedG2C ADD END A/B/C =====
