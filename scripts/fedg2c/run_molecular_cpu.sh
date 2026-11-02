#!/usr/bin/env bash
# ===== FedG2C ADD BEGIN M1: real molecular CPU engineering run =====
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-2026}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/molecular_m1_cpu_$(date +%Y%m%d_%H%M%S)}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/cache_data}"
ROUNDS="${ROUNDS:-2}"
LOCAL_STEPS="${LOCAL_STEPS:-3}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-2}"
ADAPTATIONS="${ADAPTATIONS:-none lora adapter prompt}"
DATA_ROOT="${DATA_ROOT:-${RUN_ROOT}/data}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if [[ -e "${RUN_ROOT}" ]]; then
    printf 'Use a new RUN_ROOT: %s\n' "${RUN_ROOT}" >&2
    exit 2
fi
mkdir -p "${RUN_ROOT}"
exec > >(tee "${RUN_ROOT}/console.log") 2>&1
printf 'FedG2C M1 molecular CPU run: %s\n' "${RUN_ROOT}"

if [[ ! -f "${DATA_ROOT}/source_manifest.json" || ! -f "${DATA_ROOT}/target_manifest.json" ]]; then
    "${PYTHON_BIN}" -u -m fedg2c.prepare_molecular_data \
        --cache-root "${CACHE_ROOT}" --output "${DATA_ROOT}" --seed "${SEED}"
fi
"${PYTHON_BIN}" - "${DATA_ROOT}" "${SEED}" <<'PY'
import json
import sys
from pathlib import Path

root, seed = Path(sys.argv[1]), int(sys.argv[2])
report = json.loads((root / "preparation_report.json").read_text())
if report["seed"] != seed:
    raise ValueError("Prepared data seed differs from SEED; use matching data or a new DATA_ROOT")
for name in ("source_manifest.json", "target_manifest.json"):
    if json.loads((root / name).read_text())["protocol_id"] != report["protocol_id"]:
        raise ValueError("Data manifests do not match their preparation report")
print(f"Using prepared data: {root} ({report['protocol_id']})", flush=True)
PY
"${PYTHON_BIN}" -u pretrain_federated_fedg2c.py \
    --source-manifest "${DATA_ROOT}/source_manifest.json" \
    --config config/fedg2c_edge_prediction.json \
    --server-config config/fedg2c_server_response.json --aggregation response \
    --rounds "${ROUNDS}" --local-epochs 1 --views-per-graph 1 \
    --local-steps "${LOCAL_STEPS}" --seed "${SEED}" --threads 1 --device cpu \
    --output "${RUN_ROOT}/federated"
printf -v LAST_ROUND 'round_%03d' "$((ROUNDS - 1))"
read -r -a MODES <<< "${ADAPTATIONS}"
for mode in "${MODES[@]}"; do
    "${PYTHON_BIN}" -u finetune_fedg2c.py --use-new-design \
        --checkpoint "${RUN_ROOT}/federated/${LAST_ROUND}/server/shared_backbone.pt" \
        --target-manifest "${DATA_ROOT}/target_manifest.json" \
        --config config/fedg2c_molecular_target.json --adaptation "${mode}" \
        --epochs "${FINETUNE_EPOCHS}" --seed "${SEED}" --device cpu \
        --output "${RUN_ROOT}/target_${mode}"
    printf 'Completed: %s\n' "${RUN_ROOT}/target_${mode}/metrics.json"
done
# ===== FedG2C ADD END M1 =====
