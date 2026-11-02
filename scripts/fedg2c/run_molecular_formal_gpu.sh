#!/usr/bin/env bash
# Formal single-seed molecular transfer experiment on one CUDA device.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-2026}"
DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/molecular_formal_gpu_20261023}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/cache_data}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/runs/molecular_m1_data_20261022/data}"
ROUNDS="${ROUNDS:-10}"
LOCAL_STEPS="${LOCAL_STEPS:-1000}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-20}"
ADAPTATIONS="${ADAPTATIONS:-none lora adapter prompt}"
LOG_EVERY="${LOG_EVERY:-100}"

export CUDA_VISIBLE_DEVICES
export ROUNDS LOCAL_STEPS FINETUNE_EPOCHS ADAPTATIONS
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if [[ -e "${RUN_ROOT}" ]]; then
    printf 'Use a new RUN_ROOT: %s\n' "${RUN_ROOT}" >&2
    exit 2
fi
mkdir -p "${RUN_ROOT}"
trap 'run_status=$?; printf "%s\n" "${run_status}" > "${RUN_ROOT}/exit_code"; date -u +%FT%TZ > "${RUN_ROOT}/finished_at.txt"' EXIT
printf '%s\n' "$$" > "${RUN_ROOT}/launcher.pid"
date -u +%FT%TZ > "${RUN_ROOT}/started_at.txt"
exec > >(tee "${RUN_ROOT}/console.log") 2>&1

printf 'FedG2C formal molecular experiment: %s\n' "${RUN_ROOT}"
printf 'seed=%s device=%s cuda_visible_devices=%s rounds=%s local_steps=%s target_epochs=%s\n' \
    "${SEED}" "${DEVICE}" "${CUDA_VISIBLE_DEVICES}" "${ROUNDS}" "${LOCAL_STEPS}" "${FINETUNE_EPOCHS}"

if [[ ! -f "${DATA_ROOT}/source_manifest.json" || ! -f "${DATA_ROOT}/target_manifest.json" ]]; then
    "${PYTHON_BIN}" -u -m fedg2c.prepare_molecular_data \
        --cache-root "${CACHE_ROOT}" --output "${DATA_ROOT}" --seed "${SEED}"
fi
"${PYTHON_BIN}" - "${DATA_ROOT}" "${SEED}" "${RUN_ROOT}" "${DEVICE}" <<'PY'
import json
import os
import hashlib
import platform
import shutil
import sys
from pathlib import Path
import torch
import torch_geometric

root, seed, output, device = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
report = json.loads((root / "preparation_report.json").read_text())
if report["seed"] != seed:
    raise ValueError("Prepared data seed differs from SEED")
for name in ("source_manifest.json", "target_manifest.json"):
    if json.loads((root / name).read_text())["protocol_id"] != report["protocol_id"]:
        raise ValueError("Data manifests do not match preparation report")
if not str(device).startswith('cuda') or not torch.cuda.is_available():
    raise RuntimeError("This experiment requires an available CUDA device")
expected_sources = {'molpcba': 10000, 'proteins': 1113, 'imdb_binary': 1000}
if {item['client_id']:item['graphs'] for item in report['source']['clients']} != expected_sources:
    raise ValueError("Prepared sources differ from the confirmed A protocol")
if report['target']['requested_counts'] != {'train':4096, 'valid':1024, 'test':1024}:
    raise ValueError("Prepared target differs from the confirmed A protocol")
config_paths = [Path('config') / name for name in (
    'fedg2c_edge_prediction.json', 'fedg2c_server_response.json', 'fedg2c_molecular_target.json')]
for path in config_paths:
    shutil.copy2(path, output / path.name)
shutil.copy2(root / 'preparation_report.json', output / 'preparation_report.json')
snapshot_paths = [*Path('fedg2c').rglob('*.py'), *Path('model').rglob('*.py'),
                  Path('pretrain_federated_fedg2c.py'), Path('finetune_fedg2c.py'),
                  Path('scripts/fedg2c/run_molecular_formal_gpu.sh')]
code_hashes = {}
for path in snapshot_paths:
    destination = output / 'code_snapshot' / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    code_hashes[path.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
output.joinpath('code_sha256.json').write_text(json.dumps(code_hashes, indent=2) + '\n')
effective_targets = {}
for mode in os.environ['ADAPTATIONS'].split():
    values = json.loads(Path('config/fedg2c_molecular_target.json').read_text())
    values.update(adaptation=mode, epochs=int(os.environ['FINETUNE_EPOCHS']), seed=seed)
    effective_targets[mode] = values
output.joinpath('effective_target_configs.json').write_text(json.dumps(effective_targets, indent=2) + '\n')
output.joinpath("formal_config.json").write_text(json.dumps({
    "protocol": "A_formal_single_seed",
    "seed": seed,
    "device": device,
    "cuda_visible_devices": os.environ['CUDA_VISIBLE_DEVICES'],
    "gpu": torch.cuda.get_device_name(0),
    "python": platform.python_version(),
    "torch": str(torch.__version__),
    "pyg": torch_geometric.__version__,
    "precision": "float32; AMP disabled",
    "data_root": str(root.resolve()),
    "source_order": "shuffled_cycle",
    "rounds": int(os.environ["ROUNDS"]),
    "local_steps": int(os.environ["LOCAL_STEPS"]),
    "target_epochs": int(os.environ["FINETUNE_EPOCHS"]),
    "effective_target_configs": effective_targets,
    "adaptations": os.environ["ADAPTATIONS"].split(),
    "data_protocol_id": report["protocol_id"],
    "source_manifest_sha256": hashlib.sha256((root / 'source_manifest.json').read_bytes()).hexdigest(),
    "target_manifest_sha256": hashlib.sha256((root / 'target_manifest.json').read_bytes()).hexdigest(),
    "local_epochs": 1,
    "views_per_graph": 1,
    "selection": "fixed last federated round; minimum target validation BCE; first tie",
    "configs": {path.stem:json.loads(path.read_text()) for path in config_paths},
}, indent=2) + "\n")
print(f"Using prepared data: {root} ({report['protocol_id']})", flush=True)
PY

"${PYTHON_BIN}" -u pretrain_federated_fedg2c.py \
    --source-manifest "${DATA_ROOT}/source_manifest.json" \
    --config config/fedg2c_edge_prediction.json \
    --server-config config/fedg2c_server_response.json --aggregation response \
    --rounds "${ROUNDS}" --local-epochs 1 --views-per-graph 1 \
    --local-steps "${LOCAL_STEPS}" --source-order shuffled_cycle --log-every "${LOG_EVERY}" \
    --seed "${SEED}" --threads 1 --device "${DEVICE}" \
    --output "${RUN_ROOT}/federated"

printf -v LAST_ROUND 'round_%03d' "$((ROUNDS - 1))"
read -r -a MODES <<< "${ADAPTATIONS}"
for mode in "${MODES[@]}"; do
    "${PYTHON_BIN}" -u finetune_fedg2c.py --use-new-design \
        --checkpoint "${RUN_ROOT}/federated/${LAST_ROUND}/server/shared_backbone.pt" \
        --target-manifest "${DATA_ROOT}/target_manifest.json" \
        --config config/fedg2c_molecular_target.json --adaptation "${mode}" \
        --epochs "${FINETUNE_EPOCHS}" --seed "${SEED}" --device "${DEVICE}" \
        --output "${RUN_ROOT}/target_${mode}"
    printf 'Completed: %s\n' "${RUN_ROOT}/target_${mode}/metrics.json"
done
