#!/usr/bin/env bash
set -euo pipefail
task_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$task_repo"
# The explicit target limits this invocation to the Cora LODO pipeline.
exec "${PYTHON:-python}" -u -m fedg2c.gefra_loo_matrix \
  --config "${1:-config/fedg2c_gefra_loo_v1_cora_cuda.json}" --targets cora
