#!/usr/bin/env bash
# run_uniformity_quantum.sh
# Uniformity solver pipeline for the Quantum Board layout (independent of SMD).

set -euo pipefail

PY=${PY:-python3}
ROOT="$(pwd)"

RUN_BASIS="${RUN_BASIS:-0}"
BASIS_PATH="${BASIS_PATH:-$ROOT/basis_quantum.npy}"
OUT_JSON="${OUT_JSON:-$ROOT/ring_powers_quantum.json}"
TARGET_PPFD="${TARGET_PPFD:-1200}"
W_MIN="${W_MIN:-10}"
W_MAX="${W_MAX:-1000}"
LAMBDA_S="${LAMBDA_S:-0.0 1e-3 1e-2 1e-1 1.0 10.0 30.0}"
LAMBDA_R="${LAMBDA_R:-0.0 1e-3 1e-2 1e-1}"
LAMBDA_MEAN="${LAMBDA_MEAN:-10.0}"
USE_CHEBYSHEV="${USE_CHEBYSHEV:-1}"

echo
echo "--- Quantum Board Uniformity pipeline ---"
echo "RUN_BASIS=${RUN_BASIS}"
echo "BASIS_PATH=${BASIS_PATH}"
echo "TARGET_PPFD=${TARGET_PPFD}"
echo "LAMBDA_S=${LAMBDA_S}"
echo "LAMBDA_R=${LAMBDA_R}"
echo "LAMBDA_MEAN=${LAMBDA_MEAN}"
echo "USE_CHEBYSHEV=${USE_CHEBYSHEV}"
echo "OUT_JSON=${OUT_JSON}"
echo

# Auto-detect ring count
QB_RING_N="$(USE_RING_POWERS_JSON=0 RING_POWERS_JSON=/dev/null ${PY} - <<'PY'
from generate_emitters_quantum import _compute_positions_from_env
try:
    _, _, meta = _compute_positions_from_env()
    rings = int(meta.get("rings", meta.get("ring_n", 0) + 1))
    print(max(0, rings - 1))
except Exception:
    print(7)
PY
)"
export QB_RING_N
echo "Detected ring_n=${QB_RING_N}"

# Rebuild basis if mismatch
BASIS_MANIFEST="${ROOT}/basis_manifest_quantum.json"
if [[ -f "${BASIS_MANIFEST}" ]]; then
  BASIS_RINGS="$(python3 - <<'PY'
import json,sys
p="basis_manifest_quantum.json"
try:
    data=json.load(open(p))
    print(int(data.get("n_rings",0)))
except Exception:
    print(0)
PY
)"
else
  BASIS_RINGS=0
fi
if [[ "${BASIS_RINGS}" -ne "$((QB_RING_N+1))" ]]; then
  echo "Basis rings (${BASIS_RINGS}) differ from layout rings ($((QB_RING_N+1))); rebuilding basis."
  RUN_BASIS=1
fi

if [[ "${RUN_BASIS}" == "1" ]]; then
  echo "[1/3] Building basis matrix A (quantum)..."
  RUN_BASIS=1 "${ROOT}/run_basis_extraction_quantum.sh"
  BASIS_PATH="$ROOT/basis_quantum.npy"
fi

if [[ ! -f "${BASIS_PATH}" ]]; then
  echo "ERROR: basis file not found at ${BASIS_PATH}. Set BASIS_PATH or run with RUN_BASIS=1."
  exit 1
fi

echo "[2/3] Solving per-ring powers..."
"${PY}" "${ROOT}/solve_uniformity.py" \
  --basis "${BASIS_PATH}" \
  --target-ppfd "${TARGET_PPFD}" \
  --w-min "${W_MIN}" \
  --w-max "${W_MAX}" \
  --lambda-s ${LAMBDA_S} \
  --lambda-r ${LAMBDA_R} \
  --lambda-mean "${LAMBDA_MEAN}" \
  $( [[ "${USE_CHEBYSHEV}" == "1" ]] && echo "--use-chebyshev" ) \
  --out-json "${OUT_JSON}"

echo "[3/3] Running Radiance simulation with solved ring powers..."
USE_RING_POWERS_JSON=1 \
RING_POWERS_JSON="${OUT_JSON}" \
"${ROOT}/run_simulation_quantum.sh"

echo
echo "Pipeline complete. Solved ring powers: ${OUT_JSON}"
echo "PPFD map: ppfd_map.txt"
echo "To visualize: python3 visualize_ppfd.py --overlay quantum"
