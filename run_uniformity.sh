#!/usr/bin/env bash
# run_uniformity.sh
# One-command workflow:
#   1) (Optional) build basis matrix A (RUN_BASIS=1)
#   2) Solve per-ring powers for target PPFD
#   3) Run Radiance with the solved ring powers injected

set -euo pipefail

PY=${PY:-python3}
ROOT="$(pwd)"

RUN_BASIS="${RUN_BASIS:-0}"           # 1 to rebuild basis via run_basis_extraction.sh
BASIS_PATH="${BASIS_PATH:-$ROOT/basis_A.npy}"
OUT_JSON="${OUT_JSON:-$ROOT/ring_powers_optimized.json}"
TARGET_PPFD="${TARGET_PPFD:-1200}"
W_MIN="${W_MIN:-10}"
W_MAX="${W_MAX:-100}"
LAMBDA_S="${LAMBDA_S:-0.0 1e-3 1e-2 1e-1 1.0 10.0 30.0}"
LAMBDA_R="${LAMBDA_R:-0.0 1e-3 1e-2 1e-1}"
LAMBDA_MEAN="${LAMBDA_MEAN:-10.0}"
USE_CHEBYSHEV="${USE_CHEBYSHEV:-1}"

echo
echo "--- Uniformity pipeline ---"
echo "RUN_BASIS=${RUN_BASIS}"
echo "BASIS_PATH=${BASIS_PATH}"
echo "TARGET_PPFD=${TARGET_PPFD}"
echo "LAMBDA_S=${LAMBDA_S}"
echo "LAMBDA_R=${LAMBDA_R}"
echo "LAMBDA_MEAN=${LAMBDA_MEAN}"
echo "USE_CHEBYSHEV=${USE_CHEBYSHEV}"
echo "OUT_JSON=${OUT_JSON}"
echo

# Auto-detect ring count for layout and propagate to downstream scripts.
# Suppress ring-power override chatter during detection.
SMD_RING_N="$(USE_RING_POWERS_JSON=0 RING_POWERS_JSON=/dev/null ${PY} - <<'PY'
from generate_emitters_smd import _compute_positions_from_env
try:
    _, _, meta = _compute_positions_from_env()
    rings = int(meta.get("rings", meta.get("ring_n", 0) + 1))
    print(max(0, rings - 1))
except Exception:
    print(7)
PY
)"
export SMD_RING_N
echo "Detected ring_n=${SMD_RING_N}"

# Rebuild basis if ring count mismatches existing basis
BASIS_MANIFEST="${ROOT}/basis_manifest.json"
if [[ -f "${BASIS_MANIFEST}" ]]; then
  BASIS_RINGS="$(python3 - <<'PY'
import json,sys
p="basis_manifest.json"
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

# Rebuild basis if key emitter settings changed (generator file or env knobs).
if [[ -f "${BASIS_MANIFEST}" ]]; then
  if ! python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

manifest = json.loads(Path("basis_manifest.json").read_text())
if "generator_sha256" not in manifest or "emitter_env" not in manifest:
    raise SystemExit("basis manifest missing emitter fingerprint (legacy basis)")
want_sha = hashlib.sha256(Path("generate_emitters_smd.py").read_bytes()).hexdigest()
have_sha = manifest.get("generator_sha256", "")
if have_sha and have_sha != want_sha:
    raise SystemExit("basis generator changed")

have_env = manifest.get("emitter_env") or {}
want_env = {
    "PPE_IS_SYSTEM": int(os.environ.get("PPE_IS_SYSTEM", "1").strip() != "0"),
    "DRIVER_EFF": float(os.environ.get("DRIVER_EFF", "0.95")),
    "THERMAL_EFF": float(os.environ.get("THERMAL_EFF", "0.92")),
    "BOARD_OPT_EFF": float(os.environ.get("BOARD_OPT_EFF", "0.95")),
    "WIRING_EFF": float(os.environ.get("WIRING_EFF", "0.99")),
    "EFF_SCALE": float(os.environ.get("EFF_SCALE", "1.0")),
    "OPTICS": os.environ.get("OPTICS", "none").strip(),
    "SUBPATCH_GRID": int(os.environ.get("SUBPATCH_GRID", "1")),
}

for k, v in want_env.items():
    if k not in have_env:
        continue
    if have_env[k] != v:
        raise SystemExit(f"basis env changed: {k} {have_env[k]} -> {v}")
PY
  then
    echo "Basis manifest differs from current emitter settings; rebuilding basis."
    RUN_BASIS=1
  fi
fi
if [[ "${BASIS_RINGS}" -ne "$((SMD_RING_N+1))" ]]; then
  echo "Basis rings (${BASIS_RINGS}) differ from layout rings ($((SMD_RING_N+1))); rebuilding basis."
  RUN_BASIS=1
fi

if [[ "${RUN_BASIS}" == "1" ]]; then
  echo "[1/3] Building basis matrix A..."
  RUN_BASIS=1 "${ROOT}/run_basis_extraction.sh"
  BASIS_PATH="$ROOT/basis_A.npy"
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
"${ROOT}/run_simulation_smd.sh"

echo
echo "Pipeline complete. Solved ring powers: ${OUT_JSON}"
echo "PPFD map: ppfd_map.txt"
echo "To visualize: python3 visualize_ppfd.py --overlay smd"
