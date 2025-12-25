#!/usr/bin/env bash
# run_basis_extraction.sh
# Build the response matrix A for SMD:
#   - default: per-ring variables
#   - optional: per-module variables for the outer ring (SMD_OUTER_PER_MODULE=1)

set -euo pipefail

PY=${PY:-python3}
ROOT="$(pwd)"

echo
echo "--- Basis extraction: building A matrix from Radiance ---"

# Config
BASIS_DIR="$ROOT/basis_runs"
BASIS_UNIT_W="${BASIS_UNIT_W:-1.0}"     # W per module for basis
SMD_OUTER_PER_MODULE="${SMD_OUTER_PER_MODULE:-0}"

# Auto-detect ring count from layout if available (quietly)
RING_N="$(USE_RING_POWERS_JSON=0 RING_POWERS_JSON=/dev/null python3 - <<'PY'
from generate_emitters_smd import _compute_positions_from_env
try:
    _, _, meta = _compute_positions_from_env()
    rings = int(meta.get("rings", meta.get("ring_n", 0) + 1))
    print(max(0, rings - 1))
except Exception:
    print(7)
PY
)"
RINGS=$((RING_N + 1))

OUTER_RING_INDEX=""
OUTER_INDICES=()
RING_VARS="$RING_N"
if [[ "${SMD_OUTER_PER_MODULE}" == "1" ]]; then
  OUTER_INFO="$(USE_RING_POWERS_JSON=0 RING_POWERS_JSON=/dev/null python3 - <<'PY'
from generate_emitters_smd import _compute_positions_from_env
try:
    pos, _, _ = _compute_positions_from_env()
    ring_max = max(int(p.get("ring", 0)) for p in pos) if pos else 0
    outer = [str(i) for i, p in enumerate(pos) if int(p.get("ring", 0)) == ring_max]
    print(ring_max)
    print(" ".join(outer))
except Exception:
    print(0)
    print("")
PY
)"
  OUTER_RING_INDEX="$(printf "%s\n" "$OUTER_INFO" | awk 'NR==1{print $1}')"
  OUTER_INDICES_STR="$(printf "%s\n" "$OUTER_INFO" | awk 'NR==2{print $0}')"
  if [[ -n "$OUTER_INDICES_STR" ]]; then
    read -r -a OUTER_INDICES <<< "$OUTER_INDICES_STR"
  fi
  RING_VARS="$OUTER_RING_INDEX"
fi

MODE="${MODE:-instant}"                 # instant|fast|quality

mkdir -p "$BASIS_DIR"

# Clean any old basis files
rm -f "$BASIS_DIR"/basis_ring_*.txt
rm -f "$BASIS_DIR"/basis_col_*.txt
rm -f "$ROOT/basis_A.npy" "$ROOT/basis_A.csv" "$ROOT/basis_manifest.json"

echo "RINGS: 0..$RING_N (total $RINGS)"
echo "Basis unit power per module: $BASIS_UNIT_W W"
echo "MODE: $MODE"
if [[ "${SMD_OUTER_PER_MODULE}" == "1" ]]; then
  echo "Outer ring per-module basis: outer_ring=${OUTER_RING_INDEX}, outer_mods=${#OUTER_INDICES[@]}"
fi
echo

col=0
if [[ "${SMD_OUTER_PER_MODULE}" == "1" ]]; then
  for ((R=0; R< RING_VARS; R++)); do
    echo "=== Ring $R / $(($RING_VARS-1))  col=$col ==="
    SMD_BASIS_MODE=1 \
    SMD_BASIS_RING=$R \
    SMD_BASIS_UNIT_W="$BASIS_UNIT_W" \
    MODE="$MODE" \
    SYM=0 \
    "$ROOT/run_simulation_smd.sh"

    if [[ ! -f "$ROOT/ppfd_map.txt" ]]; then
      echo "ERROR: ppfd_map.txt missing after basis run for ring $R" >&2
      exit 1
    fi
    mv "$ROOT/ppfd_map.txt" "$BASIS_DIR/basis_col_${col}.txt"
    col=$((col+1))
  done

  for idx in "${OUTER_INDICES[@]}"; do
    echo "=== Outer module idx=$idx  col=$col ==="
    SMD_BASIS_MODE=1 \
    SMD_BASIS_OUTER_MODULE_IDX=$idx \
    SMD_BASIS_UNIT_W="$BASIS_UNIT_W" \
    MODE="$MODE" \
    SYM=0 \
    "$ROOT/run_simulation_smd.sh"

    if [[ ! -f "$ROOT/ppfd_map.txt" ]]; then
      echo "ERROR: ppfd_map.txt missing after basis run for outer idx=$idx" >&2
      exit 1
    fi
    mv "$ROOT/ppfd_map.txt" "$BASIS_DIR/basis_col_${col}.txt"
    col=$((col+1))
  done
else
  for ((R=0; R< RINGS; R++)); do
    echo "=== Ring $R / $(($RINGS-1)) ==="
    SMD_BASIS_MODE=1 \
    SMD_BASIS_RING=$R \
    SMD_BASIS_UNIT_W="$BASIS_UNIT_W" \
    MODE="$MODE" \
    SYM=0 \
    "$ROOT/run_simulation_smd.sh"

    if [[ ! -f "$ROOT/ppfd_map.txt" ]]; then
      echo "ERROR: ppfd_map.txt missing after basis run for ring $R" >&2
      exit 1
    fi
    mv "$ROOT/ppfd_map.txt" "$BASIS_DIR/basis_ring_${R}.txt"
  done
fi

echo
echo "✓ Basis runs complete. Building A matrix..."

export SMD_OUTER_PER_MODULE
export SMD_RING_VARS="$RING_VARS"
export SMD_OUTER_RING_INDEX="${OUTER_RING_INDEX}"
export SMD_OUTER_INDICES="${OUTER_INDICES[*]:-}"

"$PY" - << 'PY'
import numpy as np
from pathlib import Path
import json
import os
import hashlib

root = Path(".")
basis_dir = root / "basis_runs"
mode = os.environ.get("SMD_OUTER_PER_MODULE", "0") == "1"
if mode:
    files = sorted(basis_dir.glob("basis_col_*.txt"),
                   key=lambda p: int(p.stem.split("_")[-1]))
else:
    files = sorted(basis_dir.glob("basis_ring_*.txt"),
                   key=lambda p: int(p.stem.split("_")[-1]))
if not files:
    raise SystemExit("No basis files found")

cols = []
coords_ref = None

for f in files:
    data = np.loadtxt(f, dtype=float)
    if data.shape[1] != 4:
        raise SystemExit(f"{f} has unexpected shape {data.shape}, expected 4 cols (x,y,z,ppfd)")
    xyz = data[:, :3]
    ppfd = data[:, 3]

    if coords_ref is None:
        coords_ref = xyz
    else:
        if not np.allclose(coords_ref, xyz, atol=1e-6):
            raise SystemExit(f"Sensor grid mismatch in {f}")

    cols.append(ppfd)

A = np.stack(cols, axis=1)

np.save("basis_A.npy", A)
np.savetxt("basis_A.csv", A, delimiter=",", fmt="%.6f")

manifest = {
    "n_points": int(A.shape[0]),
    "n_rings": int(A.shape[1]),
    "ring_indices": list(range(A.shape[1])),
    "sensor_file": "sensor_points.txt",
    "basis_unit_w_per_module": float(os.environ.get("BASIS_UNIT_W", "1.0")),
    "generator": "generate_emitters_smd.py",
    "generator_sha256": hashlib.sha256(Path("generate_emitters_smd.py").read_bytes()).hexdigest(),
    "emitter_env": {
        "SMD_BASE_RING_N": int(os.environ.get("SMD_BASE_RING_N", "7")),
        "SMD_PERIM_GAP_FILL": int(os.environ.get("SMD_PERIM_GAP_FILL", "0")),
        "SMD_OUTER_PER_MODULE": int(os.environ.get("SMD_OUTER_PER_MODULE", "0")),
        "SMD_OUTER_OPTICS": int(os.environ.get("SMD_OUTER_OPTICS", "0")),
        "SMD_OUTER_FWHM_DEG": float(os.environ.get("SMD_OUTER_FWHM_DEG", "40.0")),
        "DROOP_P_NOM": float(os.environ.get("DROOP_P_NOM", "100.0")),
        "DROOP_K": float(os.environ.get("DROOP_K", "0.12")),
        "MODULE_SIDE_M": float(os.environ.get("MODULE_SIDE_M", "0.127")),
        "SMD_FIXTURE_ANGLE_IN": float(os.environ.get("SMD_FIXTURE_ANGLE_IN", "0.125")),
        "MARGIN_IN": float(os.environ.get("MARGIN_IN", "0.0")),
        "PPE_IS_SYSTEM": int(os.environ.get("PPE_IS_SYSTEM", "0").strip() != "0"),
        "DRIVER_EFF": float(os.environ.get("DRIVER_EFF", "0.95")),
        "THERMAL_EFF": float(os.environ.get("THERMAL_EFF", "0.92")),
        "BOARD_OPT_EFF": float(os.environ.get("BOARD_OPT_EFF", "0.95")),
        "WIRING_EFF": float(os.environ.get("WIRING_EFF", "0.99")),
        "EFF_SCALE": float(os.environ.get("EFF_SCALE", "1.0")),
        "OPTICS": os.environ.get("OPTICS", "none").strip(),
        "SUBPATCH_GRID": int(os.environ.get("SUBPATCH_GRID", "1")),
    },
}

if mode:
    ring_vars = int(os.environ.get("SMD_RING_VARS", "0") or "0")
    outer_idx_str = os.environ.get("SMD_OUTER_INDICES", "").strip()
    outer_idx = [int(x) for x in outer_idx_str.split()] if outer_idx_str else []
    outer_ring = int(os.environ.get("SMD_OUTER_RING_INDEX", "0") or "0")
    manifest.update({
        "variables": "ring_plus_outer_modules",
        "ring_indices": list(range(ring_vars)),
        "outer_ring_index": outer_ring,
        "outer_ring_indices": outer_idx,
        "variable_groups": {"rings": ring_vars, "outer_modules": len(outer_idx)},
    })

(Path("basis_manifest.json")).write_text(json.dumps(manifest, indent=2))

print("A shape:", A.shape)
print("Saved basis_A.npy and basis_A.csv and basis_manifest.json")
PY

echo
echo "✓ Basis matrix A built."
