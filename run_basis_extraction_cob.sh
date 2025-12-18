#!/usr/bin/env bash
# run_basis_extraction_cob.sh
# Build the per-ring response matrix A for the COB layout:
#   - Variables are COB watts-per-COB for each ring.
#   - Strips are proportional to each ring's COB watts (supplemental) and are included automatically.

set -euo pipefail

PY=${PY:-python3}
ROOT="$(pwd)"

echo
echo "--- Basis extraction (COB) ---"

BASIS_DIR="$ROOT/basis_runs_cob"
BASIS_UNIT_W_PER_COB="${BASIS_UNIT_W_PER_COB:-1.0}"  # watts-per-COB for the active ring

# Auto-detect ring count from COB layout
COB_RING_N="$(USE_RING_POWERS_JSON=0 RING_POWERS_JSON=/dev/null ${PY} - <<'PY'
from generate_emitters_cob import _compute_positions_from_env
try:
    _, _, meta = _compute_positions_from_env()
    rings = int(meta.get("rings", meta.get("ring_n", 0) + 1))
    print(max(0, rings - 1))
except Exception:
    print(6)
PY
)"
export COB_RING_N
RINGS=$((COB_RING_N + 1))

MODE="${MODE:-instant}"

mkdir -p "$BASIS_DIR"
rm -f "$BASIS_DIR"/basis_col_*.txt
rm -f "$ROOT/basis_cob.npy" "$ROOT/basis_cob.csv" "$ROOT/basis_manifest_cob.json"

echo "RINGS: 0..$COB_RING_N (total $RINGS)"
echo "Basis unit power (per ring, per COB): $BASIS_UNIT_W_PER_COB W/COB"
echo "MODE: $MODE"
echo

col=0
for ((R=0; R< RINGS; R++)); do
    echo "=== Ring $R / $(($RINGS-1))  col=$col ==="
    COB_BASIS_MODE=1 \
    COB_BASIS_RING=$R \
    COB_BASIS_UNIT_W_PER_COB="$BASIS_UNIT_W_PER_COB" \
    MODE="$MODE" \
    SYM=0 \
    "$ROOT/run_simulation_cob.sh"

    if [[ ! -f "$ROOT/ppfd_map.txt" ]]; then
      echo "ERROR: ppfd_map.txt missing after basis run (ring=$R)" >&2
      exit 1
    fi
    mv "$ROOT/ppfd_map.txt" "$BASIS_DIR/basis_col_${col}.txt"
    col=$((col+1))
done

echo
echo "✓ All ring basis runs complete. Building A matrix..."

"$PY" - << 'PY'
import hashlib
import json
import os
from pathlib import Path

import numpy as np

root = Path(".")
basis_dir = root / "basis_runs_cob"
files = sorted(basis_dir.glob("basis_col_*.txt"),
               key=lambda p: int(p.stem.split("_")[-1]))
if not files:
    raise SystemExit("No basis_col_*.txt files found")

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

np.save("basis_cob.npy", A)
np.savetxt("basis_cob.csv", A, delimiter=",", fmt="%.6f")

manifest = {
    "n_points": int(A.shape[0]),
    "n_rings": int(A.shape[1]),
    "ring_indices": list(range(A.shape[1])),
    "variables": "cob_w_per_cob",
    "cob_rings": int(os.environ.get("COB_RING_N", "0")) + 1,
    "basis_unit_w_per_cob": float(os.environ.get("BASIS_UNIT_W_PER_COB", "1.0")),
    "generator": "generate_emitters_cob.py",
    "generator_sha256": hashlib.sha256(Path("generate_emitters_cob.py").read_bytes()).hexdigest(),
    "emitter_env": {
        "EFF_SCALE": float(os.environ.get("EFF_SCALE", "1.0")),
        "COB_BASE_RING_N": int(os.environ.get("COB_BASE_RING_N", "6")),
        "COB_PPE_UMOL_PER_J": float(os.environ.get("COB_PPE_UMOL_PER_J", "2.76")),
        "STRIP_PPE_UMOL_PER_J": float(os.environ.get("STRIP_PPE_UMOL_PER_J", "2.76")),
        "COB_STRIP_MODE": os.environ.get("COB_STRIP_MODE", "proportional"),
        "COB_STRIP_MIN_RING": int(os.environ.get("COB_STRIP_MIN_RING", "1")),
        "COB_STRIP_W_FRACTION": float(os.environ.get("COB_STRIP_W_FRACTION", "0.05")),
        "COB_DISK_SIDES": int(os.environ.get("COB_DISK_SIDES", "16")),
    },
}
(Path("basis_manifest_cob.json")).write_text(json.dumps(manifest, indent=2))

print("A shape:", A.shape)
print("Saved basis_cob.npy and basis_cob.csv and basis_manifest_cob.json")
PY

echo
echo "✓ Basis matrix (COB) built."
