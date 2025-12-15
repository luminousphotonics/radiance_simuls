#!/usr/bin/env bash
# run_basis_extraction_quantum.sh
# Build the per-ring response matrix A for the Quantum Board layout.

set -euo pipefail

PY=${PY:-python3}
ROOT="$(pwd)"

echo
echo "--- Basis extraction (Quantum Board) ---"

SENSOR_FILE="$ROOT/sensor_points.txt"
BASIS_DIR="$ROOT/basis_runs_quantum"
BASIS_UNIT_W="${BASIS_UNIT_W:-1.0}"

# Auto-detect ring count from quantum layout
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
RINGS=$((QB_RING_N + 1))

MODE="${MODE:-instant}"

mkdir -p "$BASIS_DIR"
rm -f "$BASIS_DIR"/basis_ring_*.txt
rm -f "$ROOT/basis_quantum.npy" "$ROOT/basis_quantum.csv" "$ROOT/basis_manifest_quantum.json"

echo "RINGS: 0..$QB_RING_N (total $RINGS)"
echo "Basis unit power per module: $BASIS_UNIT_W W"
echo "MODE: $MODE"
echo

for ((R=0; R< RINGS; R++)); do
  echo "=== Ring $R / $(($RINGS-1)) ==="
  QB_BASIS_MODE=1 \
  QB_BASIS_RING=$R \
  QB_BASIS_UNIT_W="$BASIS_UNIT_W" \
  MODE="$MODE" \
  SYM=0 \
  "$ROOT/run_simulation_quantum.sh"

  if [[ ! -f "$ROOT/ppfd_map.txt" ]]; then
    echo "ERROR: ppfd_map.txt missing after basis run for ring $R" >&2
    exit 1
  fi
  mv "$ROOT/ppfd_map.txt" "$BASIS_DIR/basis_ring_${R}.txt"
done

echo
echo "✓ All ring basis runs complete. Building A matrix..."

"$PY" - << 'PY'
import numpy as np
from pathlib import Path
import json
import os

root = Path(".")
basis_dir = root / "basis_runs_quantum"
files = sorted(basis_dir.glob("basis_ring_*.txt"),
               key=lambda p: int(p.stem.split("_")[-1]))
if not files:
    raise SystemExit("No basis_ring_*.txt files found")

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

np.save("basis_quantum.npy", A)
np.savetxt("basis_quantum.csv", A, delimiter=",", fmt="%.6f")

manifest = {
    "n_points": int(A.shape[0]),
    "n_rings": int(A.shape[1]),
    "ring_indices": list(range(A.shape[1])),
    "sensor_file": "sensor_points.txt",
    "basis_unit_w_per_module": float(os.environ.get("BASIS_UNIT_W", "1.0")),
    "layout": "quantum",
}
(Path("basis_manifest_quantum.json")).write_text(json.dumps(manifest, indent=2))

print("A shape:", A.shape)
print("Saved basis_quantum.npy and basis_quantum.csv and basis_manifest_quantum.json")
PY

echo
echo "✓ Basis matrix (Quantum Board) built."
