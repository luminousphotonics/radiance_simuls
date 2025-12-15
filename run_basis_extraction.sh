#!/usr/bin/env bash
# run_basis_extraction.sh
# Build the per-ring response matrix A (PPFD per W per module per ring)
# using the existing Radiance pipeline.

set -euo pipefail

PY=${PY:-python3}
ROOT="$(pwd)"

echo
echo "--- Basis extraction: building A matrix from Radiance ---"

# Config
SENSOR_FILE="$ROOT/sensor_points.txt"   # existing grid
BASIS_DIR="$ROOT/basis_runs"
BASIS_UNIT_W="${BASIS_UNIT_W:-1.0}"     # W per module for basis
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

MODE="${MODE:-instant}"                 # instant|fast|quality

mkdir -p "$BASIS_DIR"

# Clean any old basis files
rm -f "$BASIS_DIR"/basis_ring_*.txt
rm -f "$ROOT/basis_A.npy" "$ROOT/basis_A.csv" "$ROOT/basis_manifest.json"

echo "RINGS: 0..$RING_N (total $RINGS)"
echo "Basis unit power per module: $BASIS_UNIT_W W"
echo "MODE: $MODE"
echo

for ((R=0; R< RINGS; R++)); do
  echo "=== Ring $R / $(($RINGS-1)) ==="

  # Run the normal pipeline, but:
  #  - Symmetry OFF (SYM=0)
  #  - Basis mode ON for a single ring
  SMD_BASIS_MODE=1 \
  SMD_BASIS_RING=$R \
  SMD_BASIS_UNIT_W="$BASIS_UNIT_W" \
  MODE="$MODE" \
  SYM=0 \
  "$ROOT/run_simulation_smd.sh"


  # Save ppfd_map.txt as this ring's column source
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
basis_dir = root / "basis_runs"
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
        # Sanity: same grid in same order
        if not np.allclose(coords_ref, xyz, atol=1e-6):
            raise SystemExit(f"Sensor grid mismatch in {f}")

    cols.append(ppfd)

A = np.stack(cols, axis=1)  # shape (n_points, K)

np.save("basis_A.npy", A)
np.savetxt("basis_A.csv", A, delimiter=",", fmt="%.6f")

manifest = {
    "n_points": int(A.shape[0]),
    "n_rings": int(A.shape[1]),
    "ring_indices": list(range(A.shape[1])),
    "sensor_file": "sensor_points.txt",
    "basis_unit_w_per_module": float(os.environ.get("BASIS_UNIT_W", "1.0")),
}
(Path("basis_manifest.json")).write_text(json.dumps(manifest, indent=2))

print("A shape:", A.shape)
print("Saved basis_A.npy and basis_A.csv and basis_manifest.json")
PY

echo
echo "✓ Basis matrix A built."
