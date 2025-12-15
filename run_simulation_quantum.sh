#!/usr/bin/env bash
# run_simulation_quantum.sh • Quantum Board PPFD (single-pass, µmol units)
# Independent pipeline so SMD remains untouched.

set -euo pipefail
echo
echo "--- Quantum Board PPFD Simulation (single-pass, PPFD units) ---"

PY=${PY:-python3}
ROOT="$(pwd)"

export RAYPATH=".:$PWD/ies_sources:${RAYPATH:-}"
rm -f "$ROOT/ppfd_map.txt"

OUTDIR="$ROOT/ies_sources"
OCT="$ROOT/quantum_scene.oct"
ROOM="$ROOT/room.rad"
SENSORS="$ROOT/sensor_points.txt"

# User knobs
MODE="${MODE:-instant}"           # instant|fast|quality|direct
SUBPATCH_GRID="${SUBPATCH_GRID:-1}"

# 0) Clean stale state & env that can trick rtrace into using an old octree
unset OCTREE || true
rm -f "$OCT"
rm -rf "$ROOT/.radcache"
mkdir -p "$ROOT/.radcache"

# 1) Build emitters (writes $OUTDIR/emitters_quantum_ALL_umol.rad)
SUBPATCH_GRID="$SUBPATCH_GRID" "$PY" "$ROOT/generate_emitters_quantum.py"

# 2) Sanity
[[ -f "$ROOM" ]] || { echo "ERROR: $ROOM not found."; exit 1; }
[[ -f "$OUTDIR/emitters_quantum_ALL_umol.rad" ]] || { echo "ERROR: $OUTDIR/emitters_quantum_ALL_umol.rad missing."; exit 1; }
[[ -f "$SENSORS" ]] || { echo "ERROR: $SENSORS not found."; exit 1; }

echo "Geometry files:"
echo "  • $ROOM"
echo "Emitters:"
echo "  • $OUTDIR/emitters_quantum_ALL_umol.rad"

# 3) Radiance presets
AMBCACHE="$ROOT/.radcache/amb"
NTHREADS="$(sysctl -n hw.ncpu 2>/dev/null || getconf _NPROCESSORS_ONLN || echo 4)"

case "$MODE" in
direct)  RT_OPTS="-ab 0 -aa 0 \
                  -u+ \
                  -dc 1.0 -dj 0.60 -ds 0.08 -dt 0 \
                  -dr 0 -lr 0 -lw 1e-5 -af $AMBCACHE" ;;
instant) RT_OPTS="-ab 3 -ad 512  -as 128 -aa 0.22 -ar 48  -dj 0.35 -ds 0.40 -dt 0.08 -dc 0.50 -dr 1 -lr 6  -lw 2e-4 -af $AMBCACHE" ;;
quality) RT_OPTS="-ab 5 -ad 2048 -as 512 -aa 0.12 -ar 96  -dj 0.65 -ds 0.20 -dt 0.03 -dc 0.85 -dr 3 -lr 12 -lw 5e-5 -af $AMBCACHE" ;;
*)       RT_OPTS="-ab 4 -ad 1024 -as 256 -aa 0.17 -ar 64  -dj 0.50 -ds 0.25 -dt 0.05 -dc 0.75 -dr 2 -lr 8  -lw 1e-4 -af $AMBCACHE" ;;
esac

if [[ "$MODE" == "direct" ]]; then NTHREADS=1; fi

# Build room.rad for current LENGTH_M, WIDTH_M, HEIGHT_M
"$PY" "$ROOT/generate_room.py" > "$ROOM"

# Build sensor grid for current room dimensions using quantum board footprint (keeps heatmap bounds consistent)
MODULE_SIDE_M="$("$PY" -c 'from generate_emitters_quantum import PATCH_MAX_M; print(PATCH_MAX_M)')" \
"$PY" "$ROOT/generate_grid.py" rtrace > "$SENSORS"

# 4) Build octree fresh
echo "Building octree..."
oconv -f "$ROOM" "$OUTDIR/emitters_quantum_ALL_umol.rad" > "$OCT"

# 5) rtrace → RGB (each channel already µmol); average for scalar PPFD
SNAKE="$ROOT/.sensors_snake.txt"

# Generate a boustrophedon reordering of the grid (order-agnostic; infers the grid)
$PY - "$SENSORS" > "$SNAKE" <<'PY'
import sys, math, collections
pts=[tuple(map(float,l.split()[:3])) for l in open(sys.argv[1]) if l.strip()]
tol=1e-6
keyy=lambda v: round(v/tol)*tol
rows=collections.defaultdict(list)
for x,y,z in pts:
    rows[keyy(y)].append((x,y,z))
ys=sorted(rows.keys())
for k in ys: rows[k].sort(key=lambda t:t[0])
snake=[]
for ri,k in enumerate(ys):
    row=rows[k]
    snake.extend(row if ri%2==0 else row[::-1])
for x,y,z in snake:
    print(f"{x:.6f} {y:.6f} {z:.6f}")
PY

# Oversample each sensor K times and average (Monte Carlo over direct samples)
OS="${OS:-4}"
SNAKE_OS="$ROOT/.sensors_snake_os.txt"
DIRS="$ROOT/.dirs_tmp.txt"

awk -v k="$OS" '{ for (i=0;i<k;i++) print $0 }' "$SNAKE" > "$SNAKE_OS"

# Build the rtrace direction file
LC_NUMERIC=C awk '{printf "%.6f %.6f %.6f 0 0 1\n",$1,$2,$3}' "$SNAKE_OS" > "$DIRS"

echo "Tracing irradiance..."
rtrace -h -I+ -n "$NTHREADS" $RT_OPTS "$OCT" < "$DIRS" > "$ROOT/.rgb_tmp.txt"

# Average every OS rows back down to one value per original sensor
paste "$SNAKE_OS" "$ROOT/.rgb_tmp.txt" \
| awk -v k="$OS" '{
    n=NF; R=$(n-2); G=$(n-1); B=$n; ppfd=(R+G+B)/3.0;
    idx = int((NR-1)/k) + 1;
    sx[idx]=$1; sy[idx]=$2; sz[idx]=$3;
    sum[idx]+=ppfd; cnt[idx]++
}
END{
  for(i=1;i<=length(sum);i++)
    printf("%.6f %.6f %.6f %.6f\n", sx[i], sy[i], sz[i], sum[i]/cnt[i]);
}' > "$ROOT/ppfd_map.txt"

rm -f "$ROOT/.rgb_tmp.txt" "$SNAKE" "$SNAKE_OS" "$DIRS"
echo "✔ Wrote $ROOT/ppfd_map.txt"

# Optional symmetry filter (near-zero cost). Toggle with SYM=1
if [[ "${SYM:-1}" = "1" ]]; then
  "${PY:-python3}" "$ROOT/symmetrize_ppfd.py" \
    --input "$ROOT/ppfd_map.txt" \
    --output "$ROOT/ppfd_map.txt" \
    --lam 1.0 \
    --tol 1e-6 \
    ${AXES_ONLY:+--axes-only} \
    --verbose
fi

echo "Done."
