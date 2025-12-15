#!/usr/bin/env bash
# run_simulation_spydr3.sh • v1.0
# Single-pass PPFD (µmol units) for Fluence SPYDR 3 3h47 stand-in (Lambertian bars).
# Modes: instant | fast | quality | direct

set -euo pipefail
echo
echo "--- SPYDR3 PPFD Simulation (single-pass, PPFD units) ---"

PY=${PY:-python3}
ROOT="$(pwd)"

export RAYPATH=".:$PWD/ies_sources:${RAYPATH:-}"
rm -f "$ROOT/ppfd_map.txt"

OUTDIR="$ROOT/ies_sources"
OCT="$ROOT/spydr_scene.oct"
ROOM="$ROOT/room.rad"
SENSORS="$ROOT/sensor_points.txt"

# User knobs (defaults chosen to match your pipeline)
MODE="${MODE:-quality}"            # instant|fast|quality|direct
SUBPATCH_GRID="${SUBPATCH_GRID:-1}"
OS="${OS:-4}"                      # rtrace oversamples per sensor
SYM="${SYM:-1}"                    # 1=apply symmetry filter, 0=skip
AXES_ONLY="${AXES_ONLY:-}"         # if set (non-empty), only x/y flips in symmetry

# Target / auto-dimming
TARGET_PPFD="${TARGET_PPFD:-}"     # if set and AUTO_DIM=1, globally scale fixture output to hit this mean PPFD
AUTO_DIM="${AUTO_DIM:-1}"          # 1=auto-dim to TARGET_PPFD, 0=disable (run at EFF_SCALE)

# SPYDR parameters (override via env if desired)
#   PPF (µmol/s) and geometry are read by the generator; pass-through here.
SPYDR_PPF="${SPYDR_PPF:-2200}"
SPYDR_Z_M="${SPYDR_Z_M:-0.4572}"   # ~18 in
NX="${NX:-1}"                       # fixture grid in room (1x1 centers at origin)
NY="${NY:-1}"
EFF_SCALE="${EFF_SCALE:-1.0}"       # dimmer fraction (0..1); capped at 1.0 (no overdrive)
SPYDR_PPE_UMOL_PER_J="${SPYDR_PPE_UMOL_PER_J:-2.7}"  # for electrical watts estimate: W ≈ PPF / PPE

clamp_0_1() {
  LC_NUMERIC=C awk -v x="$1" 'BEGIN{
    if (x+0 != x) x=1.0;
    if (x < 0.0) x=0.0;
    if (x > 1.0) x=1.0;
    printf("%.8f\n", x);
  }'
}

EFF_SCALE_CLAMPED="$(clamp_0_1 "$EFF_SCALE")"
if LC_NUMERIC=C awk -v a="$EFF_SCALE" -v b="$EFF_SCALE_CLAMPED" 'BEGIN{exit !(a!=b)}'; then
  echo "NOTE: clamped EFF_SCALE from $EFF_SCALE to $EFF_SCALE_CLAMPED (no overdrive)."
fi
EFF_SCALE="$EFF_SCALE_CLAMPED"

# 0) Clean stale state & env that can trick rtrace into using an old octree
unset OCTREE || true
rm -f "$OCT"
rm -rf "$ROOT/.radcache"
mkdir -p "$ROOT/.radcache"

# 1) Build room.rad for current LENGTH/WIDTH and regenerate sensor grid to match
"$PY" "$ROOT/generate_room.py" > "$ROOM"
# Use the bar width as MODULE_SIDE_M so the sensor grid matches the SPYDR footprint
MODULE_SIDE_M_VAL="$("$PY" -c 'import os; w_in=float(os.getenv("SPYDR_BAR_WIDTH_IN","3")); print(w_in*0.0254)')"
MODULE_SIDE_M="$MODULE_SIDE_M_VAL" "$PY" "$ROOT/generate_grid.py" rtrace > "$SENSORS"

# 2) Sanity
[[ -f "$ROOM" ]] || { echo "ERROR: $ROOM not found."; exit 1; }
[[ -f "$SENSORS" ]] || { echo "ERROR: $SENSORS not found."; exit 1; }

echo "Geometry files:"
echo "  • $ROOM"

# 3) Radiance presets
NTHREADS="$(sysctl -n hw.ncpu 2>/dev/null || getconf _NPROCESSORS_ONLN || echo 4)"

case "$MODE" in
direct)  RT_OPTS_BASE="-ab 0 -aa 0 \
	                  -u+ \
	                  -dc 1.0 -dj 0.60 -ds 0.08 -dt 0 \
	                  -dr 0 -lr 0 -lw 1e-5" ;;
instant) RT_OPTS_BASE="-ab 3 -ad 512  -as 128 -aa 0.22 -ar 48  -dj 0.35 -ds 0.40 -dt 0.08 -dc 0.50 -dr 1 -lr 6  -lw 2e-4" ;;
quality) RT_OPTS_BASE="-ab 5 -ad 2048 -as 512 -aa 0.12 -ar 96  -dj 0.65 -ds 0.20 -dt 0.03 -dc 0.85 -dr 3 -lr 12 -lw 5e-5" ;;
*)       RT_OPTS_BASE="-ab 4 -ad 1024 -as 256 -aa 0.17 -ar 64  -dj 0.50 -ds 0.25 -dt 0.05 -dc 0.75 -dr 2 -lr 8  -lw 1e-4" ;;
esac

if [[ "$MODE" == "direct" ]]; then NTHREADS=1; fi

ppfd_mean() {
  LC_NUMERIC=C awk '{sum+=$4; n++} END { if(n>0) printf("%.6f\n", sum/n); else print("0.0") }' "$1"
}

spy_total_watts() {
  LC_NUMERIC=C awk -v ppf="$1" -v ppe="$2" 'BEGIN{
    if (!(ppe>0)) { print("nan"); exit 0 }
    printf("%.1f\n", ppf/ppe)
  }'
}

run_pass() {
  local eff="$1"
  local tag="$2"
  local out_map="$3"

  # Build emitters (writes $OUTDIR/emitters_spydr3_ALL_umol.rad)
  if [[ ! -f "$ROOT/generate_emitters_spydr3.py" ]]; then
    echo "ERROR: $ROOT/generate_emitters_spydr3.py not found."
    exit 1
  fi

  SPYDR_PPF="$SPYDR_PPF" SPYDR_Z_M="$SPYDR_Z_M" \
  NX="$NX" NY="$NY" SUBPATCH_GRID="$SUBPATCH_GRID" \
  EFF_SCALE="$eff" \
  "$PY" "$ROOT/generate_emitters_spydr3.py"

  [[ -f "$OUTDIR/emitters_spydr3_ALL_umol.rad" ]] || { echo "ERROR: $OUTDIR/emitters_spydr3_ALL_umol.rad missing."; exit 1; }
  echo "Emitters:"
  echo "  • $OUTDIR/emitters_spydr3_ALL_umol.rad"

  # Build octree fresh
  echo "Building octree..."
  oconv -f "$ROOM" "$OUTDIR/emitters_spydr3_ALL_umol.rad" > "$OCT"

  # rtrace → RGB (each channel already µmol); average for scalar PPFD
  local snake="$ROOT/.sensors_snake_${tag}.txt"
  local ambcache="$ROOT/.radcache/amb_${tag}"
  rm -f "${ambcache}"*

  # Generate a boustrophedon reordering of the grid (order-agnostic; infers the grid)
  "$PY" - "$SENSORS" > "$snake" <<'PY'
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
  local snake_os="$ROOT/.sensors_snake_os_${tag}.txt"
  local dirs="$ROOT/.dirs_tmp_${tag}.txt"
  local rgb_tmp="$ROOT/.rgb_tmp_${tag}.txt"
  local map_tmp="$ROOT/.ppfd_map_tmp_${tag}.txt"

  awk -v k="$OS" '{ for (i=0;i<k;i++) print $0 }' "$snake" > "$snake_os"

  # Build the rtrace direction file
  LC_NUMERIC=C awk '{printf "%.6f %.6f %.6f 0 0 1\n",$1,$2,$3}' "$snake_os" > "$dirs"

  echo "Tracing irradiance..."
  rtrace -h -I+ -n "$NTHREADS" $RT_OPTS_BASE -af "$ambcache" "$OCT" < "$dirs" > "$rgb_tmp"

  # Average every OS rows back down to one value per original sensor
  paste "$snake_os" "$rgb_tmp" \
  | awk -v k="$OS" '{
    n=NF; R=$(n-2); G=$(n-1); B=$n; ppfd=(R+G+B)/3.0;
    idx = int((NR-1)/k) + 1;
    sx[idx]=$1; sy[idx]=$2; sz[idx]=$3;
    sum[idx]+=ppfd; cnt[idx]++
}
END{
  for(i=1;i<=length(sum);i++)
    printf("%.6f %.6f %.6f %.6f\n", sx[i], sy[i], sz[i], sum[i]/cnt[i]);
}' > "$map_tmp"

  rm -f "$rgb_tmp" "$snake" "$snake_os" "$dirs"
  echo "✔ Wrote $map_tmp"

  # Optional symmetry filter (near-zero cost). Toggle with SYM=1
  if [[ "$SYM" = "1" ]]; then
    "${PY:-python3}" "$ROOT/symmetrize_ppfd.py" \
      --input "$map_tmp" \
      --output "$out_map" \
      --lam 1.0 \
      --tol 1e-6 \
      ${AXES_ONLY:+--axes-only} \
      --verbose
    rm -f "$map_tmp"
  else
    mv -f "$map_tmp" "$out_map"
  fi
  echo "✔ Wrote $out_map"
}

PASS1="$ROOT/.ppfd_map_pass1.txt"

echo "Target:"
if [[ -n "$TARGET_PPFD" ]]; then
  echo "  target mean PPFD: $TARGET_PPFD"
else
  echo "  target mean PPFD: (not set)"
fi

echo "Pass 1 (EFF_SCALE=$EFF_SCALE)..."
run_pass "$EFF_SCALE" "p1" "$PASS1"
MEAN1="$(ppfd_mean "$PASS1")"
echo "Pass 1 mean PPFD: $MEAN1"

FINAL_EFF_SCALE="$EFF_SCALE"
if [[ "$AUTO_DIM" = "1" && -n "$TARGET_PPFD" ]]; then
  if LC_NUMERIC=C awk -v m="$MEAN1" 'BEGIN{exit !(m>0)}'; then
    MULT="$(LC_NUMERIC=C awk -v t="$TARGET_PPFD" -v m="$MEAN1" 'BEGIN{printf("%.8f", t/m)}')"
    DESIRED_EFF_SCALE="$(LC_NUMERIC=C awk -v e="$EFF_SCALE" -v k="$MULT" 'BEGIN{printf("%.8f", e*k)}')"
    FINAL_EFF_SCALE="$(clamp_0_1 "$DESIRED_EFF_SCALE")"
    echo "Auto-dim: multiplier=$MULT → desired EFF_SCALE=$DESIRED_EFF_SCALE → final EFF_SCALE=$FINAL_EFF_SCALE"
    if LC_NUMERIC=C awk -v d="$DESIRED_EFF_SCALE" 'BEGIN{exit !(d>1.000001)}'; then
      echo "NOTE: target requires more output than the fixture can produce at this SPYDR_PPF; capping at EFF_SCALE=1.0 (no overdrive)."
    fi

    if LC_NUMERIC=C awk -v a="$FINAL_EFF_SCALE" -v b="$EFF_SCALE" 'BEGIN{exit !(a!=b)}'; then
      echo "Pass 2 (EFF_SCALE=$FINAL_EFF_SCALE)..."
      run_pass "$FINAL_EFF_SCALE" "p2" "$ROOT/ppfd_map.txt"
      MEAN2="$(ppfd_mean "$ROOT/ppfd_map.txt")"
      echo "Pass 2 mean PPFD: $MEAN2"
    else
      cp -f "$PASS1" "$ROOT/ppfd_map.txt"
    fi

    if LC_NUMERIC=C awk -v d="$DESIRED_EFF_SCALE" -v f="$FINAL_EFF_SCALE" 'BEGIN{exit !((d>1.000001) && (f>=0.999999))}'; then
      echo "NOTE: target mean PPFD is unattainable with current SPYDR_PPF/NX/NY; increase fixtures or SPYDR_PPF."
    fi
  else
    echo "WARNING: Pass 1 mean PPFD is <= 0; skipping auto-dim."
    cp -f "$PASS1" "$ROOT/ppfd_map.txt"
  fi
else
  cp -f "$PASS1" "$ROOT/ppfd_map.txt"
fi

rm -f "$PASS1"

FIXTURES=$(( NX * NY ))
FINAL_TOTAL_PPF="$(LC_NUMERIC=C awk -v ppf="$SPYDR_PPF" -v n="$FIXTURES" -v e="$FINAL_EFF_SCALE" 'BEGIN{printf("%.2f", ppf*n*e)}')"
FINAL_TOTAL_W="$(spy_total_watts "$FINAL_TOTAL_PPF" "$SPYDR_PPE_UMOL_PER_J")"
echo "Final SPYDR output:"
echo "  fixtures=$FIXTURES (NX=$NX, NY=$NY)"
echo "  PPF/fixture=${SPYDR_PPF} µmol/s  dimmer(EFF_SCALE)=${FINAL_EFF_SCALE}"
echo "  total PPF ≈ ${FINAL_TOTAL_PPF} µmol/s"
if [[ "$FINAL_TOTAL_W" = "nan" ]]; then
  echo "  total electrical power: (set SPYDR_PPE_UMOL_PER_J > 0 to estimate)"
else
  echo "  total electrical power ≈ ${FINAL_TOTAL_W} W  (assumed PPE=${SPYDR_PPE_UMOL_PER_J} µmol/J)"
fi
