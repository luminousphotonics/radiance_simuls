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
SPYDR_PPE_UMOL_PER_J="${SPYDR_PPE_UMOL_PER_J:-2.76}"  # full-power PPE (µmol/J) for electrical watts estimate
SPYDR_DROOP="${SPYDR_DROOP:-1}"     # 1=apply PPE droop vs fixture power, 0=disable
SPYDR_W_FULL="${SPYDR_W_FULL:-800}" # W at full-power PPE
SPYDR_PPE_FULL="${SPYDR_PPE_FULL:-$SPYDR_PPE_UMOL_PER_J}"
SPYDR_W_LOW="${SPYDR_W_LOW:-390}"   # W at low-power PPE
SPYDR_PPE_LOW="${SPYDR_PPE_LOW:-3.0}"

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
# Match COB: sample the full grow space footprint with a tiny wall inset.
GRID_WALL_MARGIN_M="${GRID_WALL_MARGIN_M:-0.005}"
WALL_MARGIN_M="$GRID_WALL_MARGIN_M" MODULE_SIDE_M="0.0" "$PY" "$ROOT/generate_grid.py" rtrace > "$SENSORS"

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

spy_droop_k() {
  LC_NUMERIC=C awk -v ppe_full="$1" -v ppe_low="$2" -v w_full="$3" -v w_low="$4" 'BEGIN{
    if (!(ppe_full>0) || !(ppe_low>0) || !(w_full>0) || !(w_low>0)) { print("0.0"); exit 0 }
    if (w_full==w_low) { print("0.0"); exit 0 }
    k = log(ppe_low/ppe_full) / -log(w_low/w_full)
    if (k!=k) k=0.0
    printf("%.6f\n", k)
  }'
}

spy_ppe_for_eff() {
  local eff="$1"
  local k="$2"
  LC_NUMERIC=C awk -v ppe_full="$SPYDR_PPE_FULL" -v w_full="$SPYDR_W_FULL" -v eff="$eff" -v k="$k" 'BEGIN{
    if (!(ppe_full>0) || !(w_full>0)) { print("nan"); exit 0 }
    p = w_full * eff
    x = p / w_full
    if (x < 0.05) x = 0.05
    if (x > 1.2) x = 1.2
    ppe = ppe_full * (x ** (-k))
    if (ppe!=ppe) ppe = ppe_full
    printf("%.6f\n", ppe)
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
SPYDR_DROOP_K="0.0"
SPYDR_PPE_EFFECTIVE="$SPYDR_PPE_UMOL_PER_J"
if [[ "$SPYDR_DROOP" = "1" ]]; then
  SPYDR_DROOP_K="$(spy_droop_k "$SPYDR_PPE_FULL" "$SPYDR_PPE_LOW" "$SPYDR_W_FULL" "$SPYDR_W_LOW")"
  SPYDR_PPE_EFFECTIVE="$(spy_ppe_for_eff "$FINAL_EFF_SCALE" "$SPYDR_DROOP_K")"
fi
FINAL_TOTAL_W="$(spy_total_watts "$FINAL_TOTAL_PPF" "$SPYDR_PPE_EFFECTIVE")"
echo "Final SPYDR output:"
echo "  fixtures=$FIXTURES (NX=$NX, NY=$NY)"
echo "  PPF/fixture=${SPYDR_PPF} µmol/s  dimmer(EFF_SCALE)=${FINAL_EFF_SCALE}"
echo "  total PPF ≈ ${FINAL_TOTAL_PPF} µmol/s"
if [[ "$FINAL_TOTAL_W" = "nan" ]]; then
  echo "  total electrical power: (set SPYDR_PPE_UMOL_PER_J > 0 to estimate)"
else
  if [[ "$SPYDR_DROOP" = "1" ]]; then
    echo "  droop: PPE_full=${SPYDR_PPE_FULL} at ${SPYDR_W_FULL} W, PPE_low=${SPYDR_PPE_LOW} at ${SPYDR_W_LOW} W, K=${SPYDR_DROOP_K}"
    echo "  total electrical power ≈ ${FINAL_TOTAL_W} W  (effective PPE=${SPYDR_PPE_EFFECTIVE} µmol/J)"
  else
    echo "  total electrical power ≈ ${FINAL_TOTAL_W} W  (assumed PPE=${SPYDR_PPE_UMOL_PER_J} µmol/J)"
  fi
fi

# Save effective PPE for GUI/metrics consumption
POWER_META="$OUTDIR/spydr3_power.txt"
{
  echo "ppe_effective=${SPYDR_PPE_EFFECTIVE}"
  echo "ppe_full=${SPYDR_PPE_FULL}"
  echo "ppe_low=${SPYDR_PPE_LOW}"
  echo "w_full=${SPYDR_W_FULL}"
  echo "w_low=${SPYDR_W_LOW}"
  echo "droop_k=${SPYDR_DROOP_K}"
  echo "eff_scale=${FINAL_EFF_SCALE}"
  echo "total_ppf=${FINAL_TOTAL_PPF}"
  echo "total_w=${FINAL_TOTAL_W}"
  echo "droop_enabled=${SPYDR_DROOP}"
} > "$POWER_META"

# Log "usable photons under a PPFD ceiling" metrics.
# Set SETPOINT_PPFD (or TARGET_PPFD) to treat that as the canopy PPFD cap.
if [[ "${LOG_CAP_METRICS:-1}" = "1" ]]; then
  CAP_PPFD="${SETPOINT_PPFD:-${TARGET_PPFD:-}}"
  TOTAL_INPUT_WATTS="${FINAL_TOTAL_W:-}"
  TOTAL_EMITTED_PPF="${FINAL_TOTAL_PPF:-}"
  TOTAL_INPUT_WATTS="$TOTAL_INPUT_WATTS" TOTAL_EMITTED_PPF="$TOTAL_EMITTED_PPF" CAP_PPFD="$CAP_PPFD" "${PY:-python3}" - <<'PY'
import os
import numpy as np

from ppfd_metrics import compute_ppfd_metrics, format_ppfd_metrics_line

root = os.environ.get("ROOT", ".")
path = os.path.join(root, "ppfd_map.txt")
if not os.path.exists(path):
    path = "ppfd_map.txt"

data = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.replace(",", " ").split()
        if len(parts) < 4:
            continue
        data.append(float(parts[-1]))

ppfd = np.asarray(data, dtype=float)
cap_raw = os.environ.get("CAP_PPFD", "").strip()
cap = float(cap_raw) if cap_raw else None

# total input watts (at current dimmer), optional
watts = None
watts_raw = os.environ.get("TOTAL_INPUT_WATTS", "").strip()
try:
    watts = float(watts_raw) if watts_raw else None
except ValueError:
    watts = None
if watts is not None and not (watts > 0):
    watts = None

# emitted photons (PPF) at current dimmer, optional
emitted_ppf = None
emitted_ppf_raw = os.environ.get("TOTAL_EMITTED_PPF", "").strip()
try:
    emitted_ppf = float(emitted_ppf_raw) if emitted_ppf_raw else None
except ValueError:
    emitted_ppf = None
if emitted_ppf is not None and not (emitted_ppf > 0):
    emitted_ppf = None

# Default canopy area to full room footprint (matches GUI behavior), unless explicitly set.
area = None
area_raw = os.environ.get("CANOPY_AREA_M2", "").strip()
try:
    area = float(area_raw) if area_raw else None
except ValueError:
    area = None
if area is None:
    try:
        L_ft = float(os.environ.get("LENGTH_FT", "").strip() or "0")
        W_ft = float(os.environ.get("WIDTH_FT", "").strip() or "0")
        if os.environ.get("ALIGN_LONG_AXIS_X", "").strip() == "1" and W_ft > L_ft:
            L_ft, W_ft = W_ft, L_ft
        if L_ft > 0 and W_ft > 0:
            area = (L_ft * W_ft) * 0.09290304
    except Exception:
        area = None

m = compute_ppfd_metrics(
    ppfd,
    setpoint_ppfd=cap,
    canopy_area_m2=area,
    total_input_watts=watts,
    emitted_ppf_umol_s=emitted_ppf,
    legacy_metrics=True,
)
print("METRICS:", format_ppfd_metrics_line(m))
PY
fi

echo "Done."
