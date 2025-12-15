#!/usr/bin/env bash
# Single-pass PPFD for SMD macro-emitters (uses your generate_emitters_smd.py)
set -Eeuo pipefail

echo "--- SMD PPFD Simulation (single pass, PPFD-scaled) ---"

# Detect cores (macOS / POSIX)
: "${NPROC:=$( (command -v sysctl >/dev/null && sysctl -n hw.logicalcpu) || \
               (command -v getconf >/dev/null && getconf _NPROCESSORS_ONLN) || echo 4 )}"

export RAYPATH=".:${PWD}/ies_sources:/usr/local/lib/ray"

# 1) Build per-channel macro emitters + ppfd_constants.env (ALL params live here)
python3 generate_emitters_smd.py

# 2) Merge per-channel into one PPFD-scaled emitter file
python3 combine_emitters_to_umol.py

# 3) Sensor grid (skip these if you already have them)
python3 generate_grid.py rtrace > sensor_points.txt
python3 generate_grid.py coords  > grid_coords.txt

# 4) Octree (all channels at once)
oconv ies_sources/emitters_smd_ALL_umol.rad room.rad > scene_smd_all.oct

# 5) Single rtrace (irradiance already in µmol units). Cache ambient for speed.
AMB="ies_sources/amb_smd.cache"
rtrace -n "$NPROC" -I -h -w \
  -af "$AMB" -aa 0.12 -ab 5 -ad 768 -as 256 -ar 64 -lr 8 -lw 1e-3 \
  -dj 0.7 -ds 0.20 -dc 0.5 -dr 2 -st 0.15 \
  scene_smd_all.oct < sensor_points.txt > irr_all_umol_RGB.txt

# 6) Convert RGB→scalar PPFD (equal-grey by design)
awk '{ print ($1+$2+$3)/3.0 }' irr_all_umol_RGB.txt > ppfd_total.txt

# 7) Attach coordinates (drop header if present)
paste <(tail -n +2 grid_coords.txt) ppfd_total.txt > ppfd_map.txt
echo "--- Done. PPFD map → ppfd_map.txt ---"
