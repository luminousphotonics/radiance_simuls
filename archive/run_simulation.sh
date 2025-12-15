#!/usr/bin/env zsh
set -euo pipefail

echo "--- Starting PPFD Simulation (manual COB emitters) ---"
export RAYPATH=".:${PWD}/ies_sources:/usr/local/lib/ray"

# 1) Build scene with hand-crafted COBs
python generate_emitters_manual.py

# 2) Generate sensor grid
python generate_grid.py rtrace  > sensor_points.txt
python generate_grid.py coords  > grid_coords.txt

# 3) Build Radiance octree
rm -f scene_manual.oct
oconv ies_sources/emitters_manual.rad room.rad > scene_manual.oct

# 4) Run Radiance simulation (RGB)
rtrace -n "${NPROC:-4}" -I -h -w \
  -ab 5 -ad 512 -as 128 -aa 0.15 -lw 1e-2 -dp 64 \
  scene_manual.oct < sensor_points.txt > irr_manual_RGB.txt

# 5) Convert RGB → PPFD
E=$(grep -m1 'E_per_μmol=' ies_sources/emitters_manual.rad | sed -E 's/.*E_per_μmol=([0-9.]+).*/\1/')
awk -v E="$E" '{ print ($1+$2+$3)/(3*E) }' irr_manual_RGB.txt > ppfd.txt

# 6) Combine coords + PPFD
paste <(tail -n +2 grid_coords.txt) ppfd.txt > ppfd_map.txt

echo "--- Done. PPFD map → ppfd_map.txt ---"
