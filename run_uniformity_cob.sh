#!/usr/bin/env bash
# run_uniformity_cob.sh
# Uniformity solver pipeline for the COB + strip layout (independent of SMD/Quantum).

set -euo pipefail

PY=${PY:-python3}
ROOT="$(pwd)"
export ROOT

RUN_BASIS="${RUN_BASIS:-0}"
BASIS_PATH="${BASIS_PATH:-$ROOT/basis_cob.npy}"
OUT_JSON="${OUT_JSON:-$ROOT/ring_powers_cob.json}"
export OUT_JSON
TARGET_PPFD="${TARGET_PPFD:-1200}"
W_MIN="${W_MIN:-0}"
W_MAX="${W_MAX:-85}"
LAMBDA_S="${LAMBDA_S:-0.0 1e-3 1e-2 1e-1 1.0 10.0 30.0}"
LAMBDA_R="${LAMBDA_R:-0.0 1e-3 1e-2 1e-1}"
LAMBDA_MEAN="${LAMBDA_MEAN:-10.0}"
USE_CHEBYSHEV="${USE_CHEBYSHEV:-1}"

echo
echo "--- COB Uniformity pipeline ---"
echo "RUN_BASIS=${RUN_BASIS}"
echo "BASIS_PATH=${BASIS_PATH}"
echo "TARGET_PPFD=${TARGET_PPFD}"
echo "W_MIN=${W_MIN}  W_MAX=${W_MAX}  (W per COB)"
echo "LAMBDA_S=${LAMBDA_S}"
echo "LAMBDA_R=${LAMBDA_R}"
echo "LAMBDA_MEAN=${LAMBDA_MEAN}"
echo "USE_CHEBYSHEV=${USE_CHEBYSHEV}"
echo "OUT_JSON=${OUT_JSON}"
echo

# Auto-detect ring count
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
echo "Detected ring_n=${COB_RING_N}"
EXPECTED_VARS=$((COB_RING_N+1))
RINGS=$((COB_RING_N+1))

# Rebuild basis if mismatch
BASIS_MANIFEST="${ROOT}/basis_manifest_cob.json"
if [[ -f "${BASIS_MANIFEST}" ]]; then
  BASIS_RINGS="$(python3 - <<'PY'
import json
p="basis_manifest_cob.json"
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

manifest = json.loads(Path("basis_manifest_cob.json").read_text())
if "generator_sha256" not in manifest or "emitter_env" not in manifest:
    raise SystemExit("basis manifest missing emitter fingerprint (legacy basis)")
want_sha = hashlib.sha256(Path("generate_emitters_cob.py").read_bytes()).hexdigest()
have_sha = manifest.get("generator_sha256", "")
if have_sha and have_sha != want_sha:
    raise SystemExit("basis generator changed")

have_env = manifest.get("emitter_env") or {}
want_env = {
    "EFF_SCALE": float(os.environ.get("EFF_SCALE", "1.0")),
    "COB_BASE_RING_N": int(os.environ.get("COB_BASE_RING_N", "6")),
    "COB_PPE_UMOL_PER_J": float(os.environ.get("COB_PPE_UMOL_PER_J", "2.76")),
    "STRIP_PPE_UMOL_PER_J": float(os.environ.get("STRIP_PPE_UMOL_PER_J", "2.76")),
    "COB_STRIP_MODE": os.environ.get("COB_STRIP_MODE", "proportional"),
    "COB_STRIP_MIN_RING": int(os.environ.get("COB_STRIP_MIN_RING", "1")),
    "COB_STRIP_W_FRACTION": float(os.environ.get("COB_STRIP_W_FRACTION", "0.05")),
    "COB_DISK_SIDES": int(os.environ.get("COB_DISK_SIDES", "16")),
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
if [[ "${BASIS_RINGS}" -ne "${EXPECTED_VARS}" ]]; then
  echo "Basis vars (${BASIS_RINGS}) differ from expected (${EXPECTED_VARS} = rings); rebuilding basis."
  RUN_BASIS=1
fi

if [[ "${RUN_BASIS}" == "1" ]]; then
  echo "[1/3] Building basis matrix A (COB)..."
  RUN_BASIS=1 "${ROOT}/run_basis_extraction_cob.sh"
  BASIS_PATH="$ROOT/basis_cob.npy"
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

echo
echo "Labeled ring powers (COB W/COB):"
"${PY}" - <<'PY'
import json, os
from pathlib import Path

out_json = Path(os.environ.get("OUT_JSON", "ring_powers_cob.json"))
if not out_json.exists():
    raise SystemExit(f"missing {out_json}")
data = json.loads(out_json.read_text())
w = data.get("ring_powers_W_per_module") or data.get("ring_powers") or []
w = list(w)
for L, wi in enumerate(w):
    print(f"  ring {L:2d}: {float(wi):6.3f} W/COB")
PY

echo "[3/3] Running Radiance simulation with solved ring powers..."
USE_RING_POWERS_JSON=1 \
RING_POWERS_JSON="${OUT_JSON}" \
"${ROOT}/run_simulation_cob.sh"

echo
echo "Writing COB solver report for Metrics tab..."
"${PY}" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ.get("ROOT", "."))
out_json = Path(os.environ.get("OUT_JSON", root / "ring_powers_cob.json"))
layout = root / "ies_sources" / "cob_layout.json"
report = root / "ies_sources" / "cob_solver_report.txt"

data = json.loads(out_json.read_text()) if out_json.exists() else {}
w = data.get("ring_powers_W_per_module") or data.get("ring_powers") or []
w = [float(x) for x in list(w)]

meta = {}
ring_counts = {}
strip_len = {}
strip_w = {}
if layout.exists():
    lj = json.loads(layout.read_text())
    meta = lj.get("meta") or {}
    rc = (lj.get("ring_channels") or {}).get("cob_counts_by_ring") or {}
    sl = (lj.get("ring_channels") or {}).get("strip_length_m_by_ring") or {}
    sw = (lj.get("ring_channels") or {}).get("strip_ring_total_watts_in") or {}
    ring_counts = {int(k): int(v) for k, v in rc.items()}
    strip_len = {int(k): float(v) for k, v in sl.items()}
    strip_w = {int(k): float(v) for k, v in sw.items()}

lines = []
lines.append("COB solver report:")
lines.append(f"  ring_powers_json: {out_json.resolve() if out_json.exists() else out_json}")
lines.append(f"  layout_json     : {layout.resolve() if layout.exists() else '(missing)'}")
lines.append("  note: solver variables are W/COB per ring (strips are proportional to COB ring watts).")

if not w:
    lines.append("  (no ring powers in JSON yet)")
else:
    lines.append(f"  rings: {len(w)} (0..{len(w)-1})")
    for L, wpc in enumerate(w):
        n = ring_counts.get(L, 0)
        total = float(wpc) * float(n)
        ln = float(strip_len.get(L, 0.0))
        sw = float(strip_w.get(L, 0.0))
        wpm = (sw / ln) if ln > 1e-12 else 0.0
        lines.append(
            f"    ring {L:2d}: {wpc:6.3f} W/COB × {n:3d} = {total:8.1f} W | "
            f"strip {sw:7.2f} W ({wpm:6.2f} W/m over {ln:6.2f} m)"
        )

report.write_text("\n".join(lines).rstrip() + "\n")
print(f"✔ Wrote {report}")
PY

echo
echo "Pipeline complete. Solved ring powers: ${OUT_JSON}"
echo "PPFD map: ppfd_map.txt"
echo "To visualize: python3 visualize_ppfd.py --overlay cob"
