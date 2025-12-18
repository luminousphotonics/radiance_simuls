#!/usr/bin/env python3
# generate_emitters_smd.py • v3.5
# - Optics toggle via OPTICS=none|lens
# - Lens parameters override via LENS_CONFIG=path/to/lens_config.json
# - Per-run layout snapshot: ies_sources/smd_layout.json (meters)
# - GUI-friendly: driven entirely by env vars (radiance_gui.py), no CLI args

from __future__ import annotations
import json, math, os, random
from pathlib import Path
from typing import Dict, List, Tuple, Any
from layout_generator import generate_layout
from rect_layout import build_rect_grid



# ──────────────────────────────────────────────────────────────────────────────
# Environment → geometry
# ──────────────────────────────────────────────────────────────────────────────
def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return float(default)
    return float(v)

def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return int(default)
    return int(v)

# Room (feet/inches) → meters
LENGTH_M      = _env_float("LENGTH_FT", 12.0) * 0.3048
WIDTH_M       = _env_float("WIDTH_FT",  12.0) * 0.3048
HEIGHT_M      = _env_float("HEIGHT_M", 3.048)
WALL_MARGIN_M = _env_float("MARGIN_IN", 0.0)  * 0.0254

# Rings: RING_N = n (outer index), RINGS = n+1 (0..n)
RING_N        = _env_int("SMD_RING_N", 7)                # GUI sets this directly or via density → ring count
RINGS         = max(1, RING_N + 1)

# Mount height (keep as-is for your scene)
MOUNT_Z_M     = _env_float("MOUNT_Z_M", 0.4572)          # 18 in

# Macro patch footprint (meters)
MODULE_SIDE_M = _env_float("MODULE_SIDE_M", 0.12)        # board side (120 mm)
PATCH_SIDE_M  = MODULE_SIDE_M
SUBPATCH_GRID = _env_int("SUBPATCH_GRID", 3)             # subdiv to distribute emission over board

# Layout mode: 'square' (current) or 'rect_rect' (rectangular rings)
LAYOUT_MODE = os.getenv("LAYOUT_MODE", "square").strip().lower()

# ──────────────────────────────────────────────────────────────────────────────
# Electrical / spectral model
# ──────────────────────────────────────────────────────────────────────────────
COUNT_PER_MODULE: Dict[str, int] = {
    "WW": 32, "CW": 32, "R": 20, "B": 12, "FR": 8, "C": 4, "UV": 6,
}
PER_LED_W: Dict[str, float] = {
    "WW": 0.60, "CW": 0.60, "R": 0.45, "B": 0.35, "C": 0.35, "FR": 0.35, "UV": 0.25,
}
PPE_UMOL_PER_J: Dict[str, float] = {
    "WW": 3.35, "CW": 3.35, "R": 4.6, "B": 2.4, "C": 2.0, "FR": 3.6, "UV": 1.0,
}

# Per-ring electrical watts (per module). Unknown outer rings clamp to last known.
RING_POWER_W: Dict[int, float] = {
    0: 31.827, 1: 28.284, 2: 33.154, 3: 30.241, 4: 25.900, 5: 44.723,
    6: 3.596, 7: 70.760, 8: 70.760  # 8 reserved for connector columns in modular layouts
}
RING_POWERS_SOURCE = "built-in"

# Global derating
# Important: PPE_UMOL_PER_J can either represent:
#   (A) fixture/system-level PPE (already includes driver/thermal/optical losses), or
#   (B) LED/board-level PPE (needs additional derate multipliers applied).
#
# For SMD we default to (A) so PPE behaves like Quantum by default; set
# PPE_IS_SYSTEM=0 if your PPE values are LED/board-level and you want additional
# driver/thermal/optical/wiring derates applied.
PPE_IS_SYSTEM = os.getenv("PPE_IS_SYSTEM", "1").strip() != "0"

DRIVER_EFF     = _env_float("DRIVER_EFF",     0.95)
THERMAL_EFF    = _env_float("THERMAL_EFF",    0.92)
BOARD_OPT_EFF  = _env_float("BOARD_OPT_EFF",  0.95)
WIRING_EFF     = _env_float("WIRING_EFF",     0.99)
USER_EFF_SCALE = _env_float("EFF_SCALE",      1.00)  # dimmer fraction (0..1)

if PPE_IS_SYSTEM:
    DERATE = USER_EFF_SCALE
else:
    DERATE = DRIVER_EFF * THERMAL_EFF * BOARD_OPT_EFF * WIRING_EFF * USER_EFF_SCALE

EFF_SCALE = DERATE

# ----------------------------------------------------------------------
# Basis-mode overrides for building A (per-ring response matrix)
# ----------------------------------------------------------------------
# When SMD_BASIS_MODE=1, we:
#   - Force one ring (SMD_BASIS_RING) to SMD_BASIS_UNIT_W (per module)
#   - Force all other rings to 0 W
#   - Optionally neutralize derates (SMD_BASIS_UNDERRATE=1 → EFF_SCALE=1.0)
BASIS_MODE      = os.getenv("SMD_BASIS_MODE", "0") == "1"
BASIS_RING      = int(os.getenv("SMD_BASIS_RING", "-1"))
BASIS_UNIT_W    = float(os.getenv("SMD_BASIS_UNIT_W", "1.0"))

if BASIS_MODE:
    if BASIS_RING >= 0:
        for L in range(RINGS):
            if L == BASIS_RING:
                RING_POWER_W[L] = BASIS_UNIT_W
            else:
                RING_POWER_W[L] = 0.0
        RING_POWERS_SOURCE = f"basis_mode_ring_{BASIS_RING}"


def _apply_ring_powers_override():
    """
    If USE_RING_POWERS_JSON=1 and a JSON file is available (default: ring_powers_optimized.json),
    override RING_POWER_W entries accordingly. Skipped when BASIS_MODE=1.
    """
    global RING_POWERS_SOURCE
    if BASIS_MODE:
        return
    use_json = os.getenv("USE_RING_POWERS_JSON", "1") != "0"
    path = Path(os.getenv("RING_POWERS_JSON", "ring_powers_optimized.json"))
    if not use_json or not path.exists():
        return
    try:
        data = json.loads(path.read_text())
        arr = data.get("ring_powers_W_per_module") or data.get("ring_powers")
        idxs = data.get("ring_indices")
        if arr is None:
            return
        arr = list(arr)
        if idxs is None:
            idxs = list(range(len(arr)))
        if len(idxs) != len(arr):
            print(f"WARNING: ring_powers length mismatch in {path}; skipping override")
            return
        for k, v in zip(idxs, arr):
            try:
                RING_POWER_W[int(k)] = float(v)
            except Exception:
                continue
        RING_POWERS_SOURCE = f"json:{path}"
        print(f"Applied ring powers from {path}")
    except Exception as e:
        print(f"WARNING: failed to load ring powers JSON ({path}): {e}")

_apply_ring_powers_override()


# ──────────────────────────────────────────────────────────────────────────────
# Optics (lens patterns)
# ──────────────────────────────────────────────────────────────────────────────
OPTICS_MODE = os.getenv("OPTICS", "lens").lower()  # "lens" or "none"
LENS_RINGS = {
    1: ("bat", {"fwhm": 51.0, "bat_k": 0.75}),
    2: ("bat", {"fwhm": 47.0, "bat_k": 0.75}),
    3: ("sym", {"fwhm": 45.0}),
    4: ("sym", {"fwhm": 43.0}),
    5: ("sym", {"fwhm": 39.0}),
    6: ("ellip", {"ex": 33.0, "ey": 24.0}),
    7: ("ellip_mix", {  # corners 25×18°, edges 27×20°
        "edge_ex": 27.0, "edge_ey": 20.0,
        "corner_ex": 25.0, "corner_ey": 18.0
    }),
}

def _apply_lens_overrides():
    cfgp = os.getenv("LENS_CONFIG", "").strip()
    if not cfgp: return
    p = Path(cfgp)
    if not p.exists():
        print(f"WARNING: LENS_CONFIG file not found: {cfgp}")
        return
    try:
        cfg = json.loads(p.read_text())
        new = {}
        for k, v in cfg.items():
            try:
                L = int(k)
            except Exception:
                continue
            kind = v.get("kind")
            if kind not in ("sym","bat","ellip","ellip_mix"): continue
            params = {kk: float(vv) for kk, vv in v.items() if kk != "kind"}
            new[L] = (kind, params)
        if new:
            LENS_RINGS.update(new)
            print("Applied lens overrides from", p)
    except Exception as e:
        print("WARNING: failed to parse LENS_CONFIG:", e)

if OPTICS_MODE == "lens":
    _apply_lens_overrides()

# ──────────────────────────────────────────────────────────────────────────────
# Files / constants
# ──────────────────────────────────────────────────────────────────────────────
OUT_DIR  = Path("ies_sources"); OUT_DIR.mkdir(parents=True, exist_ok=True)
CAL_NAME = "beam.cal"
CAL_FILE = OUT_DIR / CAL_NAME
SMD_JSON = OUT_DIR / "smd_layout.json"

PI    = math.pi
SQRT2 = math.sqrt(2.0)
EPS   = 1e-12
rng   = random.Random(4242)

# ──────────────────────────────────────────────────────────────────────────────
# Lattice math (diamond / centered-square)
# ──────────────────────────────────────────────────────────────────────────────
def _solve_spacing(rings: int, length_m: float, width_m: float,
                   wall_margin_m: float, patch_side_m: float) -> float:
    """
    rings = n+1 → use L = n for geometry.
    Enforce: module square stays inside (margin) and lattice fits n rings.
    s = min(half_x, half_y) / (sqrt(2)*n)
    """
    L = max(1, rings - 1)       # outer index n (rings include 0)
    hs = 0.5 * patch_side_m
    half_x = 0.5 * length_m - wall_margin_m - hs
    half_y = 0.5 * width_m  - wall_margin_m - hs
    lim = min(half_x, half_y)
    if lim <= 0:
        raise SystemExit("ERROR: Not enough interior span given margin and module size.")
    return lim * math.sqrt(2.0) / L

def _ring_ij(L: int):
    if L == 0: return [(0,0)]
    pts = []
    i=L; j=0
    for k in range(L): pts.append((i-k, j-k))
    for k in range(L): pts.append((0-k, -L+k))
    for k in range(L): pts.append((-L+k, 0+k))
    for k in range(L): pts.append((0+k, L-k))
    return pts

def _ij_to_xy(i: int, j: int, spacing: float) -> Tuple[float,float]:
    # axis-aligned step d_axis = sqrt(2)*s; this mapping yields diamond coordinates (centered)
    return ((i - j) * spacing / SQRT2, (i + j) * spacing / SQRT2)

# ──────────────────────────────────────────────────────────────────────────────
# Public API for overlay/GUI
# ──────────────────────────────────────────────────────────────────────────────
def read_smd_layout_json() -> Tuple[List[dict], float]:
    """Return (positions, spacing_m) from snapshot (all meters)."""
    data = json.loads(SMD_JSON.read_text())
    if data.get("units") != "meters":
        raise ValueError("smd_layout.json is not in meters.")
    positions = data.get("positions", [])
    spacing = float(data.get("spacing", 0.0))
    return positions, spacing

def _compute_positions_from_env() -> Tuple[List[dict], float, dict]:
    """
    Compute positions from env (meters), no JSON dependency.
    Returns (positions, spacing, meta).
    """
    # Pull fresh room dims from env to allow per-run overrides
    L_m = _env_float("LENGTH_M", _env_float("LENGTH_FT", 12.0) * 0.3048)
    W_m = _env_float("WIDTH_M",  _env_float("WIDTH_FT",  12.0) * 0.3048)
    margin_m = WALL_MARGIN_M
    patch_m  = PATCH_SIDE_M
    patch_half_m = 0.5 * patch_m
    layout_mode = os.environ.get("LAYOUT_MODE", LAYOUT_MODE).strip().lower()

    # Guard usable spans
    usable_half_x = (L_m * 0.5) - margin_m - patch_half_m
    usable_half_y = (W_m * 0.5) - margin_m - patch_half_m
    if usable_half_x <= 0 or usable_half_y <= 0:
        raise SystemExit("ERROR: negative/zero usable half-span; check margin/patch/room.")

    # Square layouts: diamond rings, center singleton, axis spacing pinned to 12'x12'
    if layout_mode == "square":
        base_short_m = 3.6576
        base_ring_n = 7
        base_pitch_axis = (base_short_m / 2.0 - margin_m - patch_half_m) * 2.0 / (2 * base_ring_n)  # ≈0.2345 m
        half_min = min(usable_half_x, usable_half_y)
        ring_n = max(1, int(math.floor(half_min / base_pitch_axis)))
        rings_local = ring_n + 1
        spacing_diag = base_pitch_axis * SQRT2  # step used by diamond mapper to get axis pitch
        d_axis = base_pitch_axis

        pos: List[Dict[str, float]] = []
        z0 = MOUNT_Z_M
        for L in range(rings_local):
            for i, j in _ring_ij(L):
                x, y = _ij_to_xy(i, j, spacing_diag)
                pos.append({
                    "ring": L, "i": i, "j": j,
                    "x": round(x, 6), "y": round(y, 6), "z": z0
                })

        meta = {
            "room_L_m": L_m, "room_W_m": W_m,
            "margin_m": margin_m, "patch_side_m": patch_m,
            "ring_n": ring_n, "rings": rings_local,
            "spacing_m": d_axis, "d_axis_m": d_axis,
            "layout_mode": layout_mode,
            "pitch_x_m": d_axis, "pitch_y_m": d_axis, "pitch_m": d_axis,
        }
        return pos, d_axis, meta

    # Rectangular layouts using rect_layout (auto scales rings/counts)
    if layout_mode == "rect_rect":
        positions, meta_rect = build_rect_grid(
            length_m=L_m,
            width_m=W_m,
            height_m=HEIGHT_M,
            wall_margin_m=margin_m,
            module_side_m=patch_m,
            mount_z_m=MOUNT_Z_M,
        )
        pitch_x = float(meta_rect.get("pitch_x_m", 0.0))
        pitch_y = float(meta_rect.get("pitch_y_m", 0.0))
        pitch = float(meta_rect.get("pitch_m", 0.0)) if "pitch_m" in meta_rect else min(pitch_x, pitch_y)
        spacing_eff = pitch if pitch > 0 else 0.0
        swapped_axes = bool(meta_rect.get("swapped_axes", False))
        rings_count = int(meta_rect.get("rings_count", RINGS))
        ring_n_effective = rings_count - 1
        meta = {
            "room_L_m": L_m,
            "room_W_m": W_m,
            "room_H_m": HEIGHT_M,
            "margin_m": margin_m,
            "patch_side_m": patch_m,
            "ring_n": ring_n_effective,
            "rings": rings_count,
            "spacing_m": spacing_eff,
            "layout_mode": layout_mode,
            "pitch_x_m": pitch_x,
            "pitch_y_m": pitch_y,
            "pitch_m": pitch,
            "swapped_axes": swapped_axes,
        }
        meta.update({k: v for k, v in meta_rect.items() if k not in meta})
        return positions, spacing_eff, meta

    # Spacing s and axis-aligned step d_axis
    s = _solve_spacing(RINGS, L_m, W_m, margin_m, patch_m)
    d_axis = SQRT2 * s

    def _ring_ij_rect(L: int) -> List[Tuple[int, int]]:
        """
        Rectangular control rings for 12x24-type rooms.

        For now this is a solver-oriented layout:
        - Ring 0: horizontal bar of 7 emitters (i from -3..+3, j=0)
        - Ring 1: 16 emitters surrounding ring 0 (2*N0+2 rule)
        - Rings >=2: we grow a rectangular frame outwards and thin it
        so the ring counts follow your pattern: +4 emitters per ring.

        This gives the solver the right *ring structure* and *spacing*,
        without modeling physical fixture modules.
        """
        # ring 0: bar of 7
        if L == 0:
            return [(i, 0) for i in range(-3, 4)]  # -3..+3

        # ring 1: 16 positions around ring 0
        if L == 1:
            pts: List[Tuple[int,int]] = []
            # rows immediately above/below
            for i in range(-3, 4):
                pts.append((i,  1))
                pts.append((i, -1))
            # horizontal extensions at ends of bar
            pts.append((-4, 0))
            pts.append(( 4, 0))
            # quick sanity: 7*2 + 2 = 16
            assert len(pts) == 16
            return pts

        # Rings >= 2:
        # Target ring counts for 12x24 rectangular mode (from you):
        #  L : count
        #  0 :  7
        #  1 : 16
        #  2 : 20
        #  3 : 24
        #  4 : 28
        #  5 : 32
        #  6 : 36
        #  7 : 40
        TARGET_COUNTS = [7, 16, 20, 24, 28, 32, 36, 40]

        if L >= len(TARGET_COUNTS):
            raise SystemExit(f"rect_rect ring L={L} beyond TARGET_COUNTS table")

        # We construct an axis-aligned bounding box whose "ideal" perimeter
        # has at least TARGET_COUNTS[L] points, then subsample it symmetrically
        # to hit the exact count. This keeps a concentric rectangle shape
        # without over-constraining module/fixture layout.

        # Basic bounding box grows linearly with ring index.
        half_x = 3 + L       # starting from bar half-length 3
        half_y = 1 + L       # grow equally in ±y

        border: List[Tuple[int,int]] = []
        for i in range(-half_x, half_x + 1):
            border.append((i, -half_y))  # bottom
            border.append((i,  half_y))  # top
        for j in range(-half_y + 1, half_y):
            border.append((-half_x, j))  # left
            border.append(( half_x, j))  # right

        # Deduplicate and sort in a stable, symmetric order
        seen = set()
        uniq: List[Tuple[int,int]] = []
        for ij in border:
            if ij in seen:
                continue
            seen.add(ij)
            uniq.append(ij)

        needed = TARGET_COUNTS[L]
        if needed > len(uniq):
            raise SystemExit(f"L={L}: need {needed} border points, only have {len(uniq)}")

        # Symmetric thinning: take every k-th point around the border
        # to reach exactly 'needed' samples.
        step = max(1, len(uniq) // needed)
        pts = uniq[0:len(uniq):step]
        if len(pts) > needed:
            pts = pts[:needed]

        # As a safeguard, pad or trim symmetrically if off by 1 due to integer division
        while len(pts) < needed:
            # insert from the middle of uniq
            idx = len(pts) % len(uniq)
            if uniq[idx] not in pts:
                pts.append(uniq[idx])
            else:
                break
        if len(pts) > needed:
            pts = pts[:needed]

        assert len(pts) == needed, f"ring {L} got {len(pts)} != {needed}"
        return pts

    def _ring_indices(L: int) -> List[Tuple[int, int]]:
        """Return (i, j) indices for ring L under the active layout mode."""
        if layout_mode == "square":
            return _ring_ij(L)
        elif layout_mode == "rect_rect":
            return _ring_ij_rect(L)
        else:
            raise SystemExit(f"Unknown LAYOUT_MODE={layout_mode!r}")

    # Positions (centered at room origin)
    pos: List[Dict[str, float]] = []
    z0 = MOUNT_Z_M
    for L in range(RINGS):
        for i, j in _ring_indices(L):
            x, y = _ij_to_xy(i, j, s)
            pos.append({
                "ring": L, "i": i, "j": j,
                "x": round(x, 6), "y": round(y, 6), "z": z0
            })


    meta = {
        "room_L_m": L_m, "room_W_m": W_m,
        "margin_m": margin_m, "patch_side_m": patch_m,
        "ring_n": RING_N, "rings": RINGS,
        "spacing_m": s, "d_axis_m": d_axis,
        "layout_mode": layout_mode
    }
    return pos, s, meta

def get_module_positions() -> Tuple[List[Dict], float]:
    """
    Decide which layout generator to use based on LAYOUT_MODE.
    """
    positions, spacing, _meta = _compute_positions_from_env()
    return positions, spacing


# ──────────────────────────────────────────────────────────────────────────────
# Watts → photons → radiance
# ──────────────────────────────────────────────────────────────────────────────
def _per_module_nominal_watts() -> float:
    return sum(COUNT_PER_MODULE[ch] * PER_LED_W[ch] for ch in COUNT_PER_MODULE)

def _module_photon_flux_umol_s(ring: int) -> float:
    # clamp to last known per-ring wattage when ring beyond calibration
    base_keys = sorted(RING_POWER_W.keys())
    pw = RING_POWER_W.get(ring, RING_POWER_W[base_keys[-1]])
    target_w = pw * EFF_SCALE
    nom_w = _per_module_nominal_watts()
    scale = 0.0 if nom_w <= 0 else target_w / nom_w
    phi = 0.0
    for ch, n in COUNT_PER_MODULE.items():
        p_elec_ch = n * PER_LED_W[ch] * scale
        phi += p_elec_ch * PPE_UMOL_PER_J[ch]
    return phi

def _module_radiance_umol_per_sr_m2(ring: int) -> float:
    A = PATCH_SIDE_M * PATCH_SIDE_M
    phi = _module_photon_flux_umol_s(ring)
    return 0.0 if A <= 0 else phi / (A * PI)  # Lambertian baseline

# ──────────────────────────────────────────────────────────────────────────────
# Optics helpers + beam.cal
# ──────────────────────────────────────────────────────────────────────────────
def _m_from_fwhm(deg: float) -> float:
    th = math.radians(deg * 0.5)
    c = max(math.cos(th), 1e-6)
    return math.log(0.5) / math.log(c)

def _avg_sym_cos(m: float) -> float:
    # 1/π ∫ c^m * c dΩ = 2/(m+2)
    return 2.0 / (m + 2.0)

def _avg_bat_cos(m: float, k: float) -> float:
    # 1/π ∫ c^m * (1 + k(1-4c+4c^2)) * c dΩ
    return (2.0 / (m + 2.0)) + k * ((2.0 / (m + 2.0)) - (8.0 / (m + 3.0)) + (8.0 / (m + 4.0)))

def _avg_ellip_cos(ex_deg: float, ey_deg: float, samples: int = 8000) -> float:
    tx0 = math.tan(math.radians(ex_deg*0.5))
    ty0 = math.tan(math.radians(ey_deg*0.5))
    ln2 = math.log(2.0)
    s = 0.0
    for _ in range(samples):
        u1, u2 = rng.random(), rng.random()
        r = math.sqrt(u1); phi = 2.0*PI*u2
        sx = r*math.cos(phi); sy = r*math.sin(phi); sz = math.sqrt(max(0.0, 1.0-u1))
        tanx = abs(sx/max(sz, 1e-12)); tany = abs(sy/max(sz, 1e-12))
        rho2 = (tanx/tx0)**2 + (tany/ty0)**2
        s += math.exp(-ln2*rho2)
    return s / samples

def _write_beam_cal_base(path: Path) -> None:
    txt = r"""{ beam.cal • v1.6 — symmetric, batwing, elliptical (no in-CAL rotation) }
PI = 3.141592653589793;
eps = 1e-9;

c    = abs(Dz);
tanx = abs(Dx)/(abs(Dz)+eps);
tany = abs(Dy)/(abs(Dz)+eps);
pwr(x,y) = exp(y*log(max(x,1e-9)));

{ Base formulas (rotation handled in per-emitter wrappers) }
f_sym(m,gain)   = gain * pwr(c, m);
f_bat(m,gain,k) = gain * pwr(c, m) * (1 + k*(1 - 4*c + 4*c*c));
f_ell(tx0,ty0,g)= g * exp(-0.6931471805599453 * ((tanx/tx0)^2 + (tany/ty0)^2));
"""
    path.write_text(txt)

_WRAPPERS: Dict[str,str] = {}
def _add_wrapper(name: str, expr: str):
    if name not in _WRAPPERS:
        _WRAPPERS[name] = f"{name} = {expr};"

def _flush_wrappers():
    if not _WRAPPERS: return
    with CAL_FILE.open("a") as fh:
        fh.write("\n{ -- auto-generated lens wrappers -- }\n")
        for _, line in sorted(_WRAPPERS.items()):
            fh.write(line + "\n")

# ──────────────────────────────────────────────────────────────────────────────
# RAD writers
# ──────────────────────────────────────────────────────────────────────────────
def _write_area_square(fh, mat: str, cx: float, cy: float, z: float, side: float):
    hs = side * 0.5
    fh.write(f"{mat} polygon poly_{abs(hash((cx,cy)))%10**8}\n0\n0\n12\n")
    fh.write(f"  {cx - hs:.6f} {cy + hs:.6f} {z:.6f}\n")
    fh.write(f"  {cx + hs:.6f} {cy + hs:.6f} {z:.6f}\n")
    fh.write(f"  {cx + hs:.6f} {cy - hs:.6f} {z:.6f}\n")
    fh.write(f"  {cx - hs:.6f} {cy - hs:.6f} {z:.6f}\n\n")

def _write_area_grid(fh, mat: str, cx: float, cy: float, z: float, side: float, grid: int):
    if grid <= 1:
        _write_area_square(fh, mat, cx, cy, z, side); return
    cell = side / grid
    start = -0.5 * side + 0.5 * cell
    for r in range(grid):
        for c in range(grid):
            px = cx + start + c * cell
            py = cy + start + r * cell
            _write_area_square(fh, mat, px, py, z, cell)

def _write_brightfunc_ref(fh, patt_name: str, funcname: str):
    # Wrapper brightfunc with zero formal args; cal file provides symbols
    fh.write(f"void brightfunc {patt_name}\n")
    fh.write(f"2 {funcname} {CAL_NAME}\n")
    fh.write("0\n")
    fh.write("0\n\n")

# ──────────────────────────────────────────────────────────────────────────────
# Snapshot writer (authoritative overlay source)
# ──────────────────────────────────────────────────────────────────────────────
def write_smd_layout_json(
    positions: List[Dict[str, float]],
    spacing_m: float,
    room_L_m: float,
    room_W_m: float,
    margin_m: float,
    ring_n: int,
    patch_side_m: float,
    z_m: float = 0.0,
    extra: Dict[str, Any] | None = None
) -> None:
    from datetime import datetime
    def _rf(v, p=6): return float(round(float(v), p))
    SMD_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 2,
        "generator": "generate_emitters_smd.py",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "units": "meters",
        "rings": int(ring_n + 1),                  # total ring count (0..n)
        "modules": int(len(positions)),
        "spacing": _rf(spacing_m),
        "patch_side": _rf(patch_side_m),
        "room": {"L": _rf(room_L_m), "W": _rf(room_W_m)},
        "margin": _rf(margin_m),
        "z": _rf(z_m),
        "positions": [
            {"x": _rf(p.get("x", 0.0)), "y": _rf(p.get("y", 0.0)), "z": _rf(p.get("z", z_m)), "ring": int(p.get("ring", 0))}
            for p in positions
        ],
    }
    if extra:
        meta = dict(extra)
        for k in ("version", "units", "positions"):
            meta.pop(k, None)
        payload["meta"] = meta
    SMD_JSON.write_text(json.dumps(payload, indent=2))

# ──────────────────────────────────────────────────────────────────────────────
# Main generation
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # Compute positions from env (no JSON dependency at generation time)
    positions, spacing, meta = _compute_positions_from_env()
    layout_mode = meta.get("layout_mode", LAYOUT_MODE)
    pitch_x = meta.get("pitch_x_m")
    pitch_y = meta.get("pitch_y_m")
    rings_local = int(meta.get("rings", RINGS))

    # Radiance materials: lambert baseline; wrappers normalized for optics
    rad_lambert = {L: _module_radiance_umol_per_sr_m2(L) for L in range(rings_local)}
    use_optics = (OPTICS_MODE == "lens")
    if use_optics:
        _write_beam_cal_base(CAL_FILE)

    out = OUT_DIR / "emitters_smd_ALL_umol.rad"
    with out.open("w") as fh:
        fh.write("# SMD macro emitters in µmol/s/sr/m²\n")
        if layout_mode == "rect_rect" and pitch_x and pitch_y:
            if abs(pitch_x - pitch_y) < 1e-6:
                fh.write(f"# rings={rings_local} modules={len(positions)} pitch={pitch_x:.4f} m (uniform; outer margin {WALL_MARGIN_M:.3f} m)\n")
            else:
                fh.write(f"# rings={rings_local} modules={len(positions)} pitch_x={pitch_x:.4f} m pitch_y={pitch_y:.4f} m (outer margin {WALL_MARGIN_M:.3f} m)\n")
        else:
            fh.write(f"# rings={rings_local} modules={len(positions)} spacing={spacing:.4f} m (outer margin {WALL_MARGIN_M:.3f} m)\n")
        fh.write(f"# patch side = {PATCH_SIDE_M*1e3:.1f} mm  subgrid = {SUBPATCH_GRID}x{SUBPATCH_GRID}\n")
        fh.write(f"# optics = {'enabled (normalized patterns)' if use_optics else 'disabled (Lambertian)'}\n\n")

        for idx, p in enumerate(positions):
            L, cx, cy, cz = int(p["ring"]), float(p["x"]), float(p["y"]), float(p["z"])
            base_rad = rad_lambert[L]
            patt = None

            if use_optics and L in LENS_RINGS and L > 0:
                kind, cfg = LENS_RINGS[L]
                if kind == "sym":
                    m = _m_from_fwhm(float(cfg["fwhm"]))
                    gain = 1.0 / max(_avg_sym_cos(m), EPS)
                    fname = f"f_sym_L{L}"
                    _add_wrapper(fname, f"{gain:.9f} * pow(c, {m:.9f})")
                    patt = f"p_sym_L{L}"; _write_brightfunc_ref(fh, patt, fname)

                elif kind == "bat":
                    m = _m_from_fwhm(float(cfg["fwhm"]))
                    k = float(cfg.get("bat_k", 0.75))
                    gain = 1.0 / max(_avg_bat_cos(m, k), EPS)
                    fname = f"f_bat_L{L}"
                    _add_wrapper(fname, f"{gain:.9f} * pow(c, {m:.9f}) * (1 + {k:.9f}*(1 - 4*c + 4*c*c))")
                    patt = f"p_bat_L{L}"; _write_brightfunc_ref(fh, patt, fname)

                elif kind == "ellip":
                    ex, ey = float(cfg["ex"]), float(cfg["ey"])
                    avg = _avg_ellip_cos(ex, ey, samples=4000)
                    gain = 1.0 / max(avg, EPS)
                    tx0 = math.tan(math.radians(ex*0.5))
                    ty0 = math.tan(math.radians(ey*0.5))
                    rlen = math.hypot(cx, cy)
                    ux, uy = (1.0, 0.0) if rlen < 1e-9 else (cx/rlen, cy/rlen)
                    fname = f"f_ellip_L{L}_m{idx:03d}"
                    expr = (
                        f"{gain:.9f} * exp(-0.6931471805599453 * ("
                        f" ((abs(Dx*{ux:.9f}+Dy*{uy:.9f})/(abs(Dz)+eps))/{tx0:.9f})^2 +"
                        f" ((abs(-Dx*{uy:.9f}+Dy*{ux:.9f})/(abs(Dz)+eps))/{ty0:.9f})^2 ))"
                    )
                    _add_wrapper(fname, expr)
                    patt = f"p_ell_L{L}_m{idx:03d}"
                    _write_brightfunc_ref(fh, patt, fname)

                elif kind == "ellip_mix" and L == 7:
                    i = int(p.get("i", 0))
                    j = int(p.get("j", 0))
                    # In this scheme, "corners" are the axial extremes (±L, 0) or (0, ±L)
                    is_corner = (abs(i) == L and j == 0) or (abs(j) == L and i == 0)
                    ex = float(cfg["corner_ex"] if is_corner else cfg["edge_ex"])
                    ey = float(cfg["corner_ey"] if is_corner else cfg["edge_ey"])
                    avg = _avg_ellip_cos(ex, ey, samples=4000)
                    gain = 1.0 / max(avg, EPS)
                    tx0 = math.tan(math.radians(ex*0.5))
                    ty0 = math.tan(math.radians(ey*0.5))
                    rlen = math.hypot(cx, cy)
                    ux, uy = (1.0, 0.0) if rlen < 1e-9 else (cx/rlen, cy/rlen)
                    tag = "C" if is_corner else "E"
                    fname = f"f_ellip_L7_{tag}_m{idx:03d}"
                    expr = (
                        f"{gain:.9f} * exp(-0.6931471805599453 * ("
                        f" ((abs(Dx*{ux:.9f}+Dy*{uy:.9f})/(abs(Dz)+eps))/{tx0:.9f})^2 +"
                        f" ((abs(-Dx*{uy:.9f}+Dy*{ux:.9f})/(abs(Dz)+eps))/{ty0:.9f})^2 ))"
                    )
                    _add_wrapper(fname, expr)
                    patt = f"p_ell_L7_{tag}_m{idx:03d}"
                    _write_brightfunc_ref(fh, patt, fname)

            mat = f"smd_L{L}_m{idx:03d}"
            if patt:
                fh.write(f"{patt} light {mat}\n0\n0\n3 {base_rad:.6f} {base_rad:.6f} {base_rad:.6f}\n\n")
            else:
                fh.write(f"void light {mat}\n0\n0\n3 {base_rad:.6f} {base_rad:.6f} {base_rad:.6f}\n\n")

            _write_area_grid(fh, mat, cx, cy, cz, PATCH_SIDE_M, SUBPATCH_GRID)

    if use_optics:
        _flush_wrappers()

    # Per-ring summary (count = 1 + 4L)
    from collections import Counter

    # Actual module counts per ring from the generated positions
    ring_counts = Counter(p["ring"] for p in positions)

    derate_desc = (
        f"  derate      : {DERATE:.4f} (= user_scale(EFF_SCALE env) {USER_EFF_SCALE:.3f})\n"
        if PPE_IS_SYSTEM
        else
        f"  derate      : {DERATE:.4f} (= DRIVER_EFF {DRIVER_EFF:.3f} × THERMAL_EFF {THERMAL_EFF:.3f} × "
        f"BOARD_OPT_EFF {BOARD_OPT_EFF:.3f} × WIRING_EFF {WIRING_EFF:.3f} × user_scale(EFF_SCALE env) {USER_EFF_SCALE:.3f})\n"
    )

    total_w_in = 0.0
    total_w_eff = 0.0
    lines = []
    for L in range(rings_local):
        nL = ring_counts.get(L, 0)
        w_in = RING_POWER_W.get(L, RING_POWER_W[max(RING_POWER_W.keys())])
        w_eff = w_in * EFF_SCALE
        total_w_in += nL * w_in
        total_w_eff += nL * w_eff
        lines.append(f"    ring {L}: {w_in:5.2f} W in ({w_eff:5.2f} W eff) × {nL} mods")

    avg_ppe = (
        sum(COUNT_PER_MODULE[ch]*PER_LED_W[ch]*PPE_UMOL_PER_J[ch]
            for ch in COUNT_PER_MODULE)
        / max(_per_module_nominal_watts(), 1e-9)
    )

    ppe_parts = []
    for ch in sorted(COUNT_PER_MODULE.keys()):
        if COUNT_PER_MODULE.get(ch, 0) <= 0:
            continue
        ppe_parts.append(f"{ch}={PPE_UMOL_PER_J[ch]:.3f}")
    ppe_str = ", ".join(ppe_parts) if ppe_parts else "(none)"

    # Total photons based on actual ring counts
    total_umol = sum(
        _module_photon_flux_umol_s(L) * ring_counts.get(L, 0)
        for L in range(rings_local)
    )

    summ = OUT_DIR / "smd_summary.txt"
    spacing_str = f"{spacing:.4f} m"
    if layout_mode == "rect_rect" and pitch_x and pitch_y:
        if abs(pitch_x - pitch_y) < 1e-6:
            spacing_str = f"pitch={pitch_x:.4f} m (uniform)"
        else:
            spacing_str = f"pitch_x={pitch_x:.4f} m, pitch_y={pitch_y:.4f} m"
    summ.write_text(
        "SMD macro emitter summary:\n"
        f"  rings       : {rings_local}  (center=ring 0)\n"
        f"  modules     : {len(positions)}  (A001844)\n"
        f"  spacing     : {spacing_str}  (outer margin {WALL_MARGIN_M:.3f} m)\n"
        f"  patch side  : {PATCH_SIDE_M*1e3:.1f} mm\n"
        f"  optics      : {'enabled (normalized patterns)' if use_optics else 'disabled (Lambertian)'}\n"
        f"  subgrid     : {SUBPATCH_GRID}x{SUBPATCH_GRID}\n"
        f"  ring powers : {RING_POWERS_SOURCE}\n"
        f"  PPE_IS_SYSTEM: {int(PPE_IS_SYSTEM)}  (1=fixture-level PPE; 0=apply driver/thermal/optical derates)\n"
        + derate_desc
        + ("  per-ring watts (per module, input → effective):\n" + "\n".join(lines) + "\n")
        + f"  total electrical input ≈ {total_w_in:.1f} W\n"
        + f"  total effective (derated) ≈ {total_w_eff:.1f} W\n"
        f"  PPE by channel (µmol/J): {ppe_str}\n"
        f"  avg PPE (mix) ≈ {avg_ppe:.3f} µmol/J → total photons ≈ {total_umol:.0f} µmol/s\n"
    )

    print("SMD macro emitter summary:")
    print(f"  rings       : {rings_local}  (center=ring 0)")
    print(f"  modules     : {len(positions)}  (A001844)")
    print(f"  spacing     : {spacing_str}  (outer margin {WALL_MARGIN_M:.3f} m)")
    print(f"  patch side  : {PATCH_SIDE_M*1e3:.1f} mm")
    print(f"  optics      : {'enabled (normalized patterns)' if use_optics else 'disabled (Lambertian)'}")
    print(f"  subgrid     : {SUBPATCH_GRID}x{SUBPATCH_GRID}")
    print(f"  ring powers : {RING_POWERS_SOURCE}")
    print("  per-ring watts (per module):"); [print(s) for s in lines]
    print(f"  PPE_IS_SYSTEM: {int(PPE_IS_SYSTEM)}  (1=fixture-level PPE; 0=apply driver/thermal/optical derates)")
    print(derate_desc.rstrip())
    print(f"  total electrical input ≈ {total_w_in:.1f} W")
    print(f"  total effective (derated) ≈ {total_w_eff:.1f} W")
    print(f"  PPE by channel (µmol/J): {ppe_str}")
    print(f"  avg PPE (mix) ≈ {avg_ppe:.3f} µmol/J → total photons ≈ {total_umol:.0f} µmol/s")
    print(f"✔ Wrote {out}")
    if use_optics: print(f"✔ Wrote {CAL_FILE}")
    print(f"✔ Wrote {summ}")

    # Authoritative snapshot for overlays/GUI (meters)
    write_smd_layout_json(
        positions=positions,
        spacing_m=spacing,
        room_L_m=LENGTH_M,
        room_W_m=WIDTH_M,
        margin_m=WALL_MARGIN_M,
        ring_n=meta.get("ring_n", RING_N),
        patch_side_m=PATCH_SIDE_M,
        z_m=MOUNT_Z_M,
        extra={
            "source": "generate_emitters_smd.py",
            "layout_mode": layout_mode,
            "pitch_x_m": pitch_x,
            "pitch_y_m": pitch_y,
            "swapped_axes": meta.get("swapped_axes", False),
            "rings": rings_local,
        }
    )
    print("✔ Wrote ies_sources/smd_layout.json")

if __name__ == "__main__":
    main()
