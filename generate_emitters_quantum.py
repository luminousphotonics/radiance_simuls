#!/usr/bin/env python3
# generate_emitters_quantum.py • Quantum Board macro emitters (Lambertian)
# Independent of generate_emitters_smd.py to avoid regressions to the SMD path.
# Layout/ring structure mirrors the SMD solver (diamond rings, centered), but
# module geometry/power are set to the 6.833" x 11.25" quantum board.

from __future__ import annotations
import json, math, os
from pathlib import Path
from typing import Dict, List, Tuple, Any
from rect_layout import _generate_rect_uv

PI = math.pi
SQRT2 = math.sqrt(2.0)
EPS = 1e-12
CLAMP_SAFE = 1e-4  # pull layouts slightly off the wall to avoid bleed

# ── Env helpers ──────────────────────────────────────────────────────────────
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

def _maybe_swap_dims(length_m: float, width_m: float) -> tuple[float, float, bool]:
    """
    Align long axis to +X when ALIGN_LONG_AXIS_X=1 to stay consistent with room/grid.
    """
    align = os.getenv("ALIGN_LONG_AXIS_X", "1") == "1"
    if align and width_m > length_m:
        return width_m, length_m, True
    return length_m, width_m, False

# ── Geometry (room) ──────────────────────────────────────────────────────────
LENGTH_M      = _env_float("LENGTH_FT", 12.0) * 0.3048
WIDTH_M       = _env_float("WIDTH_FT",  12.0) * 0.3048
HEIGHT_M      = _env_float("HEIGHT_M", 3.048)
WALL_MARGIN_M = _env_float("MARGIN_IN", 0.0)  * 0.0254
LAYOUT_MODE   = os.getenv("LAYOUT_MODE", "square").strip().lower()

# ── Quantum board footprint (meters) ─────────────────────────────────────────
# Default orientation: long edge along +X for a more "horizontal" layout.
PATCH_X_M     = _env_float("PATCH_X_M", 11.25 * 0.0254)   # ≈0.2858 m (long)
PATCH_Y_M     = _env_float("PATCH_Y_M",  6.833 * 0.0254)  # ≈0.1736 m (short)
PATCH_MAX_M   = max(PATCH_X_M, PATCH_Y_M)                 # used for spacing guard
SUBPATCH_GRID = _env_int("SUBPATCH_GRID", 3)
PERIM_INSET_M = _env_float("QB_PERIM_INSET_M", 0.02)

# Rings: same structure as SMD solver
RING_N        = _env_int("QB_RING_N", 7)
RINGS         = max(1, RING_N + 1)

# Mount height
MOUNT_Z_M     = _env_float("MOUNT_Z_M", 0.4572)

# ── Electrical / spectral model (mapped to existing channels) ───────────────
COUNT_PER_MODULE: Dict[str, int] = {
    "WW": 128, "CW": 128, "R": 36, "FR": 0, "B": 0, "C": 0, "UV": 0,
}
PER_LED_W: Dict[str, float] = {
    "WW": 0.60, "CW": 0.60, "R": 0.45, "B": 0.35, "C": 0.35, "FR": 0.35, "UV": 0.25,
}
PPE_UMOL_PER_J: Dict[str, float] = {
    "WW": 3.30, "CW": 3.30, "R": 3.30, "B": 2.4, "C": 2.0, "FR": 3.6, "UV": 1.0,
}

# Per-ring electrical watts (per module). Scaled from SMD baseline by ~5.04×.
RING_POWER_W: Dict[int, float] = {
    0: 160.4, 1: 142.544, 2: 167.088, 3: 152.407, 4: 130.529,
    5: 225.393, 6: 18.123, 7: 356.612, 8: 356.612
}
RING_POWERS_SOURCE = "quantum_builtin"

# Global derating
DRIVER_EFF     = _env_float("DRIVER_EFF",     0.95)
THERMAL_EFF    = _env_float("THERMAL_EFF",    0.92)
BOARD_OPT_EFF  = _env_float("BOARD_OPT_EFF",  0.95)
WIRING_EFF     = _env_float("WIRING_EFF",     0.99)
USER_EFF_SCALE = _env_float("EFF_SCALE",      1.00)
DERATE = DRIVER_EFF * THERMAL_EFF * BOARD_OPT_EFF * WIRING_EFF * USER_EFF_SCALE
EFF_SCALE = DERATE

# Basis mode (independent from SMD env names)
BASIS_MODE      = os.getenv("QB_BASIS_MODE", "0") == "1"
BASIS_RING      = int(os.getenv("QB_BASIS_RING", "-1"))
BASIS_UNIT_W    = float(os.getenv("QB_BASIS_UNIT_W", "1.0"))

if BASIS_MODE:
    if BASIS_RING >= 0:
        for L in range(RINGS):
            if L == BASIS_RING:
                RING_POWER_W[L] = BASIS_UNIT_W
            else:
                RING_POWER_W[L] = 0.0
        RING_POWERS_SOURCE = f"basis_mode_ring_{BASIS_RING}"

def _apply_ring_powers_override():
    """Allow per-run overrides via JSON (USE_RING_POWERS_JSON=1, RING_POWERS_JSON=path)."""
    global RING_POWERS_SOURCE
    if BASIS_MODE:
        return
    use_json = os.getenv("USE_RING_POWERS_JSON", "1") != "0"
    path = Path(os.getenv("RING_POWERS_JSON", "ring_powers_quantum.json"))
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

# ── Layout helpers ───────────────────────────────────────────────────────────
def _solve_spacing(rings: int, length_m: float, width_m: float,
                   wall_margin_m: float, patch_side_m: float) -> float:
    """Diamond lattice spacing based on max patch dimension."""
    L = max(1, rings - 1)
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
    return ((i - j) * spacing / SQRT2, (i + j) * spacing / SQRT2)

def _compute_positions_from_env() -> Tuple[List[dict], float, dict]:
    L_m = _env_float("LENGTH_M", _env_float("LENGTH_FT", 12.0) * 0.3048)
    W_m = _env_float("WIDTH_M",  _env_float("WIDTH_FT",  12.0) * 0.3048)
    L_m, W_m, swapped = _maybe_swap_dims(L_m, W_m)

    margin_m = WALL_MARGIN_M
    patch_eff = PATCH_MAX_M
    layout_mode = os.environ.get("LAYOUT_MODE", LAYOUT_MODE).strip().lower()

    # Max allowable center coordinate (keeps the full board inside the room + margin).
    usable_half_x = (L_m * 0.5) - margin_m - 0.5 * PATCH_X_M - CLAMP_SAFE
    usable_half_y = (W_m * 0.5) - margin_m - 0.5 * PATCH_Y_M - CLAMP_SAFE
    if usable_half_x <= 0 or usable_half_y <= 0:
        raise SystemExit("ERROR: negative/zero usable half-span; check margin/patch/room.")

    # Match SMD ring logic:
    #   - square rooms: diamond rings (ring counts 4*L)
    #   - rectangular rooms: central spine + rectangular UV rings (rect_layout._generate_rect_uv)
    #
    # Pitch is chosen from a preferred spacing (QB_PITCH_SCALE) and then adjusted so
    # the outermost geometry lands on the usable bounds.
    scale = max(_env_float("QB_PITCH_SCALE", 2.0), 1e-6)
    rectangular_room = abs(L_m - W_m) > 1e-6

    pos: List[dict] = []
    z0 = MOUNT_Z_M

    if not rectangular_room:
        # Square: diamond rings in (i,j), centered at origin.
        half_min = min(usable_half_x, usable_half_y)
        pitch_pref = max(PATCH_Y_M * scale, 1e-9)
        ring_n = max(1, int(round(half_min / pitch_pref)))
        pitch_axis = half_min / ring_n
        spacing_diag = pitch_axis * SQRT2

        for L in range(ring_n + 1):
            for i, j in _ring_ij(L):
                x, y = _ij_to_xy(i, j, spacing_diag)
                pos.append(
                    {
                        "ring": int(L),
                        "i": int(i),
                        "j": int(j),
                        "x": round(float(x), 6),
                        "y": round(float(y), 6),
                        "z": float(z0),
                        "lx": float(PATCH_X_M),
                        "ly": float(PATCH_Y_M),
                    }
                )

        rings_local = int(ring_n + 1)
        pitch_x = pitch_y = float(pitch_axis)
        spacing_eff = float(pitch_axis)
        offset = 0
        u_max = ring_n
        v_max = ring_n
        layout_mode_effective = "square"
    else:
        # Rectangular: use rect_layout UV pattern, then scale pitch_x/pitch_y to fill.
        pitch_y_pref = max(PATCH_Y_M * scale, 1e-9)
        ring_n = max(1, int(round(usable_half_y / pitch_y_pref)))

        # In rect_uv, adjacent emitters in a row are separated by Δu=2, so the
        # effective x-to-x spacing is ~2*pitch_x. Use pitch_x_pref = PATCH_X_M/2.
        pitch_x_pref = max(0.5 * PATCH_X_M * scale, 1e-9)
        u_max = max(1, int(round(usable_half_x / pitch_x_pref)))
        if (u_max % 2) != (ring_n % 2):
            u_max = max(1, u_max - 1)

        pitch_x = usable_half_x / u_max
        pitch_y = usable_half_y / ring_n
        offset = max(0, int(u_max - ring_n))

        uv_points = _generate_rect_uv(ring_n=ring_n, offset=offset)
        for u, v, ring in uv_points:
            x = u * pitch_x
            y = v * pitch_y
            pos.append(
                {
                    "ring": int(ring),
                    "i": int(u),
                    "j": int(v),
                    "x": round(float(x), 6),
                    "y": round(float(y), 6),
                    "z": float(z0),
                    "lx": float(PATCH_X_M),
                    "ly": float(PATCH_Y_M),
                }
            )

        rings_local = int(ring_n + 1)
        spacing_eff = float(max(pitch_x, pitch_y))
        v_max = ring_n
        layout_mode_effective = "rect_rect"

    xs = [p["x"] for p in pos] if pos else [0.0]
    ys = [p["y"] for p in pos] if pos else [0.0]
    meta = {
        "room_L_m": float(L_m),
        "room_W_m": float(W_m),
        "margin_m": float(margin_m),
        "patch_side_m": float(patch_eff),
        "patch_x_m": float(PATCH_X_M),
        "patch_y_m": float(PATCH_Y_M),
        "ring_n": int(rings_local - 1),
        "rings": int(rings_local),
        "spacing_m": float(spacing_eff),
        "d_axis_m": float(spacing_eff),
        "layout_mode": layout_mode_effective,
        "source_layout_mode": layout_mode,
        "ring_mode": "diamond" if not rectangular_room else "rect_uv",
        "qb_pitch_scale": float(scale),
        "offset": int(offset),
        "u_max": int(u_max),
        "v_max": int(v_max),
        "pitch_x_m": float(pitch_x),
        "pitch_y_m": float(pitch_y),
        "pitch_m": float(0.5 * (pitch_x + pitch_y)),
        "x_span_m": float(max(xs) - min(xs)),
        "y_span_m": float(max(ys) - min(ys)),
        "usable_half_x_m": float(usable_half_x),
        "usable_half_y_m": float(usable_half_y),
        "swapped_axes": bool(swapped),
    }
    return pos, spacing_eff, meta

def get_module_positions() -> Tuple[List[Dict], float]:
    positions, spacing, _ = _compute_positions_from_env()
    return positions, spacing

# ── Power / radiance ────────────────────────────────────────────────────────
def _per_module_nominal_watts() -> float:
    return sum(COUNT_PER_MODULE[ch] * PER_LED_W[ch] for ch in COUNT_PER_MODULE)

def _module_photon_flux_umol_s(ring: int) -> float:
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
    A = PATCH_X_M * PATCH_Y_M
    phi = _module_photon_flux_umol_s(ring)
    return 0.0 if A <= 0 else phi / (A * PI)  # Lambertian

# ── RAD writers ─────────────────────────────────────────────────────────────
def _write_area_rect(fh, mat: str, cx: float, cy: float, z: float, lx: float, ly: float):
    hx = lx * 0.5; hy = ly * 0.5
    fh.write(f"{mat} polygon poly_{abs(hash((cx,cy)))%10**8}\n0\n0\n12\n")
    fh.write(f"  {cx - hx:.6f} {cy + hy:.6f} {z:.6f}\n")
    fh.write(f"  {cx + hx:.6f} {cy + hy:.6f} {z:.6f}\n")
    fh.write(f"  {cx + hx:.6f} {cy - hy:.6f} {z:.6f}\n")
    fh.write(f"  {cx - hx:.6f} {cy - hy:.6f} {z:.6f}\n\n")

def _write_area_grid_rect(fh, mat: str, cx: float, cy: float, z: float, lx: float, ly: float, grid: int):
    if grid <= 1:
        _write_area_rect(fh, mat, cx, cy, z, lx, ly); return
    cell_x = lx / grid
    cell_y = ly / grid
    start_x = -0.5 * lx + 0.5 * cell_x
    start_y = -0.5 * ly + 0.5 * cell_y
    for r in range(grid):
        for c in range(grid):
            px = cx + start_x + c * cell_x
            py = cy + start_y + r * cell_y
            _write_area_rect(fh, mat, px, py, z, cell_x, cell_y)

# ── Snapshot writer ─────────────────────────────────────────────────────────
OUT_DIR  = Path("ies_sources"); OUT_DIR.mkdir(parents=True, exist_ok=True)
CAL_NAME = "beam.cal"  # unused but kept for parity
SMD_JSON = OUT_DIR / "quantum_layout.json"

def write_layout_json(
    positions: List[Dict[str, float]],
    spacing_m: float,
    room_L_m: float,
    room_W_m: float,
    margin_m: float,
    ring_n: int,
    patch_x_m: float,
    patch_y_m: float,
    z_m: float = 0.0,
    extra: Dict[str, Any] | None = None
) -> None:
    from datetime import datetime
    def _rf(v, p=6): return float(round(float(v), p))
    SMD_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "generator": "generate_emitters_quantum.py",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "units": "meters",
        "rings": int(ring_n + 1),
        "modules": int(len(positions)),
        "spacing": _rf(spacing_m),
        "patch_x": _rf(patch_x_m),
        "patch_y": _rf(patch_y_m),
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

# ── Main generation ─────────────────────────────────────────────────────────
def main():
    positions, spacing, meta = _compute_positions_from_env()
    rings_local = int(meta.get("rings", RINGS))

    rad_lambert = {L: _module_radiance_umol_per_sr_m2(L) for L in range(rings_local)}

    out = OUT_DIR / "emitters_quantum_ALL_umol.rad"
    with out.open("w") as fh:
        fh.write("# Quantum Board macro emitters in µmol/s/sr/m² (Lambertian)\n")
        fh.write(f"# rings={rings_local} modules={len(positions)} spacing={spacing:.4f} m (outer margin {WALL_MARGIN_M:.3f} m)\n")
        fh.write(f"# patch_x={PATCH_X_M:.4f} m  patch_y={PATCH_Y_M:.4f} m  subgrid={SUBPATCH_GRID}x{SUBPATCH_GRID}\n\n")

        for idx, p in enumerate(positions):
            L, cx, cy, cz = int(p["ring"]), float(p["x"]), float(p["y"]), float(p["z"])
            lx = float(p.get("lx", PATCH_X_M)); ly = float(p.get("ly", PATCH_Y_M))
            base_rad = rad_lambert[L]
            mat = f"qb_L{L}_m{idx:03d}"
            fh.write(f"void light {mat}\n0\n0\n3 {base_rad:.6f} {base_rad:.6f} {base_rad:.6f}\n\n")
            _write_area_grid_rect(fh, mat, cx, cy, cz, lx, ly, SUBPATCH_GRID)

    # Per-ring summary
    from collections import Counter
    ring_counts = Counter(p["ring"] for p in positions)
    total_w_in = 0.0
    total_w_eff = 0.0
    lines = []
    for L in range(rings_local):
        nL = ring_counts.get(L, 0)
        w_in = RING_POWER_W.get(L, RING_POWER_W[max(RING_POWER_W.keys())])
        w_eff = w_in * EFF_SCALE
        total_w_in += nL * w_in
        total_w_eff += nL * w_eff
        lines.append(f"    ring {L}: {w_in:6.2f} W in ({w_eff:6.2f} W eff) × {nL} boards")

    avg_ppe = (
        sum(COUNT_PER_MODULE[ch]*PER_LED_W[ch]*PPE_UMOL_PER_J[ch]
            for ch in COUNT_PER_MODULE)
        / max(_per_module_nominal_watts(), 1e-9)
    )

    total_umol = sum(
        _module_photon_flux_umol_s(L) * ring_counts.get(L, 0)
        for L in range(rings_local)
    )

    summ = OUT_DIR / "quantum_summary.txt"
    summ.write_text(
        "Quantum Board emitter summary:\n"
        f"  rings       : {rings_local}  (center=ring 0)\n"
        f"  modules     : {len(positions)}\n"
        f"  spacing     : {spacing:.4f} m (outer margin {WALL_MARGIN_M:.3f} m)\n"
        f"  patch_x/y   : {PATCH_X_M*1e3:.1f} mm × {PATCH_Y_M*1e3:.1f} mm\n"
        f"  subgrid     : {SUBPATCH_GRID}x{SUBPATCH_GRID}\n"
        f"  ring powers : {RING_POWERS_SOURCE}\n"
        f"  derate      : {DERATE:.4f} (= DRIVER_EFF {DRIVER_EFF:.3f} × THERMAL_EFF {THERMAL_EFF:.3f} × "
        f"BOARD_OPT_EFF {BOARD_OPT_EFF:.3f} × WIRING_EFF {WIRING_EFF:.3f} × user_scale(EFF_SCALE env) {USER_EFF_SCALE:.3f})\n"
        f"  per-ring watts (per module, input → effective):\n" + "\n".join(lines) + "\n"
        f"  total electrical input ≈ {total_w_in:.1f} W\n"
        f"  total effective (derated) ≈ {total_w_eff:.1f} W\n"
        f"  avg PPE (mix) ≈ {avg_ppe:.3f} µmol/J → total photons ≈ {total_umol:.0f} µmol/s\n"
    )

    print("Quantum Board emitter summary:")
    print(f"  rings       : {rings_local}  (center=ring 0)")
    print(f"  modules     : {len(positions)}")
    print(f"  spacing     : {spacing:.4f} m (outer margin {WALL_MARGIN_M:.3f} m)")
    print(f"  patch_x/y   : {PATCH_X_M*1e3:.1f} mm × {PATCH_Y_M*1e3:.1f} mm")
    print(f"  subgrid     : {SUBPATCH_GRID}x{SUBPATCH_GRID}")
    print(f"  ring powers : {RING_POWERS_SOURCE}")
    print(f"  derate      : {DERATE:.4f} (= DRIVER_EFF {DRIVER_EFF:.3f} × THERMAL_EFF {THERMAL_EFF:.3f} × "
          f"BOARD_OPT_EFF {BOARD_OPT_EFF:.3f} × WIRING_EFF {WIRING_EFF:.3f} × user_scale(EFF_SCALE env) {USER_EFF_SCALE:.3f})")
    print("  per-ring watts (per module, input → effective):"); [print(s) for s in lines]
    print(f"  total electrical input ≈ {total_w_in:.1f} W")
    print(f"  total effective (derated) ≈ {total_w_eff:.1f} W")
    print(f"  avg PPE (mix) ≈ {avg_ppe:.3f} µmol/J → total photons ≈ {total_umol:.0f} µmol/s")
    print(f"✔ Wrote {out}")
    print(f"✔ Wrote {summ}")

    # Snapshot for overlays
    write_layout_json(
        positions=positions,
        spacing_m=spacing,
        room_L_m=LENGTH_M,
        room_W_m=WIDTH_M,
        margin_m=WALL_MARGIN_M,
        ring_n=meta.get("ring_n", RING_N),
        patch_x_m=PATCH_X_M,
        patch_y_m=PATCH_Y_M,
        z_m=MOUNT_Z_M,
        extra={
            "source": "generate_emitters_quantum.py",
            "layout_mode": meta.get("layout_mode", LAYOUT_MODE),
            "pitch_x_m": meta.get("pitch_x_m"),
            "pitch_y_m": meta.get("pitch_y_m"),
            "rings": rings_local,
        }
    )
    print("✔ Wrote ies_sources/quantum_layout.json")

if __name__ == "__main__":
    main()
