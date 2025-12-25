#!/usr/bin/env python3
# generate_emitters_cob.py • COB + halo emitters (Lambertian + optional optics)
#
# Outputs:
#   - ies_sources/emitters_cob_ALL_umol.rad
#   - ies_sources/cob_layout.json
#   - overlays/strip_overlay.rad
#
# Notes:
#   - Emitters are written in "µmol/s/sr/m²" units (RGB channels are identical).
#   - COBs are modeled as Lambertian circular disks (polygon approximation).
#   - Halos are Lambertian rectangular area emitters placed around each COB heatsink.
#   - Halos are NOT solver-controlled in the default workflow; they are emitted as a fixed fraction
#     of each ring’s COB watts (supplemental 660nm) so they don’t destabilize uniformity.

from __future__ import annotations

import json
import math
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


PI = math.pi
SQRT2 = math.sqrt(2.0)
EPS = 1e-12
CLAMP_SAFE = 1e-4  # pull layouts slightly off the wall to avoid bleed


# ── Physical defaults (user-tweakable later) ─────────────────────────────────
# COB: Bridgelux Vero 22 Gen 8 Array, 3500K (LES modeled as emitting disk)
COB_LES_DIAM_M = 0.0233
COB_LES_R_M = 0.5 * COB_LES_DIAM_M
COB_HEATSINK_DIAM_IN_DEFAULT = 5.0  # mechanical placement model (halo perimeter)
COB_HALO_BAR_IN_DEFAULT = 0.1875  # 3/16" square bar for halo segments
FRAME_ANGLE_LEG_IN_DEFAULT = 0.125  # 1/8" x 1/8" angle
COB_VF_TYP_V = 50.7
COB_IF_TYP_A = 1.2
COB_IF_MAX_A = 3.0
COB_LM_PER_W_REF = 186.0
COB_PPE_UMOL_PER_J = 2.76  # fixture-level placeholder (white)

# Strips: KingBrite W55 style (perimeter mode only)
STRIP_W_M = 0.055
STRIP_LONG_L_M = 0.510
STRIP_SHORT_L_M = 0.410
STRIP_LONG_ELEC_MAX_W = 91.2
STRIP_SHORT_ELEC_MAX_W = 73.0
STRIP_PPE_UMOL_PER_J = 2.76  # placeholder; user can tune later

# Simple strip power allocation (placeholder)
STRIP_W_PER_M = 120.0  # W/m

# Geometry / placement tweaks
COB_DISK_SIDES = 16
STRIP_CLEARANCE_M = 0.005  # outward offset so strips don't overlap fixture body
STRIP_MIN_RING_DEFAULT = 1  # exclude the center singleton; first strips start at ring 1

# Halo power is proportional to COB power per ring (supplemental spectrum, not uniformity driver).
COB_STRIP_W_FRACTION_DEFAULT = 0.111  # halo watts = fraction * COB ring total watts

# Physical clearance: COB centers must be at least this far from the wall (heatsink clearance).
COB_WALL_CLEARANCE_IN_DEFAULT = 3.0

# Optics defaults (optional, per-ring)
COB_OPTICS_DEFAULT = "outer"  # none|outer|all|<list>
COB_OPTICS_KIND_DEFAULT = "sym"  # symmetric beam shaping
COB_OPTICS_FWHM_DEG_DEFAULT = 60.0
CAL_NAME = "beam.cal"


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

def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return bool(default)
    return str(v).strip().lower() not in ("0", "false", "no", "off", "")

def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    if v is None:
        return str(default)
    v = str(v).strip()
    return v if v else str(default)


def _maybe_swap_dims(length_m: float, width_m: float) -> tuple[float, float, bool]:
    align = os.getenv("ALIGN_LONG_AXIS_X", "1") == "1"
    if align and width_m > length_m:
        return width_m, length_m, True
    return length_m, width_m, False


# ── Layout helpers (diamond rings) ───────────────────────────────────────────
def _ring_ij(L: int) -> List[Tuple[int, int]]:
    if L == 0:
        return [(0, 0)]
    pts: List[Tuple[int, int]] = []
    i = L
    j = 0
    for k in range(L):
        pts.append((i - k, j - k))
    for k in range(L):
        pts.append((0 - k, -L + k))
    for k in range(L):
        pts.append((-L + k, 0 + k))
    for k in range(L):
        pts.append((0 + k, L - k))
    return pts


def _generate_rect_uv_staggered(ring_n: int, offset: int) -> List[Tuple[int, int, int]]:
    """
    Rectangular UV rings on the diamond-staggered lattice (u+v even).
      - ring 0: horizontal spine from u=-offset..+offset (step 2), v=0
      - rings 1..ring_n: rectangular perimeters with u_max = offset + k, v_max = k
    """
    pts: List[Tuple[int, int, int]] = []

    for u in range(-offset, offset + 1, 2):
        pts.append((u, 0, 0))

    for k in range(1, ring_n + 1):
        u_max = offset + k
        v_max = k
        for u in range(-u_max, u_max + 1):
            for v in range(-v_max, v_max + 1):
                if not (abs(u) == u_max or abs(v) == v_max):
                    continue
                if (u + v) % 2 != 0:
                    continue
                pts.append((u, v, k))

    dedup: Dict[Tuple[int, int], int] = {}
    for u, v, ring in pts:
        key = (u, v)
        if key not in dedup or ring < dedup[key]:
            dedup[key] = ring

    return [(u, v, ring) for (u, v), ring in dedup.items()]


def _ij_to_xy(i: int, j: int, spacing_diag: float) -> Tuple[float, float]:
    return ((i - j) * spacing_diag / SQRT2, (i + j) * spacing_diag / SQRT2)




def _layout_mode() -> str:
    return os.getenv("LAYOUT_MODE", "square").strip().lower()


def _parse_ring_spec(spec: str, ring_max: int) -> List[int]:
    s = str(spec or "").strip().lower()
    if not s:
        return []
    if s in ("none", "off", "0", "false"):
        return []
    if s in ("outer", "perimeter", "edge"):
        return [ring_max] if ring_max > 0 else []
    if s in ("all", "on", "1", "true"):
        return list(range(1, ring_max + 1))

    out: List[int] = []
    for raw in s.replace(";", ",").replace(" ", ",").split(","):
        tok = raw.strip()
        if not tok:
            continue
        if "-" in tok:
            parts = tok.split("-", 1)
            try:
                a = int(parts[0].strip())
                b = int(parts[1].strip())
            except Exception:
                continue
            if b < a:
                a, b = b, a
            out.extend(range(a, b + 1))
        else:
            try:
                out.append(int(tok))
            except Exception:
                continue

    cleaned = sorted({r for r in out if 0 <= r <= ring_max})
    return cleaned


def _optics_settings(ring_max: int) -> Dict[str, Any]:
    rings_spec = _env_str("COB_OPTICS_RINGS", "")
    mode = _env_str("COB_OPTICS", COB_OPTICS_DEFAULT).lower()
    rings = _parse_ring_spec(rings_spec, ring_max) if rings_spec else _parse_ring_spec(mode, ring_max)
    kind = _env_str("COB_OPTICS_KIND", COB_OPTICS_KIND_DEFAULT).lower()
    if kind not in ("sym",):
        kind = "sym"
    fwhm = _env_float("COB_OPTICS_FWHM_DEG", COB_OPTICS_FWHM_DEG_DEFAULT)
    return {
        "mode": mode,
        "rings": rings,
        "kind": kind,
        "fwhm_deg": fwhm,
    }


def _compute_positions_from_env() -> Tuple[List[Dict[str, float]], float, Dict[str, Any]]:
    """
    Returns (fixture_centers, pitch_axis_m, meta).

    - square: diamond-ring layout with pitch pinned so that 10'x10' fits ring_n=6 (85 COBs).
    - rect_rect: axis-aligned UV rectangular rings (concentric rectangles).
    """
    L_m = _env_float("LENGTH_M", _env_float("LENGTH_FT", 12.0) * 0.3048)
    W_m = _env_float("WIDTH_M", _env_float("WIDTH_FT", 12.0) * 0.3048)
    L_m, W_m, swapped = _maybe_swap_dims(L_m, W_m)

    margin_m = _env_float("MARGIN_IN", 0.0) * 0.0254
    z_emit = _env_float("MOUNT_Z_M", 0.6096)  # 24"
    mode = _layout_mode()
    requested_mode = mode
    aspect = max(L_m, W_m) / max(min(L_m, W_m), EPS)
    if mode == "square" and aspect >= 1.2 and _env_bool("COB_RECT_AUTO", True):
        mode = "rect_rect"
    wall_clear_m = _cob_wall_clearance_m()
    min_center_guard_m = max(float(COB_LES_R_M), float(wall_clear_m))

    # Pitch pinned to a 10'x10' baseline so a 10'x10' square gets ring_n=6 → 85 COBs.
    base_short_m = 10.0 * 0.3048
    base_ring_n = _env_int("COB_BASE_RING_N", 6)
    base_usable_half = (0.5 * base_short_m) - margin_m - min_center_guard_m - CLAMP_SAFE
    if base_usable_half <= 0:
        raise SystemExit("ERROR: invalid baseline usable half-span for COB pitch.")
    pitch_axis = base_usable_half / max(base_ring_n, 1)

    # Wall/fixture guard (optional): default preserves legacy behavior (LES-only guard).
    # If you want to keep the full "fixture footprint" inside the room, set COB_EDGE_GUARD_FRAC=0.5
    # (square) or COB_EDGE_GUARD_M explicitly.
    edge_guard_m = _env_float("COB_EDGE_GUARD_M", 0.0)
    if edge_guard_m <= 0:
        edge_guard_m = _env_float("COB_EDGE_GUARD_FRAC", 0.0) * pitch_axis

    # Keep COB centers away from walls (heatsink clearance) + optional extra guard + clamp.
    total_guard_m = float(min_center_guard_m) + float(edge_guard_m)
    usable_half_x = (0.5 * L_m) - margin_m - total_guard_m - CLAMP_SAFE
    usable_half_y = (0.5 * W_m) - margin_m - total_guard_m - CLAMP_SAFE
    if usable_half_x <= 0 or usable_half_y <= 0:
        raise SystemExit("ERROR: negative/zero usable half-span; check margin/room.")

    ring_n_override = _env_int("COB_RING_N", -1)

    if mode == "square":
        half_min = min(usable_half_x, usable_half_y)
        ring_n = int(ring_n_override) if ring_n_override >= 1 else max(1, int(math.floor(half_min / max(pitch_axis, EPS))))
        rings_local = ring_n + 1
        spacing_diag = pitch_axis * SQRT2

        fixtures: List[Dict[str, float]] = []
        for L in range(rings_local):
            for i, j in _ring_ij(L):
                x, y = _ij_to_xy(i, j, spacing_diag)
                fixtures.append(
                    {
                        "ring": float(L),
                        "i": float(i),
                        "j": float(j),
                        "x": float(round(x, 6)),
                        "y": float(round(y, 6)),
                        "z": float(z_emit),
                    }
                )

        fixture_w = pitch_axis
        fixture_h = pitch_axis
        xs = [p["x"] for p in fixtures] if fixtures else [0.0]
        ys = [p["y"] for p in fixtures] if fixtures else [0.0]
        meta = {
            "layout_mode": "square",
            "layout_mode_requested": str(requested_mode),
            "layout_mode_auto": bool(mode != requested_mode),
            "room_L_m": float(L_m),
            "room_W_m": float(W_m),
            "margin_m": float(margin_m),
            "wall_clearance_m": float(wall_clear_m),
            "edge_guard_extra_m": float(edge_guard_m),
            "z_emit_m": float(z_emit),
            "rings": int(rings_local),
            "ring_n": int(ring_n),
            "pitch_axis_m": float(pitch_axis),
            "pitch_x_m": float(pitch_axis),
            "pitch_y_m": float(pitch_axis),
            "fixture_w_m": float(fixture_w),
            "fixture_h_m": float(fixture_h),
            "swapped_axes": bool(swapped),
            "x_span_m": float(max(xs) - min(xs)),
            "y_span_m": float(max(ys) - min(ys)),
        }
        return fixtures, pitch_axis, meta

    if mode == "rect_rect":
        # Rectangular UV rings on the diamond-staggered lattice, pitch pinned to baseline.
        ring_n = int(ring_n_override) if ring_n_override >= 1 else max(1, int(math.floor(usable_half_y / max(pitch_axis, EPS))))
        u_max = max(1, int(math.floor(usable_half_x / max(pitch_axis, EPS))))
        offset = max(0, int(u_max))
        if offset % 2 != 0:
            offset = max(0, offset - 1)

        uv_points = _generate_rect_uv_staggered(ring_n=ring_n, offset=offset)
        fixtures = []
        for u, v, ring in uv_points:
            fixtures.append(
                {
                    "ring": float(ring),
                    "i": float(u),
                    "j": float(v),
                    "x": float(round(u * pitch_axis, 6)),
                    "y": float(round(v * pitch_axis, 6)),
                    "z": float(z_emit),
                }
            )

        fixture_w = pitch_axis
        fixture_h = pitch_axis
        xs = [p["x"] for p in fixtures] if fixtures else [0.0]
        ys = [p["y"] for p in fixtures] if fixtures else [0.0]
        meta = {
            "layout_mode": "rect_rect",
            "layout_mode_requested": str(requested_mode),
            "layout_mode_auto": bool(mode != requested_mode),
            "room_L_m": float(L_m),
            "room_W_m": float(W_m),
            "margin_m": float(margin_m),
            "wall_clearance_m": float(wall_clear_m),
            "edge_guard_extra_m": float(edge_guard_m),
            "z_emit_m": float(z_emit),
            "rings": int(ring_n + 1),
            "ring_n": int(ring_n),
            "pitch_axis_m": float(pitch_axis),
            "pitch_x_m": float(pitch_axis),
            "pitch_y_m": float(pitch_axis),
            "fixture_w_m": float(fixture_w),
            "fixture_h_m": float(fixture_h),
            "offset": int(offset),
            "u_max": int(u_max),
            "v_max": int(ring_n),
            "swapped_axes": bool(swapped),
            "x_span_m": float(max(xs) - min(xs)),
            "y_span_m": float(max(ys) - min(ys)),
        }
        return fixtures, pitch_axis, meta

    raise SystemExit(f"ERROR: unknown LAYOUT_MODE={mode!r} (expected 'square' or 'rect_rect')")


# ── Watts → photons → radiance (Lambertian) ─────────────────────────────────
def _eff_scale() -> float:
    # Mirror the existing convention: fixture-level PPE defaults to system-level, and EFF_SCALE is the user dimmer.
    return _env_float("EFF_SCALE", 1.0)


def _cob_ppe() -> float:
    return _env_float("COB_PPE_UMOL_PER_J", COB_PPE_UMOL_PER_J)


def _strip_ppe() -> float:
    return _env_float("STRIP_PPE_UMOL_PER_J", STRIP_PPE_UMOL_PER_J)


def _strip_w_per_m() -> float:
    return _env_float("STRIP_W_PER_M", STRIP_W_PER_M)

def _cob_strip_w_fraction() -> float:
    return _env_float("COB_STRIP_W_FRACTION", COB_STRIP_W_FRACTION_DEFAULT)

def _cob_wall_clearance_m() -> float:
    m = _env_float("COB_WALL_CLEARANCE_M", 0.0)
    if m > 0:
        return float(m)
    inches = _env_float("COB_WALL_CLEARANCE_IN", COB_WALL_CLEARANCE_IN_DEFAULT)
    return float(inches) * 0.0254


def _heatsink_radius_m() -> float:
    diam_in = _env_float("COB_HEATSINK_DIAM_IN", COB_HEATSINK_DIAM_IN_DEFAULT)
    return 0.5 * float(diam_in) * 0.0254


def _cob_halo_bar_m() -> float:
    m = _env_float("COB_HALO_BAR_M", 0.0)
    if m > 0:
        return float(m)
    inches = _env_float("COB_HALO_BAR_IN", COB_HALO_BAR_IN_DEFAULT)
    return float(inches) * 0.0254


def _frame_leg_m() -> float:
    leg_in = _env_float("COB_FRAME_ANGLE_IN", FRAME_ANGLE_LEG_IN_DEFAULT)
    return float(leg_in) * 0.0254


def _cob_area_m2() -> float:
    return PI * COB_LES_R_M * COB_LES_R_M


def _radiance_from_ppf(ppf_umol_s: float, area_m2: float) -> float:
    return 0.0 if area_m2 <= 0 else float(ppf_umol_s) / (float(area_m2) * PI)


# ── Optics helpers (brightfunc beam shaping) ─────────────────────────────────
def _m_from_fwhm(deg: float) -> float:
    th = math.radians(deg * 0.5)
    c = max(math.cos(th), 1e-6)
    return math.log(0.5) / math.log(c)


def _avg_sym_cos(m: float) -> float:
    # 1/π ∫ c^m * c dΩ = 2/(m+2)
    return 2.0 / (m + 2.0)


def _write_beam_cal_base(path: Path) -> None:
    txt = r"""{ beam.cal v1.6 - symmetric, batwing, elliptical (no in-CAL rotation) }
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


_WRAPPERS: Dict[str, str] = {}


def _add_wrapper(name: str, expr: str) -> None:
    if name not in _WRAPPERS:
        _WRAPPERS[name] = f"{name} = {expr};"


def _flush_wrappers(cal_file: Path) -> None:
    if not _WRAPPERS:
        return
    with cal_file.open("a") as fh:
        fh.write("\n{ -- auto-generated lens wrappers -- }\n")
        for _, line in sorted(_WRAPPERS.items()):
            fh.write(line + "\n")


def _write_brightfunc_ref(fh, patt_name: str, funcname: str) -> None:
    fh.write(f"void brightfunc {patt_name}\n")
    fh.write(f"2 {funcname} {CAL_NAME}\n")
    fh.write("0\n")
    fh.write("0\n\n")


# ── Geometry writers ─────────────────────────────────────────────────────────
def _write_rect_polygon(fh, mat: str, name: str, cx: float, cy: float, z: float, lx: float, ly: float):
    hx = 0.5 * lx
    hy = 0.5 * ly
    fh.write(f"{mat} polygon {name}\n0\n0\n12\n")
    fh.write(f"  {cx - hx:.6f} {cy + hy:.6f} {z:.6f}\n")
    fh.write(f"  {cx + hx:.6f} {cy + hy:.6f} {z:.6f}\n")
    fh.write(f"  {cx + hx:.6f} {cy - hy:.6f} {z:.6f}\n")
    fh.write(f"  {cx - hx:.6f} {cy - hy:.6f} {z:.6f}\n\n")


def _write_quad_polygon(fh, mat: str, name: str, corners_xy: List[Tuple[float, float]], z: float):
    if len(corners_xy) < 4:
        return
    fh.write(f"{mat} polygon {name}\n0\n0\n12\n")
    for x, y in corners_xy[:4]:
        fh.write(f"  {float(x):.6f} {float(y):.6f} {float(z):.6f}\n")
    fh.write("\n")


def _write_disk_polygon(fh, mat: str, name: str, cx: float, cy: float, z: float, r: float, n: int):
    n = max(3, int(n))
    fh.write(f"{mat} polygon {name}\n0\n0\n{3*n}\n")
    for k in range(n):
        th = -2.0 * PI * (k / n)  # clockwise when viewed from +Z → normal points down
        x = cx + r * math.cos(th)
        y = cy + r * math.sin(th)
        fh.write(f"  {x:.6f} {y:.6f} {z:.6f}\n")
    fh.write("\n")


# ── Perimeter strips (per-ring) ──────────────────────────────────────────────
def _tile_strip_segments_along_edge(edge_len: float, seg_len: float) -> List[float]:
    """
    Return center offsets along a 1D edge of length edge_len, using seg_len tiles.
    """
    if edge_len <= 0 or seg_len <= 0:
        return []
    if seg_len > edge_len:
        return [0.0]
    n = max(1, int(edge_len // seg_len))
    used = n * seg_len
    margin = 0.5 * max(0.0, edge_len - used)
    centers = []
    start = -0.5 * edge_len + margin + 0.5 * seg_len
    for k in range(n):
        centers.append(start + k * seg_len)
    return centers


def _choose_strip_len(edge_len: float) -> float:
    return STRIP_LONG_L_M if edge_len >= 0.85 else STRIP_SHORT_L_M


def _build_ring_perimeter_strips(fixtures: List[Dict[str, float]], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not fixtures:
        return []
    fx_w = float(meta.get("fixture_w_m", 0.0))
    fx_h = float(meta.get("fixture_h_m", 0.0))
    if fx_w <= 0 or fx_h <= 0:
        return []

    z = float(meta.get("z_emit_m", fixtures[0].get("z", 0.6096)))
    # Frame-mounted strips: place strips on a fixed frame offset from COB centers (heatsink + angle),
    # with the strip's outer edge aligned to the frame line.
    frame_offset_m = _heatsink_radius_m() + _frame_leg_m()
    strip_inset_m = 0.5 * STRIP_W_M
    room_half_x = 0.5 * float(meta.get("room_L_m", 0.0)) - float(meta.get("margin_m", 0.0))
    room_half_y = 0.5 * float(meta.get("room_W_m", 0.0)) - float(meta.get("margin_m", 0.0))
    clamp_top_y = room_half_y - 0.5 * STRIP_W_M - CLAMP_SAFE
    clamp_bot_y = -clamp_top_y
    clamp_right_x = room_half_x - 0.5 * STRIP_W_M - CLAMP_SAFE
    clamp_left_x = -clamp_right_x

    strips: List[Dict[str, Any]] = []

    # Which rings should get perimeter strips?
    strip_mode = os.getenv("COB_STRIP_MODE", "proportional").strip().lower()
    # Never place strips around the center singleton (ring 0).
    min_ring = max(0, _env_int("COB_STRIP_MIN_RING", 0))
    max_ring_override = _env_int("COB_STRIP_MAX_RING", -1)

    by_ring: Dict[int, List[Dict[str, float]]] = {}
    ring_max = 0
    for p in fixtures:
        ring = int(float(p.get("ring", 0)))
        ring_max = max(ring_max, ring)
        by_ring.setdefault(ring, []).append(p)

    if strip_mode == "off":
        meta["strip_mode"] = "off"
        meta["strip_clearance_m"] = float(STRIP_CLEARANCE_M)
        return []

    if strip_mode == "outer_constant":
        rings = [ring_max]
        meta["strip_mode"] = "outer_constant"
    else:
        rings = list(range(max(min_ring, 0), ring_max + 1))
        meta["strip_mode"] = strip_mode or "proportional"

    if max_ring_override >= 0:
        rings = [r for r in rings if r <= max_ring_override]

    boundary_by_ring: Dict[int, Dict[str, float]] = {}
    prev_bx = 0.0
    prev_by = 0.0
    tol = 1e-6
    for ring in rings:
        pts = by_ring.get(ring, [])
        if not pts:
            continue
        max_x = max(abs(float(p["x"])) for p in pts)
        max_y = max(abs(float(p["y"])) for p in pts)
        # Default: outer frame around the ring (heatsink + angle).
        boundary_x = max_x + frame_offset_m
        boundary_y = max_y + frame_offset_m

        # Special-case the first strip ring in the square (diamond) layout:
        # we want 4 strips between the 4 COBs in ring 1 (no stacked center strips).
        if str(meta.get("layout_mode", "")).strip().lower() == "square" and ring == 1 and len(pts) == 4:
            boundary_x = max(0.0, max_x - frame_offset_m)
            boundary_y = max(0.0, max_y - frame_offset_m)
        boundary_by_ring[ring] = {"x": float(boundary_x), "y": float(boundary_y)}

        # Only place strips on edges that are truly "outer" for this ring.
        # For rectangular concentric layouts, some rings expand only on one axis; the other axis edges would
        # coincide with an inner ring and are considered interior edges (skip them to avoid double-stacking).
        add_lr = True
        add_tb = True
        if strip_mode not in ("outer_constant",):
            add_lr = boundary_x > (prev_bx + tol)
            add_tb = boundary_y > (prev_by + tol)
        prev_bx = max(prev_bx, boundary_x)
        prev_by = max(prev_by, boundary_y)

        # Horizontal edges (top/bottom)
        if add_tb:
            edge_len_x = 2.0 * boundary_x
            seg_len_x = _choose_strip_len(edge_len_x)
            x_centers = _tile_strip_segments_along_edge(edge_len_x, seg_len_x)
            for side, y0 in (
                ("top", min(+boundary_y - strip_inset_m, clamp_top_y)),
                ("bottom", max(-boundary_y + strip_inset_m, clamp_bot_y)),
            ):
                for k, xc in enumerate(x_centers):
                    strips.append(
                        {
                            "ring": int(ring),
                            "side": side,
                            "cx": float(xc),
                            "cy": float(y0),
                            "cz": float(z),
                            "length_m": float(seg_len_x),
                            "width_m": float(STRIP_W_M),
                            "lx_m": float(seg_len_x),
                            "ly_m": float(STRIP_W_M),
                        }
                    )

        # Vertical edges (left/right) — long axis along Y
        if add_lr:
            edge_len_y = 2.0 * boundary_y
            seg_len_y = _choose_strip_len(edge_len_y)
            y_centers = _tile_strip_segments_along_edge(edge_len_y, seg_len_y)
            for side, x0 in (
                ("right", min(+boundary_x - strip_inset_m, clamp_right_x)),
                ("left", max(-boundary_x + strip_inset_m, clamp_left_x)),
            ):
                for k, yc in enumerate(y_centers):
                    strips.append(
                        {
                            "ring": int(ring),
                            "side": side,
                            "cx": float(x0),
                            "cy": float(yc),
                            "cz": float(z),
                            "length_m": float(seg_len_y),
                            "width_m": float(STRIP_W_M),
                            "lx_m": float(STRIP_W_M),
                            "ly_m": float(seg_len_y),
                        }
                    )

    # Backwards-friendly convenience: expose outermost boundary too.
    if boundary_by_ring:
        last = max(boundary_by_ring.keys())
        meta["perimeter_boundary_x_m"] = float(boundary_by_ring[last]["x"])
        meta["perimeter_boundary_y_m"] = float(boundary_by_ring[last]["y"])
    meta["perimeter_boundaries_by_ring_m"] = boundary_by_ring
    meta["strip_clearance_m"] = float(STRIP_CLEARANCE_M)
    return strips


# ── COB halo segments (per COB) ──────────────────────────────────────────────
def _build_cob_halo_segments(fixtures: List[Dict[str, float]], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not fixtures:
        return []
    z = float(meta.get("z_emit_m", fixtures[0].get("z", 0.6096)))
    bar_m = _cob_halo_bar_m()
    r = _heatsink_radius_m()
    span = 2.0 * (r + bar_m)
    offset = r + 0.5 * bar_m
    meta["halo_bar_m"] = float(bar_m)
    meta["halo_span_m"] = float(span)
    meta["halo_offset_m"] = float(offset)

    strips: List[Dict[str, Any]] = []
    min_ring = max(0, _env_int("COB_STRIP_MIN_RING", 0))
    max_ring_override = _env_int("COB_STRIP_MAX_RING", -1)

    for f in fixtures:
        ring = int(float(f.get("ring", 0)))
        if ring < min_ring:
            continue
        if max_ring_override >= 0 and ring > max_ring_override:
            continue
        cx = float(f.get("x", 0.0))
        cy = float(f.get("y", 0.0))

        strips.append(
            {
                "ring": ring,
                "side": "top",
                "cx": cx,
                "cy": cy + offset,
                "cz": z,
                "length_m": span,
                "width_m": bar_m,
                "lx_m": span,
                "ly_m": bar_m,
            }
        )
        strips.append(
            {
                "ring": ring,
                "side": "bottom",
                "cx": cx,
                "cy": cy - offset,
                "cz": z,
                "length_m": span,
                "width_m": bar_m,
                "lx_m": span,
                "ly_m": bar_m,
            }
        )
        strips.append(
            {
                "ring": ring,
                "side": "right",
                "cx": cx + offset,
                "cy": cy,
                "cz": z,
                "length_m": span,
                "width_m": bar_m,
                "lx_m": bar_m,
                "ly_m": span,
            }
        )
        strips.append(
            {
                "ring": ring,
                "side": "left",
                "cx": cx - offset,
                "cy": cy,
                "cz": z,
                "length_m": span,
                "width_m": bar_m,
                "lx_m": bar_m,
                "ly_m": span,
            }
        )
    return strips


# ── Writers ──────────────────────────────────────────────────────────────────
def _write_strip_overlay_rad(strips: List[Dict[str, Any]], *, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        fh.write("# Halo overlay geometry (non-emitting)\n\n")
        fh.write("void plastic strip_overlay_red\n0\n0\n5 1.0 0.0 0.0 0.0 0.05\n\n")
        for idx, s in enumerate(strips):
            name = f"strip_overlay_{idx:04d}"
            if "corners_xy" in s:
                _write_quad_polygon(
                    fh,
                    "strip_overlay_red",
                    name,
                    s["corners_xy"],
                    float(s["cz"]),
                )
            else:
                _write_rect_polygon(
                    fh,
                    "strip_overlay_red",
                    name,
                    float(s["cx"]),
                    float(s["cy"]),
                    float(s["cz"]),
                    float(s["lx_m"]),
                    float(s["ly_m"]),
                )


def _write_cob_layout_json(
    fixtures: List[Dict[str, float]],
    *,
    meta: Dict[str, Any],
    cob_ring_total_w_in: Dict[int, float],
    strip_ring_total_w_in: Dict[int, float],
    strips: List[Dict[str, Any]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    eff = _eff_scale()
    cob_ppe = _cob_ppe()
    strip_ppe = _strip_ppe()

    fixture_w = float(meta.get("fixture_w_m", 0.0))
    fixture_h = float(meta.get("fixture_h_m", 0.0))

    ring_counts = Counter(int(float(f.get("ring", 0))) for f in fixtures)
    strip_len_by_ring: Dict[int, float] = {}
    for s in strips:
        L = int(s.get("ring", 0))
        strip_len_by_ring[L] = strip_len_by_ring.get(L, 0.0) + float(s.get("length_m", 0.0))

    fixtures_out: List[Dict[str, Any]] = []
    cobs_out: List[Dict[str, Any]] = []

    for idx, f in enumerate(fixtures):
        ring = int(float(f.get("ring", 0)))
        cx = float(f.get("x", 0.0))
        cy = float(f.get("y", 0.0))
        cz = float(f.get("z", meta.get("z_emit_m", 0.6096)))

        ring_total_w_in = float(cob_ring_total_w_in.get(ring, 0.0))
        nL = max(0, int(ring_counts.get(ring, 0)))
        per_cob_w_in = (ring_total_w_in / nL) if nL > 0 else 0.0
        per_cob_w_eff = per_cob_w_in * eff
        ppf = per_cob_w_eff * cob_ppe

        fixtures_out.append(
            {
                "id": idx,
                "ring": ring,
                "center": [round(cx, 6), round(cy, 6), round(cz, 6)],
                "size": [round(fixture_w, 6), round(fixture_h, 6)],
            }
        )
        cobs_out.append(
            {
                "id": idx,
                "ring": ring,
                "center": [round(cx, 6), round(cy, 6), round(cz, 6)],
                "les_diameter_m": float(COB_LES_DIAM_M),
                "ring_total_watts_in": round(ring_total_w_in, 6),
                "input_watts": round(per_cob_w_in, 6),
                "effective_watts": round(per_cob_w_eff, 6),
                "ppf_umol_s": round(ppf, 6),
            }
        )

    strips_out: List[Dict[str, Any]] = []
    for idx, s in enumerate(strips):
        lx = float(s["lx_m"])
        ly = float(s["ly_m"])
        ring = int(s.get("ring", 0))
        length = float(s["length_m"])
        ring_total_w_in = float(strip_ring_total_w_in.get(ring, 0.0))
        ring_len = float(strip_len_by_ring.get(ring, 0.0))
        seg_w_in = float((ring_total_w_in * (length / ring_len)) if ring_len > 0 else 0.0)
        seg_w_eff = seg_w_in * eff
        ppf = seg_w_eff * strip_ppe
        cx = float(s["cx"])
        cy = float(s["cy"])
        corners_src = s.get("corners_xy")
        if isinstance(corners_src, list) and len(corners_src) >= 4:
            corners = [[round(float(p[0]), 6), round(float(p[1]), 6)] for p in corners_src[:4]]
        else:
            hx = 0.5 * lx
            hy = 0.5 * ly
            corners = [
                [round(cx - hx, 6), round(cy + hy, 6)],
                [round(cx + hx, 6), round(cy + hy, 6)],
                [round(cx + hx, 6), round(cy - hy, 6)],
                [round(cx - hx, 6), round(cy - hy, 6)],
            ]
        strips_out.append(
            {
                "id": idx,
                "ring": ring,
                "side": str(s["side"]),
                "center": [round(cx, 6), round(cy, 6), round(float(s["cz"]), 6)],
                "length_m": round(length, 6),
                "width_m": round(float(s["width_m"]), 6),
                "ring_total_watts_in": round(ring_total_w_in, 6),
                "input_watts": round(seg_w_in, 6),
                "effective_watts": round(seg_w_eff, 6),
                "ppf_umol_s": round(ppf, 6),
                "corners_xy": corners,
            }
        )

    payload = {
        "version": 1,
        "generator": "generate_emitters_cob.py",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "units": "meters",
        "room": {"L": round(float(meta.get("room_L_m", 0.0)), 6), "W": round(float(meta.get("room_W_m", 0.0)), 6)},
        "margin_m": round(float(meta.get("margin_m", 0.0)), 6),
        "z_emit_m": round(float(meta.get("z_emit_m", 0.0)), 6),
        "layout_mode": str(meta.get("layout_mode", "")),
        "ring_n": int(meta.get("ring_n", 0)),
        "rings": int(meta.get("rings", 1)),
        "pitch_axis_m": round(float(meta.get("pitch_axis_m", 0.0)), 6),
        "eff_scale": round(float(eff), 6),
        "cob": {
            "les_diameter_m": float(COB_LES_DIAM_M),
            "ppe_umol_per_j": float(cob_ppe),
            "vf_typ_v": float(COB_VF_TYP_V),
            "if_typ_a": float(COB_IF_TYP_A),
            "if_max_a": float(COB_IF_MAX_A),
            "lm_per_w_ref": float(COB_LM_PER_W_REF),
        },
        "strips": {
            "w_m": float(STRIP_W_M),
            "long_l_m": float(STRIP_LONG_L_M),
            "short_l_m": float(STRIP_SHORT_L_M),
            "ppe_umol_per_j": float(strip_ppe),
            "elec_caps_w": {"long": float(STRIP_LONG_ELEC_MAX_W), "short": float(STRIP_SHORT_ELEC_MAX_W)},
            "geometry": str(meta.get("strip_geometry", "")),
            "halo_bar_m": float(meta.get("halo_bar_m", 0.0)),
            "halo_span_m": float(meta.get("halo_span_m", 0.0)),
        },
        "fixtures": fixtures_out,
        "cobs": cobs_out,
        "strip_segments": strips_out,
        "ring_channels": {
            "cob_ring_total_watts_in": {str(k): float(v) for k, v in sorted(cob_ring_total_w_in.items())},
            "strip_ring_total_watts_in": {str(k): float(v) for k, v in sorted(strip_ring_total_w_in.items())},
            "cob_counts_by_ring": {str(k): int(v) for k, v in sorted(ring_counts.items())},
            "strip_length_m_by_ring": {str(k): float(v) for k, v in sorted(strip_len_by_ring.items())},
        },
        "meta": {k: v for k, v in meta.items() if k not in ("room_L_m", "room_W_m", "margin_m", "z_emit_m")},
    }
    out_path.write_text(json.dumps(payload, indent=2))


# ── Public generator API (for GUI/other scripts) ─────────────────────────────
def generate_emitters_cob(
    *,
    fixtures: List[Dict[str, float]] | None = None,
    layout_meta: Dict[str, Any] | None = None,
    cob_ring_total_w_in: Dict[int, float] | None = None,
    strip_ring_total_w_in: Dict[int, float] | None = None,
    z_emit_m: float | None = None,
    out_emitters: Path | None = None,
    out_layout_json: Path | None = None,
    out_strip_overlay: Path | None = None,
) -> Dict[str, Any]:
    """
    Build COB + halo emitters.

    COBs are driven by per-ring watts-per-COB (solver output). Halos are proportional to
    each ring's COB total watts unless COB_STRIP_MODE=off/outer_constant/explicit.

    Returns a dict with summary totals for convenience.
    """
    out_emitters = out_emitters or Path("ies_sources/emitters_cob_ALL_umol.rad")
    out_layout_json = out_layout_json or Path("ies_sources/cob_layout.json")
    out_strip_overlay = out_strip_overlay or Path("overlays/strip_overlay.rad")

    out_emitters.parent.mkdir(parents=True, exist_ok=True)

    if fixtures is None:
        fixtures, pitch_axis, meta = _compute_positions_from_env()
    else:
        meta = dict(layout_meta or {})
        pitch_axis = float(meta.get("pitch_axis_m", meta.get("pitch_axis", 0.0)) or 0.0)
        # Infer rings from fixture ring indices if not provided.
        if "rings" not in meta:
            try:
                ring_max = max(int(float(p.get("ring", 0))) for p in fixtures) if fixtures else 0
            except Exception:
                ring_max = 0
            meta["rings"] = int(ring_max + 1)
        meta.setdefault("layout_mode", "custom")
        meta.setdefault("z_emit_m", float(z_emit_m) if z_emit_m is not None else float(meta.get("z_emit_m", 0.6096)))

    if z_emit_m is not None:
        meta["z_emit_m"] = float(z_emit_m)
        for p in fixtures:
            p["z"] = float(z_emit_m)

    rings_local = int(meta.get("rings", int(meta.get("ring_n", 0)) + 1))
    ring_max = max((int(float(p.get("ring", 0))) for p in fixtures), default=0)

    optics = _optics_settings(ring_max)
    optics_rings = {int(r) for r in optics.get("rings", [])}
    use_optics = bool(optics_rings)
    meta["cob_optics_mode"] = str(optics.get("mode", "off"))
    meta["cob_optics_kind"] = str(optics.get("kind", "sym"))
    meta["cob_optics_fwhm_deg"] = float(optics.get("fwhm_deg", 0.0))
    meta["cob_optics_rings"] = sorted(optics_rings)

    strip_geom = os.getenv("COB_STRIP_GEOM", "halo").strip().lower()
    if strip_geom not in ("halo", "perimeter"):
        strip_geom = "halo"

    # Halo/perimeter geometry first, so we can compute per-ring supplemental length.
    if strip_geom == "perimeter":
        strips = _build_ring_perimeter_strips(fixtures, meta)
    else:
        strips = _build_cob_halo_segments(fixtures, meta)
    meta["strip_geometry"] = strip_geom
    meta.setdefault("strip_clearance_m", float(STRIP_CLEARANCE_M))

    ring_counts = Counter(int(float(p.get("ring", 0))) for p in fixtures)
    strip_len_by_ring: Dict[int, float] = {}
    strip_area_by_ring: Dict[int, float] = {}
    for s in strips:
        L = int(s.get("ring", 0))
        seg_len = float(s.get("length_m", 0.0))
        seg_w = float(s.get("width_m", 0.0))
        strip_len_by_ring[L] = strip_len_by_ring.get(L, 0.0) + seg_len
        strip_area_by_ring[L] = strip_area_by_ring.get(L, 0.0) + (seg_len * seg_w)

    # Defaults (no solver): COB baseline is set as W/COB.
    source = "defaults"
    baseline_w_per_cob = _env_float("COB_W_PER_COB", 60.0)
    cob_w_per_cob_by_ring = {L: float(baseline_w_per_cob) for L in range(rings_local)}

    strip_mode = str(meta.get("strip_mode", os.getenv("COB_STRIP_MODE", "proportional"))).strip().lower()
    meta["strip_mode"] = strip_mode
    if strip_mode == "outer_constant":
        # Legacy: constant W/m on the outermost ring perimeter only.
        strip_wpm = _strip_w_per_m()
        outer = max((int(float(p.get("ring", 0))) for p in fixtures), default=0)
        strip_ring_total_w = {L: 0.0 for L in range(rings_local)}
        strip_ring_total_w[outer] = float(strip_wpm) * float(strip_len_by_ring.get(outer, 0.0))
        source = "outer_constant"
    elif strip_mode == "off":
        strip_ring_total_w = {L: 0.0 for L in range(rings_local)}
    else:
        # Default: proportional supplemental strips (computed after COB ring watts are known).
        strip_ring_total_w = {L: 0.0 for L in range(rings_local)}

    # Explicit caller overrides (treated as total ring watts, not per-emitter).
    if cob_ring_total_w_in is not None:
        # Back-compat: if caller provides total ring watts, convert to W/COB using counts.
        for k, v in cob_ring_total_w_in.items():
            L = int(k)
            nL = max(0, int(ring_counts.get(L, 0)))
            cob_w_per_cob_by_ring[L] = (float(v) / nL) if nL > 0 else 0.0
        source = "provided"
    if strip_ring_total_w_in is not None:
        # Only honored if user explicitly opts-in; otherwise strips are derived from COB watts.
        if strip_mode == "explicit":
            strip_ring_total_w = {int(k): float(v) for k, v in strip_ring_total_w_in.items()}
            source = "provided"

    # Basis mode + JSON overrides (COB ring values are watts-per-COB).
    basis_mode = os.getenv("COB_BASIS_MODE", "0") == "1"
    basis_ring = int(os.getenv("COB_BASIS_RING", "-1"))
    basis_unit_w_per_cob = float(os.getenv("COB_BASIS_UNIT_W_PER_COB", os.getenv("COB_BASIS_UNIT_W", "1.0")))

    if basis_mode and basis_ring >= 0:
        for L in range(rings_local):
            cob_w_per_cob_by_ring[L] = 0.0
        cob_w_per_cob_by_ring[basis_ring] = float(basis_unit_w_per_cob)
        source = f"basis_mode_cob_ring_{basis_ring}"
    elif source != "provided":
        use_json = os.getenv("USE_RING_POWERS_JSON", "1") != "0"
        path = Path(os.getenv("RING_POWERS_JSON", "ring_powers_cob.json"))
        if use_json and path.exists():
            try:
                data = json.loads(path.read_text())

                # Back-compat: solve_uniformity schema (1 vector, indices optional) → treated as W/COB.
                arr = data.get("ring_powers_W_per_module") or data.get("ring_powers")
                idxs = data.get("ring_indices")
                if arr is not None:
                    arr = list(arr)
                    if idxs is None:
                        idxs = list(range(len(arr)))
                    if len(idxs) == len(arr):
                        for k, v in zip(idxs, arr):
                            try:
                                kk = int(k)
                                vv = float(v)
                            except Exception:
                                continue
                            if 0 <= kk < rings_local:
                                cob_w_per_cob_by_ring[kk] = vv
                        source = f"json:{path}"
                        print(f"Applied ring powers from {path}")
            except Exception as e:
                print(f"WARNING: failed to load ring powers JSON ({path}): {e}")

    eff = _eff_scale()
    cob_ppe = _cob_ppe()
    strip_ppe = _strip_ppe()

    # Convert W/COB to total ring watts.
    cob_ring_total_w = {L: float(cob_w_per_cob_by_ring.get(L, 0.0)) * float(ring_counts.get(L, 0)) for L in range(rings_local)}

    # Halos: derived from COB ring watts unless explicitly overridden.
    if strip_mode not in ("off", "outer_constant", "explicit"):
        frac = max(0.0, float(_cob_strip_w_fraction()))
        for L in range(rings_local):
            strip_ring_total_w[L] = float(cob_ring_total_w.get(L, 0.0)) * frac
        meta["strip_mode"] = "proportional"
        meta["strip_w_fraction_of_cob"] = float(frac)

    for L in range(rings_local):
        if float(strip_len_by_ring.get(L, 0.0)) <= 0.0:
            strip_ring_total_w[L] = 0.0

    cob_area = _cob_area_m2()
    cob_rad_by_ring: Dict[int, float] = {}
    for L in range(rings_local):
        nL = max(0, int(ring_counts.get(L, 0)))
        per_cob_w = (float(cob_ring_total_w.get(L, 0.0)) / nL) if nL > 0 else 0.0
        cob_rad_by_ring[L] = _radiance_from_ppf((per_cob_w * eff) * cob_ppe, cob_area)

    strip_rad_by_ring: Dict[int, float] = {}
    for L in range(rings_local):
        total_w = float(strip_ring_total_w.get(L, 0.0))
        total_area = float(strip_area_by_ring.get(L, 0.0))
        # Lambertian rectangle, uniform radiance across ring: PPF_total / (A_total * π)
        strip_rad_by_ring[L] = _radiance_from_ppf((total_w * eff) * strip_ppe, total_area) if total_area > 0 else 0.0

    cal_file = out_emitters.parent / CAL_NAME
    if use_optics:
        _WRAPPERS.clear()
        _write_beam_cal_base(cal_file)
        optics_fwhm = float(optics.get("fwhm_deg", COB_OPTICS_FWHM_DEG_DEFAULT))
        optics_m = _m_from_fwhm(optics_fwhm)
        optics_gain = 1.0 / max(_avg_sym_cos(optics_m), EPS)
    else:
        optics_fwhm = 0.0
        optics_m = 0.0
        optics_gain = 1.0

    with out_emitters.open("w") as fh:
        fh.write("# COB + per-ring halo emitters in µmol/s/sr/m²\n")
        fh.write(f"# layout={meta.get('layout_mode')} rings={rings_local} fixtures={len(fixtures)} pitch_axis={pitch_axis:.6f} m\n")
        fh.write(f"# COB: LES_diam={COB_LES_DIAM_M:.4f} m  PPE={cob_ppe:.3f} µmol/J\n")
        fh.write(
            f"# Halos: bar={float(meta.get('halo_bar_m', 0.0)):.4f} m  PPE={strip_ppe:.3f} µmol/J  mode={meta.get('strip_mode')}\n\n"
        )
        if use_optics:
            rings_txt = ",".join(str(r) for r in sorted(optics_rings)) or "-"
            fh.write(f"# optics: {optics.get('kind')} rings={rings_txt} fwhm={optics_fwhm:.1f}\n\n")
        else:
            fh.write("# optics: none\n\n")

        # Per-ring COB materials
        optics_written: set[int] = set()
        for L in range(rings_local):
            rad = float(cob_rad_by_ring.get(L, 0.0))
            if use_optics and L in optics_rings:
                patt = f"p_sym_cob_L{L}"
                if L not in optics_written:
                    fname = f"f_sym_cob_L{L}"
                    _add_wrapper(fname, f"{optics_gain:.9f} * pwr(c, {optics_m:.9f})")
                    _write_brightfunc_ref(fh, patt, fname)
                    optics_written.add(L)
                fh.write(f"{patt} light cob_L{L}\n0\n0\n3 {rad:.6f} {rad:.6f} {rad:.6f}\n\n")
            else:
                fh.write(f"void light cob_L{L}\n0\n0\n3 {rad:.6f} {rad:.6f} {rad:.6f}\n\n")

        # Per-ring halo materials (separate solver channels)
        for L in range(rings_local):
            rad = float(strip_rad_by_ring.get(L, 0.0))
            fh.write(f"void light strip_L{L}\n0\n0\n3 {rad:.6f} {rad:.6f} {rad:.6f}\n\n")

        # COB disks
        for idx, p in enumerate(fixtures):
            L = int(float(p.get("ring", 0)))
            name = f"cob_L{L}_{idx:04d}"
            _write_disk_polygon(
                fh,
                f"cob_L{L}",
                name,
                float(p["x"]),
                float(p["y"]),
                float(p["z"]),
                float(COB_LES_R_M),
                int(_env_int("COB_DISK_SIDES", COB_DISK_SIDES)),
            )

        # Strip rectangles
        for idx, s in enumerate(strips):
            L = int(s.get("ring", 0))
            name = f"strip_L{L}_{s['side']}_{idx:04d}"
            if "corners_xy" in s:
                _write_quad_polygon(
                    fh,
                    f"strip_L{L}",
                    name,
                    s["corners_xy"],
                    float(s["cz"]),
                )
            else:
                _write_rect_polygon(
                    fh,
                    f"strip_L{L}",
                    name,
                    float(s["cx"]),
                    float(s["cy"]),
                    float(s["cz"]),
                    float(s["lx_m"]),
                    float(s["ly_m"]),
                )

    if use_optics:
        _flush_wrappers(cal_file)

    # Write overlay geometry (non-emitting) for strip placement preview.
    _write_strip_overlay_rad(strips, out_path=out_strip_overlay)

    # Snapshot JSON (fixtures, COBs, strips)
    _write_cob_layout_json(
        fixtures,
        meta=meta,
        cob_ring_total_w_in=cob_ring_total_w,
        strip_ring_total_w_in=strip_ring_total_w,
        strips=strips,
        out_path=out_layout_json,
    )

    # Summary
    total_cob_w_in = sum(float(cob_ring_total_w.get(L, 0.0)) for L in range(rings_local))
    total_cob_w_eff = total_cob_w_in * eff
    total_strip_w_in = sum(float(strip_ring_total_w.get(L, 0.0)) for L in range(rings_local))
    total_strip_w_eff = total_strip_w_in * eff

    total_cob_ppf = total_cob_w_eff * cob_ppe
    total_strip_ppf = total_strip_w_eff * strip_ppe
    total_ppf = total_cob_ppf + total_strip_ppf

    # Per-ring breakdown (input + effective + normalized convenience)
    per_ring_lines: List[str] = []
    for L in range(rings_local):
        nL = int(ring_counts.get(L, 0))
        cob_w_in_L = float(cob_ring_total_w.get(L, 0.0))
        cob_w_eff_L = cob_w_in_L * eff
        per_cob_w_in = float(cob_w_per_cob_by_ring.get(L, 0.0))
        per_cob_w_eff = per_cob_w_in * eff
        cob_ppf_L = cob_w_eff_L * cob_ppe

        strip_len_L = float(strip_len_by_ring.get(L, 0.0))
        strip_w_in_L = float(strip_ring_total_w.get(L, 0.0))
        strip_w_eff_L = strip_w_in_L * eff
        strip_wpm_in = (strip_w_in_L / strip_len_L) if strip_len_L > 0 else 0.0
        strip_wpm_eff = strip_wpm_in * eff
        strip_ppf_L = strip_w_eff_L * strip_ppe

        per_ring_lines.append(
            f"    ring {L:2d}: COB {cob_w_in_L:8.2f} W in ({cob_w_eff_L:8.2f} W eff) "
            f"→ {per_cob_w_in:6.2f} W/COB ({per_cob_w_eff:6.2f} eff) × {nL:3d} | "
            f"HALO {strip_w_in_L:8.2f} W in ({strip_w_eff_L:8.2f} W eff) "
            f"→ {strip_wpm_in:7.2f} W/m ({strip_wpm_eff:7.2f} eff) over {strip_len_L:6.2f} m | "
            f"PPF {cob_ppf_L+strip_ppf_L:8.1f} µmol/s"
        )

    optics_rings_txt = ",".join(str(r) for r in sorted(optics_rings)) if optics_rings else "off"
    optics_desc = (
        f"{optics.get('kind')} rings={optics_rings_txt} fwhm={optics_fwhm:.1f}"
        if optics_rings
        else "off"
    )

    summ = Path("ies_sources/cob_summary.txt")
    summ.write_text(
        "COB emitter summary:\n"
        + f"  layout      : {meta.get('layout_mode')} (ring_n={meta.get('ring_n')}, rings={rings_local})\n"
        + f"  fixtures    : {len(fixtures)} COBs\n"
        + f"  ring powers : {source}\n"
        + f"  derate(EFF_SCALE)={eff:.4f}\n"
        + f"  halo mode   : {meta.get('strip_mode')}\n"
        + f"  halo geom   : {meta.get('strip_geometry')}\n"
        + f"  halo frac   : {float(meta.get('strip_w_fraction_of_cob', 0.0)):.4f} (of COB ring watts)\n"
        + f"  optics      : {optics_desc}\n"
        + f"  COB PPE     : {cob_ppe:.3f} µmol/J\n"
        + f"  HALO PPE    : {strip_ppe:.3f} µmol/J\n"
        + f"  COB watts   : input ≈ {total_cob_w_in:.1f} W  effective ≈ {total_cob_w_eff:.1f} W\n"
        + f"  halo watts  : input ≈ {total_strip_w_in:.1f} W  effective ≈ {total_strip_w_eff:.1f} W\n"
        + f"  total electrical input ≈ {total_cob_w_in + total_strip_w_in:.1f} W\n"
        + f"  total effective (derated) ≈ {total_cob_w_eff + total_strip_w_eff:.1f} W\n"
        + f"  total photons ≈ {total_ppf:.0f} µmol/s\n"
        + "  per-ring (COB + HALO):\n"
        + "\n".join(per_ring_lines)
        + "\n"
    )

    print("COB emitter summary:")
    print(f"  layout      : {meta.get('layout_mode')} (ring_n={meta.get('ring_n')}, rings={rings_local})")
    print(f"  fixtures    : {len(fixtures)} COBs")
    print(f"  ring powers : {source}")
    print(f"  derate(EFF_SCALE)={eff:.4f}")
    print(f"  halo mode   : {meta.get('strip_mode')}")
    print(f"  halo geom   : {meta.get('strip_geometry')}")
    print(f"  optics      : {optics_desc}")
    print(f"  COB watts   : input ≈ {total_cob_w_in:.1f} W  effective ≈ {total_cob_w_eff:.1f} W")
    print(f"  halo watts  : input ≈ {total_strip_w_in:.1f} W  effective ≈ {total_strip_w_eff:.1f} W")
    print(f"  total photons ≈ {total_ppf:.0f} µmol/s  (COB {total_cob_ppf:.0f} + halos {total_strip_ppf:.0f})")
    if per_ring_lines:
        print("  per-ring (COB + HALO):")
        for ln in per_ring_lines:
            print(ln)
    print(f"✔ Wrote {out_emitters}")
    print(f"✔ Wrote {out_layout_json}")
    print(f"✔ Wrote {out_strip_overlay}")
    print(f"✔ Wrote {summ}")

    return {
        "rings": rings_local,
        "fixtures": len(fixtures),
        "strip_segments": len(strips),
        "cob_w_in": total_cob_w_in,
        "cob_w_eff": total_cob_w_eff,
        "strip_w_in": total_strip_w_in,
        "strip_w_eff": total_strip_w_eff,
        "cob_ppf_umol_s": total_cob_ppf,
        "strip_ppf_umol_s": total_strip_ppf,
        "total_ppf_umol_s": total_ppf,
    }


def get_module_positions() -> Tuple[List[Dict[str, float]], float]:
    fixtures, pitch_axis, _meta = _compute_positions_from_env()
    return fixtures, pitch_axis


def main() -> None:
    generate_emitters_cob()


if __name__ == "__main__":
    main()
