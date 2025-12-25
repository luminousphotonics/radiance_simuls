#!/usr/bin/env python3
# rect_layout.py
#
# Exact 203-emitter rectangular layout for SMD modules, axis-aligned:
#   n = 7, offset = 6
#   - ring 0: horizontal spine on v = 0, u = -6,-4,-2,0,2,4,6
#   - rings 1..7: rectangular perimeters in (u,v) with (u+v) even
#
# Mapping:
#   x = u * pitch_x
#   y = v * pitch_y
#
# where pitch_x and pitch_y are chosen so the full pattern just fits inside
# the usable interior of the current room (LENGTH_M x WIDTH_M), so for
# 12' x 24' you get a proper 2:1 rectangle, axis-aligned, not rotated.

from __future__ import annotations
import os
import math
from typing import List, Dict, Tuple

FT_TO_M = 0.3048


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _room_dims() -> Tuple[float, float, float]:
    """
    Resolve room dimensions from env, preferring the ft-based inputs used by
    generate_emitters_smd.py. Falls back to explicit *_M values.
    Returns (length_x_m, width_y_m, height_m).
    """
    len_ft = _float_env("LENGTH_FT", 0.0)
    wid_ft = _float_env("WIDTH_FT", 0.0)
    len_m_env = _float_env("LENGTH_M", 0.0)
    wid_m_env = _float_env("WIDTH_M", 0.0)

    length_m = len_m_env if len_m_env > 0 else len_ft * FT_TO_M
    width_m  = wid_m_env if wid_m_env > 0 else wid_ft * FT_TO_M

    if length_m <= 0:
        length_m = 3.6576  # 12 ft default
    if width_m <= 0:
        width_m = 3.6576   # square fallback

    height_m = _float_env("HEIGHT_M", 3.048)  # default 10 ft
    return length_m, width_m, height_m


def _maybe_swap_dims(length_m: float, width_m: float) -> Tuple[float, float, bool]:
    """
    Optionally swap dimensions so the long axis lies along +X for rectangular
    layouts. Controlled by ALIGN_LONG_AXIS_X (default: "1" to align).
    """
    align = os.getenv("ALIGN_LONG_AXIS_X", "1") == "1"
    if align and width_m > length_m:
        return width_m, length_m, True
    return length_m, width_m, False


# ──────────────────────────────────────────────────────────────────────────────
# 1) UV pattern helpers: spine + rectangular rings, parameterized by ring_n, offset
# ──────────────────────────────────────────────────────────────────────────────

def _generate_rect_uv(ring_n: int, offset: int) -> List[Tuple[int, int, int]]:
    """
    Generate (u, v, ring) for a rectangular layout with:
      - ring 0: horizontal spine from u=-offset..+offset (step 2), v=0
      - rings 1..ring_n: rectangular perimeters with u_max = offset + k, v_max = k
    """
    pts: List[Tuple[int, int, int]] = []

    # Central horizontal spine: ring 0
    for u in range(-offset, offset + 1, 2):
        pts.append((u, 0, 0))

    # Rectangular rings: ring = k, k = 1..ring_n
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

    dedup = {}
    for u, v, ring in pts:
        key = (u, v)
        if key not in dedup or ring < dedup[key]:
            dedup[key] = ring

    return [(u, v, ring) for (u, v), ring in dedup.items()]

# Legacy alias: generate the 233-emitter layout (ring_n=7, offset=8)
def _generate_uv_203() -> List[Tuple[int, int, int]]:
    return _generate_rect_uv(ring_n=7, offset=8)


def _generate_square_uv(ring_n: int) -> List[Tuple[int, int, int]]:
    """
    Axis-aligned diamond rings (no rotation in physical space):
      ring 0: single center (0,0)
      ring k>=1: perimeter of |u|+|v| = k (no parity thinning), 4k modules.
    """
    pts: List[Tuple[int, int, int]] = [(0, 0, 0)]
    for k in range(1, ring_n + 1):
        # traverse diamond perimeter
        for i in range(k + 1):
            pts.append((k - i, i, k))
            pts.append((-i, k - i, k))
            pts.append((-k + i, -i, k))
            pts.append((i, -k + i, k))
    dedup = {}
    for u, v, ring in pts:
        key = (u, v)
        if key not in dedup or ring < dedup[key]:
            dedup[key] = ring
    return [(u, v, ring) for (u, v), ring in dedup.items()]


# ──────────────────────────────────────────────────────────────────────────────
# 2) Map UV -> physical X,Y, axis-aligned
# ──────────────────────────────────────────────────────────────────────────────

def build_rect_grid(
    length_m: float | None = None,
    width_m: float | None = None,
    height_m: float | None = None,
    wall_margin_m: float | None = None,
    module_side_m: float | None = None,
    mount_z_m: float | None = None,
) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Called by generate_emitters_smd.get_module_positions() when LAYOUT_MODE=rect_rect.

    Steps:
      1) Generate fixed 203-emitter (u,v,ring) pattern.
      2) Compute min/max u and v over all points.
      3) Pick pitch_x and pitch_y so that:
           (max_u - min_u) * pitch_x ≈ interior X span
           (max_v - min_v) * pitch_y ≈ interior Y span
      4) Map:
           x = (u - u_center) * pitch_x
           y = (v - v_center) * pitch_y
         so pattern is centered in the room and axis-aligned.
    """

    # Room geometry in meters (X = LENGTH, Y = WIDTH to match generate_room.py)
    len_m_env, wid_m_env, h_m_env = _room_dims()
    length_m = len_m_env if length_m is None else length_m
    width_m  = wid_m_env if width_m is None else width_m
    length_m, width_m, swapped = _maybe_swap_dims(length_m, width_m)
    height_m = h_m_env  if height_m is None else height_m

    wall_margin_m = _float_env("WALL_MARGIN_M", 0.127) if wall_margin_m is None else wall_margin_m
    module_side_m = _float_env("MODULE_SIDE_M", 0.12)  if module_side_m is None else module_side_m

    fixture_leg_m = _float_env("SMD_FIXTURE_ANGLE_IN", 0.125) * 0.0254
    # Interior usable half-spans (module + fixture angle)
    half_x_int = (length_m / 2.0) - wall_margin_m - module_side_m / 2.0 - fixture_leg_m
    half_y_int = (width_m / 2.0) - wall_margin_m - module_side_m / 2.0 - fixture_leg_m
    if half_x_int <= 0 or half_y_int <= 0:
        raise RuntimeError(
            "Rect layout: non-positive interior region; "
            "check LENGTH_M, WIDTH_M, WALL_MARGIN_M, MODULE_SIDE_M."
        )

    dim_x_int = 2.0 * half_x_int
    dim_y_int = 2.0 * half_y_int

    # Reference pitch from canonical 12 ft short side (ring_n=7 → span_v=14)
    base_short_m = 3.6576
    base_half_short = (base_short_m / 2.0) - wall_margin_m - module_side_m / 2.0 - fixture_leg_m
    base_ring_n = int(_float_env("SMD_BASE_RING_N", 7) or 7)
    if base_ring_n < 1:
        base_ring_n = 1
    base_span_v = float(2 * base_ring_n)
    base_pitch = (2.0 * base_half_short) / base_span_v  # ≈0.2345 m
    base_long_m = 7.3152  # 24 ft
    base_half_long = (base_long_m / 2.0) - wall_margin_m - module_side_m / 2.0 - fixture_leg_m
    base_span_long_m = 2.0 * base_half_long
    base_offset = int(_float_env("SMD_BASE_OFFSET", base_ring_n + 1))

    aspect = dim_x_int / dim_y_int if dim_y_int > 0 else 1.0
    squareish = 0.9 <= aspect <= 1.1

    if squareish:
        # Square: diamond rings, center singleton; pitch fixed to base
        pitch = base_pitch
        pitch_x = pitch_y = pitch
        ring_n = max(1, int(round(dim_y_int / (2.0 * pitch))))
        offset = 0
        uv_points = _generate_square_uv(ring_n=ring_n)
    else:
        # Rectangular: ring_n from short axis; offset from long axis; pitch fixed to base
        pitch = base_pitch
        pitch_x = pitch_y = pitch
        ring_n = max(1, int(math.floor(dim_y_int / (2.0 * pitch))))
        required_u = math.ceil(dim_x_int / (2.0 * pitch))
        offset_scaled = int(round(base_offset * (dim_x_int / base_span_long_m)))
        offset = max(ring_n, offset_scaled, int(required_u - ring_n))
        if offset % 2 != 0:
            offset += 1
        uv_points = _generate_rect_uv(ring_n=ring_n, offset=offset)

    us_all = [u for u, _, _ in uv_points]
    vs_all = [v for _, v, _ in uv_points]
    min_u, max_u = min(us_all), max(us_all)
    min_v, max_v = min(vs_all), max(vs_all)

    # Center of UV pattern (in u,v grid)
    u_center = 0.5 * (min_u + max_u)
    v_center = 0.5 * (min_v + max_v)

    # Mounting height consistent with SMD baseline
    z_mount = _float_env("MOUNT_Z_M", height_m - 0.15875) if mount_z_m is None else mount_z_m

    positions: List[Dict] = []
    for u, v, ring in uv_points:
        x = (u - u_center) * pitch_x
        y = (v - v_center) * pitch_y
        positions.append(
            {
                "x": float(x),
                "y": float(y),
                "z": float(z_mount),
                "ring": int(ring),
            }
        )

    ring_max = max(r for _, _, r in uv_points)
    rings_count = int(ring_max + 1)

    meta = {
        "pitch_x_m": float(pitch_x),
        "pitch_y_m": float(pitch_y),
        "pitch_m": float(pitch),
        "span_u": int(max_u - min_u),
        "span_v": int(max_v - min_v),
        "footprint_x_m": float((max_u - min_u) * pitch_x),
        "footprint_y_m": float((max_v - min_v) * pitch_y),
        "z_mount_m": float(z_mount),
        "length_m": float(length_m),
        "width_m": float(width_m),
        "height_m": float(height_m),
        "wall_margin_m": float(wall_margin_m),
        "module_side_m": float(module_side_m),
        "layout_mode": "rect_rect",
        "swapped_axes": bool(swapped),
        "modules": len(positions),
        "rings_count": rings_count,
        "ring_n": ring_n,
        "offset": offset,
    }

    return positions, meta
