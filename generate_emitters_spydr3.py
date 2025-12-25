#!/usr/bin/env python3
# generate_emitters_spydr3.py • SPYDR 3 3h47 → Radiance µmol/s area emitters (Lambertian)
# - Polygons face DOWN (−Z)
# - Writes ies_sources/spydr3_layout.json for robust overlays

import json, math, os
from pathlib import Path

IN2M = 0.0254
PI   = math.pi
OUT_DIR = Path("ies_sources"); OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Helpers ---
def _maybe_swap_dims(length_m: float, width_m: float) -> tuple[float, float]:
    align = os.getenv("ALIGN_LONG_AXIS_X", "1") == "1"
    if align and width_m > length_m:
        return width_m, length_m
    return length_m, width_m

# --- Fixture (env-overridable) ---
MODEL           = os.getenv("SPYDR_MODEL", "3h47")
PPF_FIXTURE     = float(os.getenv("SPYDR_PPF", "2200"))  # µmol/s
BARS            = int(os.getenv("SPYDR_BARS", "6"))
BAR_SPACING_IN  = float(os.getenv("SPYDR_BAR_SPACING_IN", "8"))
BAR_WIDTH_IN    = float(os.getenv("SPYDR_BAR_WIDTH_IN", "3"))
BAR_LENGTH_IN   = float(os.getenv("SPYDR_BAR_LENGTH_IN", "47"))
TOTAL_WIDTH_IN  = float(os.getenv("SPYDR_TOTAL_WIDTH_IN", "43"))
# SPYDR_PPF is treated as fixture/system-level photon flux (µmol/s).
# EFF_SCALE is the only scaling applied here (dimmer fraction); this avoids any
# chance of "double derating" from electrical efficiency assumptions.
DERATE          = float(os.getenv("EFF_SCALE", "1.0"))
if DERATE < 0.0:
    print(f"NOTE: EFF_SCALE {DERATE} < 0; clamping to 0.0")
    DERATE = 0.0
if DERATE > 1.0:
    print(f"NOTE: EFF_SCALE {DERATE} > 1; clamping to 1.0 (no overdrive)")
    DERATE = 1.0

# --- Room / placement ---
_len_ft        = float(os.getenv("LENGTH_FT", "12").strip())
_wid_ft        = float(os.getenv("WIDTH_FT",  "12").strip())
LENGTH_M       = _len_ft * 0.3048
WIDTH_M        = _wid_ft * 0.3048
LENGTH_M, WIDTH_M = _maybe_swap_dims(LENGTH_M, WIDTH_M)
WALL_MARGIN_M  = float(os.getenv("MARGIN_IN", "0").strip()) * 0.0254
EDGE_INSET_M   = float(os.getenv("SPYDR_EDGE_INSET_IN", "0").strip()) * 0.0254
MOUNT_Z_M       = float(os.getenv("SPYDR_Z_M", "0.4572"))

# Grid of fixtures (MUST be passed in when you generate)
NX              = int(os.getenv("NX", "1"))
NY              = int(os.getenv("NY", "1"))
ROT_DEG         = float(os.getenv("ROT_DEG", "0"))
SUBPATCH        = int(os.getenv("SUBPATCH_GRID", "3"))
COMPAT          = os.getenv("COMPAT", "0") == "1"

def rect_corners(ccx, ccy, lx, ly, rot_rad):
    c = math.cos(rot_rad); s = math.sin(rot_rad)
    # CCW (face +Z) → we'll reverse for −Z emission
    pts = [(-lx/2, -ly/2), (lx/2,-ly/2), (lx/2,ly/2), (-lx/2,ly/2)]
    out=[]
    for (dx,dy) in pts:
        x = ccx + c*dx - s*dy
        y = ccy + s*dx + c*dy
        out.append((x,y))
    return out  # CCW

def write_poly(fh, mat, name, z, corners_face_up):
    # flip winding so surface normal is DOWN (−Z)
    corners = list(corners_face_up)[::-1]
    fh.write(f"{mat} polygon {name}\n0\n0\n12\n")
    for (x,y) in corners:
        fh.write(f"  {x:.6f} {y:.6f} {z:.6f}\n")
    fh.write("\n")

def write_bar(fh, mat, cx, cy, z, L, W, rot_rad, sub, layout_bars):
    if sub <= 1:
        cu = rect_corners(cx, cy, L, W, rot_rad)
        write_poly(fh, mat, f"bar_{abs(hash((cx,cy)))%10**8}", z, cu)
        layout_bars.append({"corners": cu[::-1]})  # store as facing down (after flip)
        return
    cellL = L / sub; cellW = W / sub
    startL = -0.5*L + 0.5*cellL
    startW = -0.5*W + 0.5*cellW
    c = math.cos(rot_rad); s = math.sin(rot_rad)
    # store the full bar outline (not every subpatch) for overlay clarity
    cu_full = rect_corners(cx, cy, L, W, rot_rad)
    layout_bars.append({"corners": cu_full[::-1]})
    for r in range(sub):
        for cidx in range(sub):
            dx = startL + cidx*cellL
            dy = startW + r*cellW
            cx2 = cx + dx*c - dy*s
            cy2 = cy + dx*s + dy*c
            cu = rect_corners(cx2, cy2, cellL, cellW, rot_rad)
            write_poly(fh, mat, f"bar_{abs(hash((cx2,cy2)))%10**8}", z, cu)

def _fixture_half_extents(L, W, spacing, bars):
    # bars along X, spaced along Y
    hx = 0.5 * L
    oy_max = abs((bars - 1) / 2.0 * spacing)     # farthest bar center offset
    hy = oy_max + 0.5 * W                        # add half bar width
    return hx, hy

def _safe_half_spans(length_m, width_m, wall_margin, hx, hy):
    half_x = 0.5 * length_m
    half_y = 0.5 * width_m
    ax = half_x - wall_margin - hx
    ay = half_y - wall_margin - hy
    if ax <= 0 or ay <= 0:
        raise SystemExit("ERROR: fixture is larger than allowed interior span. Reduce bar length/spacing or margin.")
    return ax, ay

def _evenly_spaced_centers(span_m: float, fixture_span_m: float, count: int, wall_margin_m: float):
    """
    Return center coordinates spanning the room axis with fixtures hugging
    the perimeter (zero wall gap) and even gaps between fixtures.
    """
    if count <= 0:
        return []
    usable = span_m - 2.0 * wall_margin_m
    free = usable - count * fixture_span_m
    if free < -1e-9:
        raise SystemExit("ERROR: fixtures overlap given current room size, margin, or bar lengths.")
    if count == 1:
        return [0.0]
    gap = free / (count - 1)
    start = -0.5 * usable + 0.5 * fixture_span_m
    return [start + i * (fixture_span_m + gap) for i in range(count)]

def _fixture_centers(L, W, spacing):
    hx, hy = _fixture_half_extents(L, W, spacing, BARS)
    # Ensure a single fixture fits; spacing uses edge-to-edge gaps that include the walls.
    eff_margin = WALL_MARGIN_M + EDGE_INSET_M
    _safe_half_spans(LENGTH_M, WIDTH_M, eff_margin, hx, hy)
    xs = _evenly_spaced_centers(LENGTH_M, 2.0 * hx, NX, eff_margin)
    ys = _evenly_spaced_centers(WIDTH_M,  2.0 * hy, NY, eff_margin)
    return [(x, y) for y in ys for x in xs]

def get_fixture_positions():
    L = BAR_LENGTH_IN * IN2M
    W = BAR_WIDTH_IN  * IN2M
    spacing = BAR_SPACING_IN * IN2M
    return [{"x": cx, "y": cy, "z": MOUNT_Z_M} for (cx,cy) in _fixture_centers(L, W, spacing)]

def main():
    L = BAR_LENGTH_IN * IN2M
    W = BAR_WIDTH_IN  * IN2M
    spacing = BAR_SPACING_IN * IN2M
    rot = math.radians(ROT_DEG)
    offs = [ (i - (BARS-1)/2) * spacing for i in range(BARS) ]

    # per-bar radiance ...
    flux_bar = (PPF_FIXTURE * DERATE) / max(BARS,1)
    area_bar = L * W
    rad_bar  = flux_bar / (PI * area_bar)

    out = OUT_DIR / ("emitters_smd_ALL_umol.rad" if COMPAT else "emitters_spydr3_ALL_umol.rad")
    layout = {"model": MODEL, "nx": NX, "ny": NY, "z": MOUNT_Z_M, "bars": BARS, "rot_deg": ROT_DEG,
              "fixtures": []}

    with out.open("w") as fh:
        fh.write(f"# SPYDR 3 {MODEL} | {BARS} bars | PPF={PPF_FIXTURE:.0f} umol/s | derate={DERATE:.3f}\n")
        fh.write(f"# bar: L={L:.4f} m  W={W:.4f} m  spacing={spacing:.4f} m  radiance={rad_bar:.3f} umol/s/sr/m^2\n")
        fh.write("# NOTE: polygons face DOWN (−Z)\n\n")
        fh.write(f"void light spydr3_bar\n0\n0\n3 {rad_bar:.6f} {rad_bar:.6f} {rad_bar:.6f}\n\n")

        centers = _fixture_centers(L, W, spacing)
        for (cx,cy) in centers:
            bars_here=[]
            for b,oy in enumerate(offs):
                write_bar(fh, "spydr3_bar", cx, cy+oy, MOUNT_Z_M, L, W, rot, SUBPATCH, bars_here)
            layout["fixtures"].append({"cx": cx, "cy": cy, "bars": bars_here})

    total_ppf = PPF_FIXTURE * len(layout["fixtures"]) * DERATE
    (OUT_DIR/"spydr3_summary.txt").write_text(
        f"SPYDR 3 {MODEL}\nfixtures={len(layout['fixtures'])}  bars/fixture={BARS}\n"
        f"PPF/fixture={PPF_FIXTURE:.0f} umol/s  PPF_IS_SYSTEM=1  derate(EFF_SCALE)={DERATE:.3f}\n"
        f"TOTAL PPF ≈ {total_ppf:.0f} umol/s\n"
    )
    (OUT_DIR/"spydr3_layout.json").write_text(json.dumps(layout, indent=2))
    print(f"✔ Wrote {out}")
    print(f"✔ Wrote {OUT_DIR/'spydr3_summary.txt'}")
    print(f"✔ Wrote {OUT_DIR/'spydr3_layout.json'}")

if __name__ == "__main__":
    main()
