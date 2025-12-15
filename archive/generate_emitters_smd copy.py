#!/usr/bin/env python3
# generate_emitters_smd.py • v3.4
# - Optics toggle via OPTICS=none|lens
# - Lens parameters override via LENS_CONFIG=path/to/lens_config.json
# - CHANGE: no real args to brightfunc; per-ring wrapper funcs written to beam.cal

from __future__ import annotations
import json, math, os, random
from pathlib import Path
from typing import Dict, List, Tuple

# ── Room / lattice
ROOM_SIDE_M   = 3.6576          # 12 ft
WALL_MARGIN_M = 0.127           # 5 in
RINGS         = 8               # 0..7 → 113 modules
MOUNT_Z_M     = 0.4572

# ── Macro patch (one per module; optional sub-grid)
# NEW (drop-in)
MODULE_SIDE_M = float(os.getenv("MODULE_SIDE_M", "0.12"))  # 120 mm board footprint; adjust if needed
PATCH_SIDE_M  = MODULE_SIDE_M
SUBPATCH_GRID = int(os.getenv("SUBPATCH_GRID", "3"))       # spread emission across the board

# ── Electrical make-up per module
COUNT_PER_MODULE: Dict[str, int] = {
    "WW": 32, "CW": 32, "R": 20, "B": 12, "FR": 8, "C": 4, "UV": 6,
}
PER_LED_W: Dict[str, float] = {
    "WW": 0.60, "CW": 0.60, "R": 0.45, "B": 0.35, "C": 0.35, "FR": 0.35, "UV": 0.25,
}
PPE_UMOL_PER_J: Dict[str, float] = {
    "WW": 2.8, "CW": 2.9, "R": 4.1, "B": 2.8, "C": 2.0, "FR": 3.6, "UV": 1.0,
}

# ── Per-ring electrical power (per module)
RING_POWER_W: Dict[int, float] = {
    0: 55.0, 1: 55.0, 2: 65.0, 3: 65.0, 4: 55.0, 5: 65.0, 6: 40.0, 7: 80.0
}
EFF_SCALE = 1.0

# ── Default lens map
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

# Optional JSON override
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
            L = int(k)
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

# ── Misc
OUT_DIR  = Path("ies_sources"); OUT_DIR.mkdir(parents=True, exist_ok=True)
CAL_NAME = "beam.cal"
CAL_FILE = OUT_DIR / CAL_NAME
PI = math.pi
SQRT2 = math.sqrt(2.0)
EPS = 1e-12
rng = random.Random(4242)

# ── Lattice helpers
def _solve_spacing(rings: int, room_side: float, wall_margin: float) -> float:
    L = rings - 1
    usable = room_side - 2.0 * wall_margin
    return usable / (math.sqrt(2.0) * L)

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

def get_module_positions() -> Tuple[List[dict], float]:
    spacing = _solve_spacing(RINGS, ROOM_SIDE_M, WALL_MARGIN_M)
    pos = []
    for L in range(RINGS):
        for i,j in _ring_ij(L):
            x,y = _ij_to_xy(i,j,spacing)
            pos.append({"ring": L, "i": i, "j": j,
                        "x": round(x,6), "y": round(y,6), "z": MOUNT_Z_M})
    return pos, spacing

# ── Power → photons → radiance
def _per_module_nominal_watts() -> float:
    return sum(COUNT_PER_MODULE[ch] * PER_LED_W[ch] for ch in COUNT_PER_MODULE)

def _module_photon_flux_umol_s(ring: int) -> float:
    target_w = RING_POWER_W[ring] * EFF_SCALE
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
    return 0.0 if A <= 0 else phi / (A * PI)  # Lambert baseline

# ── Optics normalization helpers
def _m_from_fwhm(deg: float) -> float:
    th = math.radians(deg * 0.5)
    c = max(math.cos(th), 1e-6)
    return math.log(0.5) / math.log(c)

def _avg_sym_cos(m: float) -> float:
    # 1/π ∫ c^m * c dΩ = 2/(m+2) ; m=0 → 1.0
    return 2.0 / (m + 2.0)

def _avg_bat_cos(m: float, k: float) -> float:
    # 1/π ∫ c^m * (1 + k(1-4c+4c^2)) * c dΩ
    return (2.0 / (m + 2.0)) + k * (
        (2.0 / (m + 2.0)) - (8.0 / (m + 3.0)) + (8.0 / (m + 4.0))
    )

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
        s += math.exp(-ln2*rho2)   # <-- no extra *sz here
    return s / samples



# ── beam.cal content + wrapper support
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

# ── RAD writers
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
    # Use wrapper functions with zero real args
    fh.write(f"void brightfunc {patt_name}\n")
    fh.write(f"2 {funcname} {CAL_NAME}\n")   # <- func then file
    fh.write("0\n")
    fh.write("0\n\n")

# ── Main
def main():
    positions, spacing = get_module_positions()
    rad_lambert = {L: _module_radiance_umol_per_sr_m2(L) for L in range(RINGS)}
    use_optics = (OPTICS_MODE == "lens")
    if use_optics:
        _write_beam_cal_base(CAL_FILE)

    out = OUT_DIR / "emitters_smd_ALL_umol.rad"
    with out.open("w") as fh:
        fh.write("# SMD macro emitters in µmol/s/sr/m²\n")
        fh.write(f"# rings={RINGS} modules={len(positions)} spacing={spacing:.4f} m (outer margin {WALL_MARGIN_M:.3f} m)\n")
        fh.write(f"# patch side = {PATCH_SIDE_M*1e3:.1f} mm  subgrid = {SUBPATCH_GRID}x{SUBPATCH_GRID}\n")
        fh.write(f"# optics = {'enabled (normalized patterns)' if use_optics else 'disabled (Lambertian)'}\n\n")

        for idx, p in enumerate(positions):
            L, cx, cy, cz = p["ring"], p["x"], p["y"], p["z"]
            base_rad = rad_lambert[L]
            patt = None

            if use_optics and L in LENS_RINGS and L > 0:
                kind, cfg = LENS_RINGS[L]
                if kind == "sym":
                    m = _m_from_fwhm(float(cfg["fwhm"]))
                    gain = 1.0 / max(_avg_sym_cos(m), EPS)
                    fname = f"f_sym_L{L}"
                    _add_wrapper(fname, f"{gain:.9f} * pwr(c, {m:.9f})")
                    patt = f"p_sym_L{L}"; _write_brightfunc_ref(fh, patt, fname)

                elif kind == "bat":
                    m = _m_from_fwhm(float(cfg["fwhm"])); k = float(cfg.get("bat_k", 0.75))
                    k = float(cfg.get("bat_k", 0.75))
                    gain = 1.0 / max(_avg_bat_cos(m, k), EPS)
                    fname = f"f_bat_L{L}"
                    _add_wrapper(fname, f"{gain:.9f} * pwr(c, {m:.9f}) * (1 + {k:.9f}*(1 - 4*c + 4*c*c))")
                    patt = f"p_bat_L{L}"; _write_brightfunc_ref(fh, patt, fname)

                elif kind == "ellip":
                    ex, ey = float(cfg["ex"]), float(cfg["ey"])
                    avg = _avg_ellip_cos(ex, ey, samples=4000)
                    gain = 1.0 / max(avg, EPS)
                    tx0 = math.tan(math.radians(ex*0.5))
                    ty0 = math.tan(math.radians(ey*0.5))

                    # per-module radial axis (ux,uy)
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
                    i, j = p["i"], p["j"]
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

    # Append wrapper funcs after file is written
    if use_optics:
        _flush_wrappers()

    # Summary
    total_w = 0.0; lines = []
    for L in range(RINGS):
        nL = 1 if L == 0 else 4*L
        wL = RING_POWER_W[L]*EFF_SCALE
        total_w += nL*wL
        lines.append(f"    ring {L}: {wL:5.2f} W × {nL} mods")

    avg_ppe = (
        sum(COUNT_PER_MODULE[ch]*PER_LED_W[ch]*PPE_UMOL_PER_J[ch] for ch in COUNT_PER_MODULE)
        / max(_per_module_nominal_watts(), 1e-9)
    )
    total_umol = sum((_module_photon_flux_umol_s(L) * (1 if L==0 else 4*L)) for L in range(RINGS))

    summ = OUT_DIR / "smd_summary.txt"
    summ.write_text(
        "SMD macro emitter summary:\n"
        f"  rings       : {RINGS}  (center=ring 0)\n"
        f"  modules     : {len(positions)}  (A001844)\n"
        f"  spacing     : {spacing:.4f} m  (outer margin {WALL_MARGIN_M:.3f} m)\n"
        f"  patch side  : {PATCH_SIDE_M*1e3:.1f} mm\n"
        f"  optics      : {'enabled (normalized patterns)' if use_optics else 'disabled (Lambertian)'}\n"
        f"  subgrid     : {SUBPATCH_GRID}x{SUBPATCH_GRID}\n"
        f"  per-ring watts (per module):\n" + "\n".join(lines) + "\n"
        f"  total electrical power ≈ {total_w:.1f} W\n"
        f"  avg PPE (mix) ≈ {avg_ppe:.3f} µmol/J → total photons ≈ {total_umol:.0f} µmol/s\n"
    )

    print("SMD macro emitter summary:")
    print(f"  rings       : {RINGS}  (center=ring 0)")
    print(f"  modules     : {len(positions)}  (A001844)")
    print(f"  spacing     : {spacing:.4f} m  (outer margin {WALL_MARGIN_M:.3f} m)")
    print(f"  patch side  : {PATCH_SIDE_M*1e3:.1f} mm")
    print(f"  optics      : {'enabled (normalized patterns)' if use_optics else 'disabled (Lambertian)'}")
    print(f"  subgrid     : {SUBPATCH_GRID}x{SUBPATCH_GRID}")
    print("  per-ring watts (per module):"); [print(s) for s in lines]
    print(f"  total electrical power ≈ {total_w:.1f} W")
    print(f"  avg PPE (mix) ≈ {avg_ppe:.3f} µmol/J → total photons ≈ {total_umol:.0f} µmol/s")
    print(f"✔ Wrote {out}")
    if use_optics: print(f"✔ Wrote {CAL_FILE}")
    print(f"✔ Wrote {summ}")

if __name__ == "__main__":
    main()
