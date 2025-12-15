#!/usr/bin/env python3
"""
generate_emitters_manual.py • v1.9

Models COB emitters from ELECTRICAL WATTS/COB using PPE (μmol/J) and CCT (K).
- photon_flux [μmol/s] = P_elec [W] * LED_PPE_UMOL_PER_J
- radiant_PAR_power [W] = photon_flux * E_per_umol(CCT, 400–700 nm)
- radiance [W/sr/m^2] = radiant_PAR_power / (area * pi)

Notes
- E_per_μmol is computed from a blackbody at the given CCT over 400–700 nm.
- Polygons are ordered to face DOWN (−Z). This was the bug before.
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, List

# ── Physical constants
NA = 6.02214076e23   # 1/mol
H  = 6.62607015e-34  # J·s
C  = 299792458.0     # m/s
K_B = 1.380649e-23   # J/K
EPSILON = 1e-9

# ── User config
COB_SIZE_M = 0.02     # COB square patch side (m)
EFF_SCALE  = 1.0      # optional global multiplier

LED_PPE_UMOL_PER_J = 3.0     # μmol/J
LED_CCT_K          = 3000.0  # Kelvin

# Geometry
SPACING_M = 0.4311
MOUNT_Z_M = 0.6096
LAYERS    = 6

# Electrical watts PER COB for each layer
LAYER_POWER_W: Dict[int, float] = {
    1: 50.0,
    2: 50.0,
    3: 50.0,
    4: 50.0,
    5: 50.0,
    6: 50.0,
}

# Perimeter
ROOM_SIDE_M    = 3.6576
N_PERIM_COBS   = 48
PERIM_OFFSET_M = 0.001
COB_POWER_W    = 80.0   # W per perimeter COB

# Output
OUT_DIR = Path('ies_sources').resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)
SCENE_FILE = OUT_DIR / 'emitters_manual.rad'


# ── Position generators
def generate_center_positions(n_layers: int, spacing: float, z: float) -> List[dict]:
    coords: Dict[tuple, int] = {(0, 0): 1}
    for L in range(2, n_layers + 1):
        r = L - 1
        for xi in range(-r, r + 1):
            yv = r - abs(xi)
            for yy in (yv, -yv):
                coords.setdefault((xi, yy), L)
    pts: List[dict] = []
    s2 = math.sqrt(2)
    for idx, ((xi, yi), layer) in enumerate(sorted(coords.items(), key=lambda kv: (kv[1], kv[0]))):
        x = round((xi - yi) * spacing / s2, 6)
        y = round((xi + yi) * spacing / s2, 6)
        pts.append({'idx': idx, 'layer': layer, 'x': x, 'y': y, 'z': z})
    return pts

def generate_perimeter_positions(n_emitters: int, side_length: float, z: float, offset: float) -> List[dict]:
    perim = 4 * side_length
    step  = perim / n_emitters
    half  = side_length / 2
    positions = []
    d = step / 2.0
    for idx in range(n_emitters):
        if d < side_length:                # bottom
            x, y = -half + d, -half + offset
        elif d < 2 * side_length:          # right
            x, y = half - offset, -half + (d - side_length)
        elif d < 3 * side_length:          # top
            x, y = half - (d - 2 * side_length), half - offset
        else:                               # left
            x, y = -half + offset, half - (d - 3 * side_length)
        positions.append({'idx': idx, 'x': round(x,6), 'y': round(y,6), 'z': z})
        d += step
    return positions


# ── Blackbody helpers (400–700 nm band)
def _planck_power_per_wavelength(lam_m: float, T: float) -> float:
    denom = math.expm1((H*C)/(lam_m*K_B*T))
    return (2*H*C*C) / (lam_m**5 * denom)

def _photon_flux_density(lam_m: float, T: float) -> float:
    return _planck_power_per_wavelength(lam_m, T) * (lam_m / (H*C))

def energy_per_umol_from_cct(T_K: float, lam_min_nm: float = 400.0, lam_max_nm: float = 700.0) -> float:
    """Energy [J] carried by 1 μmol of photons for a blackbody at T_K over 400–700 nm."""
    n = 20000
    lam0 = lam_min_nm * 1e-9
    lam1 = lam_max_nm * 1e-9
    step = (lam1 - lam0) / (n - 1)
    total_power = 0.0
    total_photons = 0.0
    lam = lam0
    for _ in range(n):
        P = _planck_power_per_wavelength(lam, T_K)
        N = _photon_flux_density(lam, T_K)
        total_power += P * step
        total_photons += N * step
        lam += step
    if total_photons <= EPSILON:
        E_ph = (H*C) / max(550e-9, 1e-9)
    else:
        E_ph = total_power / total_photons
    return E_ph * NA * 1e-6  # J/μmol


def elec_watts_to_radiance_w_per_sr_m2(elec_watts: float, patch_area: float, T_K: float) -> float:
    """Electrical watts → Radiance RGB (W/sr/m^2) using PPE and blackbody(CCT) over 400–700 nm."""
    if patch_area <= EPSILON or elec_watts <= 0:
        return 0.0
    e_umol = energy_per_umol_from_cct(T_K)
    rad_watts_par = elec_watts * LED_PPE_UMOL_PER_J * e_umol  # W in PAR band
    return rad_watts_par / (patch_area * math.pi)


# ── Scene writer (DOWNWARD normals)
def write_scene(scene_file: Path, center_pts: List[dict], perim_pts: List[dict], cob_size: float) -> None:
    hs = cob_size / 2.0
    area = cob_size * cob_size
    e_umol = energy_per_umol_from_cct(LED_CCT_K)

    with scene_file.open('w') as fh:
        fh.write('# Combined manual COB array: center + perimeter\n')
        fh.write(f'# PPE={LED_PPE_UMOL_PER_J:.3f} μmol/J, CCT={LED_CCT_K:.1f} K, '
                 f'E_per_μmol={e_umol:.6f} J/μmol (400–700 nm)\n')
        fh.write(f'# Approx radiant PAR per electrical watt: {LED_PPE_UMOL_PER_J*e_umol:.3f} W/W\n')
        fh.write('# NOTE: Polygons ordered to face DOWN (−Z)\n\n')

        # Center COBs
        for p in center_pts:
            layer = p['layer']
            elec_w = LAYER_POWER_W.get(layer, 0.0) * EFF_SCALE
            rad_val = elec_watts_to_radiance_w_per_sr_m2(elec_w, area, LED_CCT_K)

            mat = f"cob_ctr_mat_{p['idx']:03d}"
            geo = f"cob_ctr_surf_{p['idx']:03d}"

            fh.write(f"# Center COB {p['idx']:03d}, Layer {layer}, Elec {elec_w:.3f} W, Radiance {rad_val:.6f} W/sr/m2\n")
            fh.write(f"void light {mat}\n0\n0\n3 {rad_val:.6f} {rad_val:.6f} {rad_val:.6f}\n\n")
            fh.write(f"{mat} polygon {geo}\n0\n0\n12\n")
            # CLOCKWISE when viewed from above -> normal DOWN (−Z)
            fh.write(f"  {p['x'] - hs:.6f} {p['y'] + hs:.6f} {p['z']:.6f}\n")  # TL
            fh.write(f"  {p['x'] + hs:.6f} {p['y'] + hs:.6f} {p['z']:.6f}\n")  # TR
            fh.write(f"  {p['x'] + hs:.6f} {p['y'] - hs:.6f} {p['z']:.6f}\n")  # BR
            fh.write(f"  {p['x'] - hs:.6f} {p['y'] - hs:.6f} {p['z']:.6f}\n\n")# BL

        # Perimeter COBs
        for p in perim_pts:
            elec_w = COB_POWER_W * EFF_SCALE
            rad_val = elec_watts_to_radiance_w_per_sr_m2(elec_w, area, LED_CCT_K)

            mat = f"cob_perim_mat_{p['idx']:02d}"
            geo = f"cob_perim_surf_{p['idx']:02d}"

            fh.write(f"# Perimeter COB {p['idx']:02d}, Elec {elec_w:.3f} W, Radiance {rad_val:.6f} W/sr/m2\n")
            fh.write(f"void light {mat}\n0\n0\n3 {rad_val:.6f} {rad_val:.6f} {rad_val:.6f}\n\n")
            fh.write(f"{mat} polygon {geo}\n0\n0\n12\n")
            # CLOCKWISE when viewed from above -> normal DOWN (−Z)
            fh.write(f"  {p['x'] - hs:.6f} {p['y'] + hs:.6f} {p['z']:.6f}\n")  # TL
            fh.write(f"  {p['x'] + hs:.6f} {p['y'] + hs:.6f} {p['z']:.6f}\n")  # TR
            fh.write(f"  {p['x'] + hs:.6f} {p['y'] - hs:.6f} {p['z']:.6f}\n")  # BR
            fh.write(f"  {p['x'] - hs:.6f} {p['y'] - hs:.6f} {p['z']:.6f}\n\n")# BL


# ── Main
if __name__ == '__main__':
    ctr = generate_center_positions(LAYERS, SPACING_M, MOUNT_Z_M)
    per = generate_perimeter_positions(N_PERIM_COBS, ROOM_SIDE_M, MOUNT_Z_M, PERIM_OFFSET_M)
    write_scene(SCENE_FILE, ctr, per, COB_SIZE_M)
    print(f'✔ Generated {len(ctr)} center + {len(per)} perimeter COBs (total {len(ctr)+len(per)}) → {SCENE_FILE}')
