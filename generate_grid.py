#!/usr/bin/env python3
"""
generate_grid.py

Build a regular sensor grid that matches the current room footprint.

Environment variables:

  LENGTH_M / LENGTH_FT, WIDTH_M / WIDTH_FT  – should match generate_room.py
  GRID_Z             – sensor height above floor (m)
  RESOLUTION_X       – number of points along X
  RESOLUTION_Y       – number of points along Y

Defaults reproduce your 15x15 grid for a 12' x 12' room.
"""

import numpy as np
import sys
import os

FT_TO_M = 0.3048

def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default

def _maybe_swap_dims(L: float, W: float) -> tuple[float, float]:
    align = os.getenv("ALIGN_LONG_AXIS_X", "1") == "1"
    if align and W > L:
        return W, L
    return L, W

L = _f("LENGTH_M", _f("LENGTH_FT", 12.0) * FT_TO_M)   # X dimension (room)
W = _f("WIDTH_M",  _f("WIDTH_FT",  12.0) * FT_TO_M)   # Y dimension (room)
L, W = _maybe_swap_dims(L, W)

GRID_Z       = float(os.environ.get("GRID_Z", "0.005"))
RESOLUTION_X = int(os.environ.get("RESOLUTION_X", "15"))
RESOLUTION_Y = int(os.environ.get("RESOLUTION_Y", "15"))

# Use the same interior notion as emitter layout: subtract wall margin + half module.
# Keep a small inset so sensors are not inside the wall surfaces.
wall_margin_m = _f("WALL_MARGIN_M", 0.127)   # 5 cm inset by default
module_side_m = _f("MODULE_SIDE_M", 0.0)
shrink = 2.0 * (wall_margin_m + module_side_m / 2.0)
L_int = max(0.1, L - shrink)
W_int = max(0.1, W - shrink)

half_x = L_int / 2.0
half_y = W_int / 2.0

# Sample to the usable boundary (after inset)
x_coords = np.linspace(-half_x, half_x, RESOLUTION_X)
y_coords = np.linspace(-half_y, half_y, RESOLUTION_Y)

def print_coords_mode():
    print("x y z")
    for y in y_coords:
        for x in x_coords:
            print(f"{x:.6f} {y:.6f} {GRID_Z:.6f}")

def print_rtrace_mode():
    # rtrace wants: x y z dx dy dz
    for y in y_coords:
        for x in x_coords:
            print(f"{x:.6f} {y:.6f} {GRID_Z:.6f} 0.000000 0.000000 1.000000")

def usage_and_exit():
    print("Usage: generate_grid.py [coords|rtrace]")
    sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        usage_and_exit()
    mode = sys.argv[1].lower()
    if mode == "coords":
        print_coords_mode()
    elif mode == "rtrace":
        print_rtrace_mode()
    else:
        usage_and_exit()
