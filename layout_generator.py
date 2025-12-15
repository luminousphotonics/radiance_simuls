#!/usr/bin/env python3
"""
layout_generator.py

Standalone port of the "Tile -> Fill -> Connect" logic from the Flask app.
Generates module placements for arbitrary rectangular rooms.
"""

from math import floor, ceil, sqrt, pi
import sys

# --- Constants & Known Solutions ---
INF = sys.maxsize
KNOWN_SOLUTIONS = {
    1: [('reverse_L', 'o3'), ('reverse_L', 'o1')],
    2: [('linear3', 'o1'), ('linear3', 'o2'), ('linear3', 'o1'), ('linear3', 'o2')],
    3: [('L', 'o2'), ('L', 'o3'), ('L', 'o4'), ('L', 'o1')],
    4: [('linear3', 'o1'), ('linear3', 'o2'), ('L', 'o3'), ('reverse_L', 'o1'), ('linear3', 'o2'), ('linear3', 'o1')],
    5: [('L', 'o2'), ('linear3', 'o2'), ('linear3', 'o1'), ('linear3', 'o1'), ('linear4', 'o2'), ('reverse_L', 'o2'), ('linear3', 'o1')],
    6: [('L', 'o2'), ('linear4', 'o2'), ('linear4', 'o1'), ('linear3', 'o1'), ('linear4', 'o2'), ('linear3', 'o2'), ('linear3', 'o1'), ('linear3', 'o1')],
    7: [('linear4', 'o2'), ('linear4', 'o2'), ('linear4', 'o1'), ('linear4', 'o1'), ('linear4', 'o2'), ('linear4', 'o2'), ('linear4', 'o1'), ('linear4', 'o1')],
    8: [('linear4', 'o2'), ('linear4', 'o2'), ('L', 'o3'), ('linear4', 'o1'), ('reverse_L', 'o1'), ('linear4', 'o2'), ('linear4', 'o2'), ('linear4', 'o1'), ('linear4', 'o1')]
}

# --- Core Helper Functions ---

def get_n_from_dimensions(d_ft):
    """Calculate base ring index n from dimension in feet."""
    if d_ft <= 4:
        return 2
    return floor((d_ft - 4) / 2) + 2

def get_ring_positions(l):
    """Get integer grid coordinates for a square ring l."""
    d = l + 1
    positions = []
    # top-right quadrant
    for i in range(0, d + 1): positions.append((i, d - i))
    # bottom-right
    for j in range(1, d + 1): positions.append((d - j, -j))
    # bottom-left
    for k in range(1, d + 1): positions.append((-k, -d + k))
    # top-left
    for p in range(1, d + 1): positions.append((-d + p, p))
    # dedupe
    return list(dict.fromkeys(positions))

def get_ring_positions_rect(k, offset):
    """Get integer coordinates for a rectangular extension ring."""
    u_max = offset + k
    v_max = k
    left, top, right, bottom = [], [], [], []
    
    # scan left edge
    u = -u_max
    for v in range(-v_max, v_max + 1):
        if (u + v) % 2 == 0: left.append((u, v))
        
    # scan top edge
    v = v_max
    for u in range(-u_max + 1, u_max + 1):
        if (u + v) % 2 == 0: 
            if (u,v) not in left: top.append((u, v))
            
    # scan right edge
    u = u_max
    for v in range(v_max - 1, -v_max - 1, -1):
        if (u + v) % 2 == 0:
            if (u,v) not in (left + top): right.append((u, v))

    # scan bottom edge
    v = -v_max
    for u in range(u_max - 1, -u_max - 1, -1):
        if (u + v) % 2 == 0:
            if (u,v) not in (left + top + right): bottom.append((u, v))
            
    ring_pos = left + top + right + bottom
    return ring_pos, len(left), len(top), len(right), len(bottom)

def get_min_fixtures(r, corner_set, known_l=None):
    """Dynamic programming to solve minimum fixture tiling for a ring of size r."""
    if known_l is not None and known_l in KNOWN_SOLUTIONS:
        return len(KNOWN_SOLUTIONS[known_l]), KNOWN_SOLUTIONS[known_l]
   
    max_size = 4
    extended = r + max_size
    DP = [INF] * (extended + 1)
    prev = [None] * (extended + 1)
    DP[r] = 0
    
    # Solve backwards
    for pos in range(r - 1, -1, -1):
        # Try linear2
        if pos + 2 <= extended:
            has_corner_middle = any((pos + j) % r in corner_set for j in range(1, 2))
            if not has_corner_middle:
                new_val = 1 + DP[pos + 2]
                if new_val < DP[pos]:
                    DP[pos] = new_val
                    prev[pos] = (2, 'linear2', 'o1')
        # Try linear3
        if pos + 3 <= extended:
            has_corner_middle = any((pos + j) % r in corner_set for j in range(1, 3))
            if not has_corner_middle:
                new_val = 1 + DP[pos + 3]
                if new_val < DP[pos]:
                    DP[pos] = new_val
                    prev[pos] = (3, 'linear3', 'o1')
        # Try linear4
        if pos + 4 <= extended:
            has_corner_middle = any((pos + j) % r in corner_set for j in range(1, 4))
            if not has_corner_middle:
                new_val = 1 + DP[pos + 4]
                if new_val < DP[pos]:
                    DP[pos] = new_val
                    prev[pos] = (4, 'linear4', 'o1')
        # Try L
        if pos + 4 <= extended:
            c = (pos + 1) % r
            m = (pos + 2) % r
            if c in corner_set and m not in corner_set:
                new_val = 1 + DP[pos + 4]
                if new_val < DP[pos]:
                    DP[pos] = new_val
                    prev[pos] = (4, 'L', 'o1')
        # Try Reverse L
        if pos + 4 <= extended:
            c = (pos + 2) % r
            m = (pos + 1) % r
            if c in corner_set and m not in corner_set:
                new_val = 1 + DP[pos + 4]
                if new_val < DP[pos]:
                    DP[pos] = new_val
                    prev[pos] = (4, 'reverse_L', 'o1')
   
    if DP[0] == INF:
        return INF, []
   
    modules = []
    pos = 0
    while pos < r and prev[pos] is not None:
        size, mtype, orient = prev[pos]
        modules.append((mtype, orient))
        pos += size
    return DP[0], modules

def get_module_cobs(r, modules, ring_pos):
    """Assign physical COB coordinates to abstract module definitions."""
    module_groups = []
    pos = 0
    used = set()
    for mtype, orient in modules:
        size = 0
        if mtype == 'linear2': size = 2
        elif mtype == 'linear3': size = 3
        else: size = 4
        
        indices = [(pos + k) % r for k in range(size)]
        
        if all(i not in used for i in indices):
            cobs = [ring_pos[i] for i in indices]
            module_groups.append((mtype, orient, cobs))
            used.update(indices)
            pos += size
            
    return module_groups

def get_central_positions(offset):
    positions = []
    for u in range(-offset, offset + 1):
        if (u + 0) % 2 == 0:
            positions.append((u, 0))
    return positions

def get_central_modules(offset):
    central_cobs = get_central_positions(offset)
    num_c = len(central_cobs)
    
    # Greedy filling logic from app.py
    number4 = num_c // 4
    rem = num_c % 4
    number3 = 0
    number2 = 0
    if rem == 1:
        if number4 >= 1:
            number4 -= 1
            number3 = 1
            number2 = 1
        else:
            number3 = 1
            number2 = num_c - 3
    elif rem == 2:
        if number4 >= 1:
            number4 -= 1
            number3 = 2
        else:
            number2 = 1
    elif rem == 3:
        number3 = 1
        
    central_modules = []
    pos = 0
    for _ in range(number4):
        cobs = central_cobs[pos:pos + 4]
        central_modules.append(('linear4', 'o2', cobs))
        pos += 4
    for _ in range(number3):
        cobs = central_cobs[pos:pos + 3]
        central_modules.append(('linear3', 'o2', cobs))
        pos += 3
    for _ in range(number2):
        cobs = central_cobs[pos:pos + 2]
        central_modules.append(('linear2', 'o2', cobs))
        pos += 2
    return central_modules

def get_extension_groups(base_n, num_steps, shift_x=0.0, shift_y=0.0):
    extension_groups = []
    for i in range(1, num_steps + 1):
        k = base_n + i
        lower = ceil(i / 2)
        upper = floor((2 * base_n + i) / 2)
        num_cobs = upper - lower + 1 if upper >= lower else 0
        if num_cobs < 2: continue
        
        # Greedy logic similar to central modules
        rem = num_cobs % 4
        number4 = num_cobs // 4
        number3 = 0
        number2 = 0
        if rem == 1:
            if number4 >= 1: number4 -= 1; number3 = 1; number2 = 1
            else: number3 = 1; number2 = num_cobs - 3
        elif rem == 2:
            if number4 >= 1: number4 -= 1; number3 = 2
            else: number2 = 1
        elif rem == 3: number3 = 1
            
        modules_for_column = []
        for _ in range(number4): modules_for_column.append(('linear4', 'o1'))
        for _ in range(number3): modules_for_column.append(('linear3', 'o1'))
        for _ in range(number2): modules_for_column.append(('linear2', 'o1'))
        
        cobs = [(float(x) + shift_x, float(k - x) + shift_y) for x in range(lower, upper + 1)]
        pos = 0
        for mtype, orient in modules_for_column:
            size = int(mtype[-1])
            group_cobs = cobs[pos:pos + size]
            extension_groups.append((mtype, orient, group_cobs))
            pos += size
            
    return extension_groups


def generate_layout(length_ft, width_ft):
    """
    Main Entry Point.
    Returns:
       layout_data (dict): Contains 'all_positions' (list of (x,y) tuples) 
                           and 'module_groups' (list of tuples (mtype, orient, cobs)).
    """
    min_dim = min(length_ft, width_ft)
    max_dim = max(length_ft, width_ft)
    
    # 1. Determine Base Unit Size (n)
    base_n = get_n_from_dimensions(min_dim)
    
    # 2. Determine Tiling Strategy (Square vs Rect Extension)
    c = 2.0
    unit = min_dim + c
    min_rect_long = min_dim + 4
    max_s = floor((max_dim + c) / unit)
    s = max_s
    found = False
    has_rect = False
    rect_long = 0
    
    while s >= 0 and not found:
        if s == 0:
            rect_long = max_dim
            found = True
            has_rect = True
        else:
            pure_length = min_dim * s + c * (s - 1)
            rem = max_dim - pure_length
            if rem == 0:
                found = True
                has_rect = False
            elif rem > c and (rem - c) >= min_rect_long:
                rect_long = rem - c
                found = True
                has_rect = True
            else:
                s -= 1

    # 3. Build Layer Modules (Pre-calc ring solutions)
    layer_modules = []
    for l in range(1, base_n):
        # Ring indices: 0, l+1, 2(l+1), 3(l+1)
        corners = set([0, l + 1, 2 * (l + 1), 3 * (l + 1)])
        min_fix, modules = get_min_fixtures(4 * (l + 1), corners, l)
        layer_modules.append(modules)

    module_groups = []
    shift_step = base_n + 1

    # 4. Place Square Tiles
    for j in range(s):
        sx = j * shift_step
        sy = j * shift_step
        
        # Centerpiece
        if base_n >= 2:
            centerpiece = [(sx, sy), (1 + sx, sy), (sx, 1 + sy), (-1 + sx, sy), (sx, -1 + sy)]
            module_groups.append(('centerpiece', 'o1', centerpiece))
            
        # Concentric Rings
        for l in range(1, base_n):
            ring_pos = get_ring_positions(l)
            ring_pos_shifted = [(xx + sx, yy + sy) for xx, yy in ring_pos]
            ring_groups = get_module_cobs(4 * (l + 1), layer_modules[l - 1], ring_pos_shifted)
            module_groups.extend(ring_groups)
            
    # 5. Place Connectors (between squares)
    for m in range(s - 1):
        csx = m * shift_step
        csy = m * shift_step
        connector_groups = get_extension_groups(base_n, 1, csx, csy)
        module_groups.extend(connector_groups)
        
    # 6. Place Rectangular Extension (if needed)
    if has_rect:
        a = get_n_from_dimensions(rect_long)
        offset = a - base_n
        if offset % 2 == 1:
            a -= 1; offset -= 1
            
        if s > 0:
            last_csx = (s - 1) * shift_step
            last_csy = (s - 1) * shift_step
            # Connector to rect
            last_connector_groups = get_extension_groups(base_n, 1, last_csx, last_csy)
            module_groups.extend(last_connector_groups)
            
            # Calculate shift based on connector index logic
            last_connector_u = base_n + 1 + (s - 1) * 2 * (base_n + 1)
            shift_u = last_connector_u + 1 + (offset + base_n)
        else:
            shift_u = 0
            
        shift_v = 0
        rect_modules = get_central_modules(offset)
        
        # Rings for Rect Extension
        for k in range(1, base_n + 1):
            ring_pos, len_left, len_top, len_right, len_bottom = get_ring_positions_rect(k, offset)
            r = len(ring_pos)
            corner_set = set([0, len_left - 1, len_left + len_top - 1, len_left + len_top + len_right - 1])
            min_fix, modules = get_min_fixtures(r, corner_set)
            ring_groups = get_module_cobs(r, modules, ring_pos)
            rect_modules.extend(ring_groups)
            
        # Transform Rect Coords back to Diamond Space
        for i in range(len(rect_modules)):
            mtype, orient, local_cobs = rect_modules[i]
            transformed_cobs = [
                ((u + shift_u + v + shift_v) / 2, (u + shift_u - (v + shift_v)) / 2) 
                for u, v in local_cobs
            ]
            rect_modules[i] = (mtype, orient, transformed_cobs)
            
        module_groups.extend(rect_modules)

    # 7. Collect All Positions
    all_positions = []
    for _, _, cobs in module_groups:
        all_positions.extend(cobs)
    all_positions = list(set(all_positions))

    return {
        "all_positions": all_positions,
        "module_groups": module_groups
    }

if __name__ == "__main__":
    # Simple CLI test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("L", type=float)
    parser.add_argument("W", type=float)
    args = parser.parse_args()
    
    data = generate_layout(args.L, args.W)
    print(f"Generated {len(data['all_positions'])} modules for {args.L}x{args.W} ft room.")
