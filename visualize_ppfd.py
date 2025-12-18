#!/usr/bin/env python3
# visualize_ppfd.py • v4.2
# - Overlays: smd / spydr3 / quantum / cob / both / none / auto
# - SPYDR overlay reads ies_sources/spydr3_layout.json to draw true bar rectangles

import argparse, json
import importlib
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from scipy.interpolate import griddata
from matplotlib.patches import Polygon

def parse_args():
    ap = argparse.ArgumentParser(description="PPFD visualizations with unified color scaling.")
    ap.add_argument("--input", default="ppfd_map.txt")
    ap.add_argument("--outdir", default="ppfd_visualizations")
    ap.add_argument("--grid-size", type=int, default=15,
                    help="Interpolation grid when data not perfectly gridded")
    ap.add_argument("--vmin", type=float, default=0.0,
                    help="Lower color/z limit for plots")
    ap.add_argument("--vmax", type=float, default=1750.0,
                    help="Upper color/z limit for plots")
    ap.add_argument("--cmap", default="jet")
    ap.add_argument("--dpi", type=int, default=300)

    # NEW: annotations on by default; add --no-annot to turn them off
    ap.add_argument("--annot", dest="annot", action="store_true",
                    help="Annotate seaborn heatmap cells")
    ap.add_argument("--no-annot", dest="annot", action="store_false",
                    help="Disable annotations on heatmap")
    ap.set_defaults(annot=True)

    ap.add_argument(
        "--overlay",
        choices=["auto","smd","spydr3","quantum","cob","both","none"],
        default="auto",
        help="Which hardware overlay to draw on heatmaps",
    )
    return ap.parse_args()


args = parse_args()
INPUT_FILE = args.input
OUTDIR = Path(args.outdir); OUTDIR.mkdir(parents=True, exist_ok=True)
GRID_SIZE = args.grid_size
VMIN, VMAX = float(args.vmin), float(args.vmax)
CMAP = args.cmap
DPI = args.dpi

# ---- Overlays ---------------------------------------------------------------
overlay_points = []   # tuples: (label, [(x,y,z), ...], scatter_kwargs)
overlay_polys  = []   # tuples: (label, [ [(x,y),...4], ...], poly_kwargs)
overlay_room_bounds = None  # (L, W) in meters if available from layout JSON
overlay_room_bounds = None  # (L, W) in meters if present in layout JSON

SMD_JSON = Path("ies_sources/smd_layout.json")
SPYDR_JSON = Path("ies_sources/spydr3_layout.json")
QUANTUM_JSON = Path("ies_sources/quantum_layout.json")
COB_JSON = Path("ies_sources/cob_layout.json")

def _safe_mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except Exception:
        return 0.0

def auto_pick_overlay_mode() -> str:
    candidates = []
    for name, path in (("smd", SMD_JSON), ("spydr3", SPYDR_JSON), ("quantum", QUANTUM_JSON), ("cob", COB_JSON)):
        if path.exists():
            candidates.append((name, _safe_mtime(path)))
    if not candidates:
        return "smd"
    # pick most recent
    candidates.sort(key=lambda t: t[1], reverse=True)
    return candidates[0][0]

def try_overlay_smd():
    # Prefer authoritative snapshot
    if SMD_JSON.exists():
        try:
            data = json.loads(SMD_JSON.read_text())
            if data.get("units") != "meters":
                print("Overlay (SMD) skipped: smd_layout.json not in meters")
                return
            global overlay_room_bounds
            room = data.get("room")
            if room and "L" in room and "W" in room:
                overlay_room_bounds = (float(room["L"]), float(room["W"]))
            z0 = float(data.get("z", 0.0))
            pts = [(float(p["x"]), float(p["y"]), float(p.get("z", z0)))
                   for p in data.get("positions", [])]
            overlay_points.append(("SMD modules", pts,
                                   dict(marker='x', s=35, color='white', linewidths=1.2)))
            print(f"Overlay: {len(pts)} SMD modules (from JSON)")
            return
        except Exception as e:
            print("Overlay (SMD) JSON read failed; falling back to module:", e)

    # Fallback: import module API
    try:
        gem = importlib.import_module("generate_emitters_smd")
        mp = gem.get_module_positions()
        positions, _spacing = mp if (isinstance(mp, tuple) and len(mp) == 2) else (mp, None)
        pts = [(p['x'], p['y'], p.get('z', 0.0)) for p in positions]
        overlay_points.append(("SMD modules", pts,
                               dict(marker='x', s=35, color='white', linewidths=1.2)))
        print(f"Overlay: {len(pts)} SMD modules (from module)")
    except Exception as e:
        print("Overlay (SMD) unavailable:", e)


def try_overlay_spydr3():
    if SPYDR_JSON.exists():
        try:
            data = json.loads(SPYDR_JSON.read_text())
            nfx = len(data.get("fixtures", []))
            print(f"Overlay: Competitor from JSON ({nfx} fixtures)")
            global overlay_room_bounds
            room = data.get("room")
            if room and "L" in room and "W" in room:
                overlay_room_bounds = (float(room["L"]), float(room["W"]))
            bars = []
            centers = []
            z0 = float(data.get("z", 0.0))
            for fx in data["fixtures"]:
                centers.append((fx["cx"], fx["cy"], z0))
                for b in fx["bars"]:
                    xy = [(float(x), float(y)) for (x, y) in b["corners"]]
                    bars.append(xy)
            overlay_polys.append(("Competitor bars", bars,
                                  dict(edgecolor='black', facecolor='none', linewidth=1.0, alpha=0.9)))
            overlay_points.append(("Competitor centers", centers,
                                   dict(marker='s', s=28, facecolors='none', edgecolors='black', linewidths=1.0)))
            return
        except Exception as e:
            print("Overlay (Competitor) JSON read failed; falling back to module:", e)

    # Fallback: centers only
    try:
        gsp = importlib.import_module("generate_emitters_spydr3")
        positions = gsp.get_fixture_positions()
        pts = [(p['x'], p['y'], p.get('z', 0.0)) for p in positions]
        overlay_points.append(("Competitor fixtures", pts,
                               dict(marker='s', s=50, facecolors='none', edgecolors='black', linewidths=1.2)))
        print(f"Overlay: {len(pts)} Competitor fixtures (centers only)")
    except Exception as e:
        print("Overlay (Competitor) unavailable:", e)

def try_overlay_quantum():
    if QUANTUM_JSON.exists():
        try:
            data = json.loads(QUANTUM_JSON.read_text())
            n = len(data.get("positions", []))
            global overlay_room_bounds
            room = data.get("room")
            if room and "L" in room and "W" in room:
                overlay_room_bounds = (float(room["L"]), float(room["W"]))
            patch_x_default = float(data.get("patch_x", data.get("patch_side", 0.12)))
            patch_y_default = float(data.get("patch_y", data.get("patch_side", 0.12)))
            z0 = float(data.get("z", 0.0))
            centers = []
            polys = []
            for p in data.get("positions", []):
                cx = float(p["x"]); cy = float(p["y"]); cz = float(p.get("z", z0))
                lx = float(p.get("lx", patch_x_default)); ly = float(p.get("ly", patch_y_default))
                hx = 0.5 * lx; hy = 0.5 * ly
                polys.append([(cx - hx, cy - hy), (cx + hx, cy - hy), (cx + hx, cy + hy), (cx - hx, cy + hy)])
                centers.append((cx, cy, cz))
            overlay_polys.append(("Quantum boards", polys,
                                  dict(edgecolor='white', facecolor='none', linewidth=0.8, alpha=0.9)))
            overlay_points.append(("Quantum centers", centers,
                                   dict(marker='o', s=10, facecolors='red', edgecolors='red', linewidths=0.6)))
            print(f"Overlay: Quantum boards from JSON ({n} modules)")
            return
        except Exception as e:
            print("Overlay (Quantum) JSON read failed:", e)

    # fallback: try module API
    try:
        gq = importlib.import_module("generate_emitters_quantum")
        positions = gq.get_module_positions()[0]
        pts = [(p['x'], p['y'], p.get('z', 0.0)) for p in positions]
        overlay_points.append(("Quantum modules", pts,
                               dict(marker='s', s=24, facecolors='none', edgecolors='white', linewidths=0.9)))
        print(f"Overlay: {len(pts)} Quantum modules (from module)")
    except Exception as e:
        print("Overlay (Quantum) unavailable:", e)

def try_overlay_cob():
    if COB_JSON.exists():
        try:
            data = json.loads(COB_JSON.read_text())
            if data.get("units") != "meters":
                print("Overlay (COB) skipped: cob_layout.json not in meters")
                return

            global overlay_room_bounds
            room = data.get("room") or {}
            if "L" in room and "W" in room:
                overlay_room_bounds = (float(room["L"]), float(room["W"]))

            z0 = float(data.get("z_emit_m", data.get("z", 0.0)))

            centers = []
            for c in data.get("cobs", []):
                ctr = c.get("center") or c.get("center_xyz") or c.get("center_m")
                if not (isinstance(ctr, list) and len(ctr) >= 2):
                    continue
                cx = float(ctr[0]); cy = float(ctr[1]); cz = float(ctr[2]) if len(ctr) >= 3 else z0
                centers.append((cx, cy, cz))

            polys = []
            for s in data.get("strip_segments", []):
                corners = s.get("corners_xy") or s.get("corners")
                if not (isinstance(corners, list) and len(corners) >= 4):
                    continue
                try:
                    xy = [(float(p[0]), float(p[1])) for p in corners[:4]]
                except Exception:
                    continue
                polys.append(xy)

            if polys:
                overlay_polys.append(("Perimeter strips", polys,
                                      dict(edgecolor='red', facecolor='none', linewidth=1.2, alpha=0.95)))
            if centers:
                overlay_points.append(("COB centers", centers,
                                       dict(marker='o', s=10, color='white', linewidths=0.6)))

            print(f"Overlay: COB from JSON ({len(centers)} COBs, {len(polys)} strip segments)")
            return
        except Exception as e:
            print("Overlay (COB) JSON read failed; falling back to module:", e)

    try:
        gc = importlib.import_module("generate_emitters_cob")
        positions = gc.get_module_positions()[0]
        pts = [(p["x"], p["y"], p.get("z", 0.0)) for p in positions]
        overlay_points.append(("COB centers", pts,
                               dict(marker='o', s=10, color='white', linewidths=0.6)))
        print(f"Overlay: {len(pts)} COB centers (from module)")
    except Exception as e:
        print("Overlay (COB) unavailable:", e)

if args.overlay == "auto":
    mode = auto_pick_overlay_mode()
    if mode == "smd":
        try_overlay_smd()
    elif mode == "spydr3":
        try_overlay_spydr3()
    elif mode == "quantum":
        try_overlay_quantum()
    elif mode == "cob":
        try_overlay_cob()
elif args.overlay == "smd":
    try_overlay_smd()
elif args.overlay == "spydr3":
    try_overlay_spydr3()
elif args.overlay == "quantum":
    try_overlay_quantum()
elif args.overlay == "cob":
    try_overlay_cob()
elif args.overlay == "both":
    try_overlay_smd(); try_overlay_spydr3()
# else: none


# ---- Load PPFD --------------------------------------------------------------
try:
    df = pd.read_csv(INPUT_FILE, sep=r"\s+", header=None, names=["x","y","z","ppfd"])
    print(f"Loaded {len(df)} PPFD data points from '{INPUT_FILE}'")
except Exception as e:
    print(f"Error reading '{INPUT_FILE}': {e}"); sys.exit(1)
if df.empty:
    print("Error: PPFD file is empty."); sys.exit(1)

x = df.x.values; y = df.y.values; zvals = df.ppfd.values

# ---- Grid reshape or interpolate -------------------------------------------
X=Y=Z=None
df_r = df.copy()
df_r["xr"] = df_r["x"].round(6)
df_r["yr"] = df_r["y"].round(6)
ptab = df_r.pivot_table(index="yr", columns="xr", values="ppfd", aggfunc="mean")
if (ptab.shape[0]*ptab.shape[1] == len(df)) and not ptab.isna().any().any():
    print("Data forms a regular grid (via pivot)")
    Xc = np.array(ptab.columns, dtype=float); Yc = np.array(ptab.index, dtype=float)
    X, Y = np.meshgrid(Xc, Yc); Z = ptab.values
else:
    print(f"Data is irregular ({len(df)} pts); interpolating to {GRID_SIZE}×{GRID_SIZE}")
    X, Y = np.meshgrid(np.linspace(x.min(), x.max(), GRID_SIZE),
                       np.linspace(y.min(), y.max(), GRID_SIZE))
    for method in ("cubic","linear","nearest"):
        try:
            Z = griddata((x,y), zvals, (X,Y), method=method); break
        except Exception:
            pass

if np.all(np.isnan(Z)):
    print("Warning: grid is all NaN after interpolation; plots may be empty.")

# ---- Metrics (raw Z) -------------------------------------------------------
mean_ppfd = np.nanmean(Z); std_ppfd = np.nanstd(Z)
if np.isfinite(mean_ppfd) and mean_ppfd != 0:
    dou = (1 - std_ppfd/mean_ppfd)*100; cv = (std_ppfd/mean_ppfd)*100
    print(f"PPFD (Grid Z) → mean {mean_ppfd:.1f}, std {std_ppfd:.1f}, DOU {dou:.1f}%, CV {cv:.1f}%")
else:
    print("PPFD (Grid Z) → mean NaN")

print(f"Color scale: vmin={VMIN:.1f}, vmax={VMAX:.1f}, cmap={CMAP}")
print("\nGenerating visualizations...")

# ---- Clipped for plotting only --------------------------------------------
Zc = np.clip(Z, VMIN, VMAX)

# ---- 3D surface ------------------------------------------------------------
try:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Zc, cmap=CMAP, edgecolor="none", antialiased=True)
    surf.set_clim(VMIN, VMAX)
    ax.set_zlim(VMIN, VMAX)
    fig.colorbar(surf, ax=ax, label="PPFD (µmol/m²/s)")
    ax.set_title(f"3D PPFD Surface (clamped to {VMIN:.0f}–{VMAX:.0f})")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("PPFD")
    fig.savefig(OUTDIR / "ppfd_surface_3d.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(" -> Saved ppfd_surface_3d.png")
except Exception as e:
    print("  ! 3D surface failed:", e)

# ---- Heatmap (annotated) ---------------------------------------------------
try:
    # Decide whether to annotate directly or on a downsampled grid
    annotate = bool(args.annot)
    Z_for_annot = Zc
    X_ticks = np.round(X[0, :], 2) if X is not None else None
    Y_ticks = np.round(Y[:, 0], 2) if Y is not None else None

    max_cells_for_annot = 35 * 35  # keep text readable
    H, W = Zc.shape
    if annotate and (H * W > max_cells_for_annot):
        # Downsample to ~35x35 (or smaller) for annotations
        fy = int(np.ceil(H / 35))
        fx = int(np.ceil(W / 35))
        Z_for_annot = Zc[::fy, ::fx]
        X_ticks = np.round(X[0, ::fx], 2)
        Y_ticks = np.round(Y[::fy, 0], 2)
        print(f"Annotated heatmap: downsampled from {H}x{W} to {Z_for_annot.shape[0]}x{Z_for_annot.shape[1]}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        Z_for_annot,
        vmin=VMIN, vmax=VMAX, cmap=CMAP,
        xticklabels=X_ticks, yticklabels=Y_ticks,
        annot=annotate, fmt=".0f",
        annot_kws={"size": 6} if annotate else None,
        linewidths=0.5, linecolor="gray",
        cbar_kws={"label": "PPFD (µmol/m²/s)"},
    )
    plt.title("PPFD Heatmap (annotated)" if annotate else "PPFD Heatmap")
    plt.xlabel("X (m)"); plt.ylabel("Y (m)")
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTDIR / "ppfd_heatmap_annotated.png", dpi=DPI)
    plt.close()
    print(" -> Saved ppfd_heatmap_annotated.png")
except Exception as e:
    print("  ! Annotated heatmap failed:", e)


# ---- Heatmap + overlays (bars drawn as rectangles) -------------------------
try:
    dx = np.mean(np.diff(X[0,:])) if X.shape[1]>1 else 0.1
    dy = np.mean(np.diff(Y[:,0])) if X.shape[0]>1 else 0.1
    xu_e = np.concatenate(([X[0,0]-dx/2], (X[0,:-1]+X[0,1:])/2, [X[0,-1]+dx/2]))
    yu_e = np.concatenate(([Y[0,0]-dy/2], (Y[:-1,0]+Y[1:,0])/2, [Y[-1,0]+dy/2]))

    fig, ax = plt.subplots(figsize=(8,6))
    pc = ax.pcolormesh(xu_e, yu_e, Zc, cmap=CMAP, edgecolors="gray",
                       linewidth=0.5, shading='auto', vmin=VMIN, vmax=VMAX)
    fig.colorbar(pc, label="PPFD (µmol/m²/s)")
    # Start with bounds from the sampled grid
    xmin, xmax = xu_e[0], xu_e[-1]
    ymin, ymax = yu_e[0], yu_e[-1]

    # polygons first (bars), then points (centers)
    for name, polys, kwargs in overlay_polys:
        for xy in polys:
            ax.add_patch(Polygon(xy, **kwargs))
            xs_p, ys_p = zip(*xy)
            xmin = min(xmin, min(xs_p)); xmax = max(xmax, max(xs_p))
            ymin = min(ymin, min(ys_p)); ymax = max(ymax, max(ys_p))
        ax.plot([], [], color='black', label=f"{name} ({len(polys)})")  # legend hook

    for name, pts, kwargs in overlay_points:
        xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
        ax.scatter(xs, ys, **kwargs, label=f"{name} ({len(pts)})")
        if xs and ys:
            xmin = min(xmin, min(xs)); xmax = max(xmax, max(xs))
            ymin = min(ymin, min(ys)); ymax = max(ymax, max(ys))

    # Prefer explicit room bounds if provided by overlay JSON; if aspect is swapped,
    # choose the orientation that best matches the PPFD grid spans.
    if overlay_room_bounds:
        grid_span_x = xmax - xmin
        grid_span_y = ymax - ymin
        L, W = overlay_room_bounds
        # Try both orientations and pick the closer match to the grid spans.
        cand1 = (L, W, abs(L - grid_span_x) + abs(W - grid_span_y))
        cand2 = (W, L, abs(W - grid_span_x) + abs(L - grid_span_y))
        best = min(cand1, cand2, key=lambda t: t[2])
        L_sel, W_sel = best[0], best[1]
        xmin, xmax = -L_sel * 0.5, L_sel * 0.5
        ymin, ymax = -W_sel * 0.5, W_sel * 0.5

    pad_x = max(0.0, 0.005 * (xmax - xmin))
    pad_y = max(0.0, 0.005 * (ymax - ymin))
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    if overlay_polys or overlay_points:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.20), ncol=2, framealpha=0.9)
    ax.set_title("PPFD Heatmap with Overlay", y=1.08)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_aspect('equal', adjustable='box'); fig.tight_layout()
    fig.savefig(OUTDIR/"ppfd_heatmap_overlay.png", dpi=DPI); plt.close(fig)
    print(" -> Saved ppfd_heatmap_overlay.png")
except Exception as e:
    print("  ! Heatmap with overlay failed:", e)

# ---- Histogram --------------------------------------------------------------
try:
    plt.figure(figsize=(8,6))
    vals = zvals[np.isfinite(zvals)]
    plt.hist(vals, bins=30, edgecolor="black")
    if np.isfinite(mean_ppfd): plt.axvline(mean_ppfd, color='red', linestyle='--', label=f"Mean {mean_ppfd:.1f}")
    plt.title("PPFD Distribution"); plt.xlabel("PPFD (µmol/m²/s)"); plt.ylabel("Frequency"); plt.legend()
    plt.tight_layout(); plt.savefig(OUTDIR/"ppfd_histogram.png", dpi=DPI); plt.close()
    print(" -> Saved ppfd_histogram.png")
except Exception as e:
    print("  ! Histogram failed:", e)

# ---- Deviation map ----------------------------------------------------------
try:
    if np.isfinite(mean_ppfd):
        Z_dev = np.abs(Z - mean_ppfd)
        plt.figure(figsize=(8,6))
        levels = 20
        cd = plt.contourf(X, Y, Z_dev, cmap="RdYlBu", levels=levels)
        plt.colorbar(cd, label="|PPFD – mean|")
        plt.title("PPFD Absolute Deviation from Mean")
        plt.xlabel("X (m)"); plt.ylabel("Y (m)")
        plt.gca().set_aspect('equal', adjustable='box'); plt.tight_layout()
        plt.savefig(OUTDIR/"ppfd_deviation.png", dpi=DPI); plt.close()
        print(" -> Saved ppfd_deviation.png")
except Exception as e:
    print("  ! Deviation plot failed:", e)

# ---- Interactive 3D scatter -------------------------------------------------
try:
    fig = go.Figure(data=[
        go.Scatter3d(
            x=df.x, y=df.y, z=df.ppfd, mode="markers",
            marker=dict(size=4, color=df.ppfd, colorscale=CMAP, cmin=VMIN, cmax=VMAX,
                        showscale=True, colorbar=dict(title="PPFD"))
        )
    ])
    for name, pts, _ in overlay_points:
        xs=[p[0] for p in pts]; ys=[p[1] for p in pts]; zs=[0]*len(pts)
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='markers',
                                   marker=dict(size=6, color='black', symbol='square-open'),
                                   name=name))
    fig.update_layout(title=f"Interactive 3D PPFD Scatter (clamped {VMIN:.0f}–{VMAX:.0f})",
                      scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="PPFD"),
                      width=900, height=650)
    fig.write_html(OUTDIR/"ppfd_scatter_3d.html"); print(" -> Saved ppfd_scatter_3d.html")
except Exception as e:
    print("  ! Interactive scatter failed:", e)

print(f"\nAll visualizations saved to '{OUTDIR}'")
