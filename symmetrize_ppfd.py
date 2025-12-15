#!/usr/bin/env python3
# symmetrize_ppfd.py — enforce D4 symmetry on ppfd_map.txt with optional blend
import argparse, math
from pathlib import Path
import numpy as np

def load_ppfd(path: Path):
    rows=[]
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            sp=ln.split()
            if len(sp) < 4: continue
            try:
                x=float(sp[0]); y=float(sp[1]); z=float(sp[2]); p=float(sp[3])
            except: continue
            rows.append((x,y,z,p,ln))  # keep original line for order
    if not rows: raise SystemExit("ERROR: no PPFD rows found")
    return rows

def grid_from_rows(rows, tol=1e-6):
    xs=sorted({r[0] for r in rows})
    ys=sorted({r[1] for r in rows})
    xi={round(x/tol)*tol:i for i,x in enumerate(xs)}
    yi={round(y/tol)*tol:i for i,y in enumerate(ys)}
    nx,ny=len(xs),len(ys)
    Z=np.full((ny,nx), np.nan, float)
    idxmap={}
    for (x,y,_,p,_) in rows:
        i=xi[round(x/tol)*tol]; j=yi[round(y/tol)*tol]
        Z[j,i]=p
        idxmap[(xs[i],ys[j])] = (i,j)
    return xs,ys,Z,idxmap

def d4_orbit(i,j,xs,ys,idxmap,tol=1e-6):
    # Generate all 8 sym equivalents for (x,y)
    x=xs[i]; y=ys[j]
    pts=[( x,  y),( x,-y),(-x, y),(-x,-y),
         ( y,  x),( y,-x),(-y, x),(-y,-x)]
    seen=set(); out=[]
    for (xx,yy) in pts:
        k=(round(xx/tol)*tol, round(yy/tol)*tol)
        if k in idxmap:
            ij=idxmap[k]
            if ij not in seen:
                seen.add(ij); out.append(ij)
    return out

def cv_dou(Z):
    v=Z[np.isfinite(Z)]
    m=float(np.mean(v)); s=float(np.std(v)); cv=s/m if m>0 else math.inf
    return m,s,cv,100*(1-cv)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",  default="ppfd_map.txt")
    ap.add_argument("--output", default=None, help="default: overwrite input")
    ap.add_argument("--lam", type=float, default=1.0, help="blend: 1.0=full sym, 0=off")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--axes-only", action="store_true", help="only x/y flips (no 90° rotations)")
    ap.add_argument("--verbose", action="store_true")
    args=ap.parse_args()

    rows = load_ppfd(Path(args.input))
    xs,ys,Z,idxmap = grid_from_rows(rows, tol=args.tol)
    Z0 = Z.copy()
    m0,s0,cv0,dou0 = cv_dou(Z0)

    # Build symmetric-averaged field
    Zs = Z.copy()
    visited=set()
    ny,nx = Z.shape
    for j in range(ny):
        for i in range(nx):
            if (i,j) in visited: continue
            orbit = d4_orbit(i,j,xs,ys,idxmap,tol=args.tol)
            # axes-only mode → keep just (x,y),(x,-y),(-x,y),(-x,-y)
            if args.axes_only:
                x=xs[i]; y=ys[j]
                want={( x, y),( x,-y),(-x, y),(-x,-y)}
                orbit=[idxmap[(xx,yy)] for (xx,yy) in want if (xx,yy) in idxmap]
            vals=[Z[o[1],o[0]] for o in orbit if math.isfinite(Z[o[1],o[0]])]
            if not vals: continue
            avg=float(np.mean(vals))
            for (ii,jj) in orbit:
                visited.add((ii,jj))
                if math.isfinite(Zs[jj,ii]):
                    Zs[jj,ii] = args.lam*avg + (1-args.lam)*Zs[jj,ii]

    m1,s1,cv1,dou1 = cv_dou(Zs)
    if args.verbose:
        print(f"Before: mean={m0:.2f} std={s0:.2f} CV={cv0*100:.2f}% DOU={dou0:.2f}%")
        print(f"After : mean={m1:.2f} std={s1:.2f} CV={cv1*100:.2f}% DOU={dou1:.2f}%")

    # Write, preserving original order & XYZ
    outp = Path(args.output) if args.output else Path(args.input)
    # map (x,y) -> ppfd (sym)
    sym_map = {(xs[i],ys[j]): Zs[j,i] for j in range(len(ys)) for i in range(len(xs))}
    with outp.open("w", encoding="utf-8") as f:
        for (x,y,z,_,ln) in rows:
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {sym_map[(x,y)]:.6f}\n")

if __name__ == "__main__":
    main()
