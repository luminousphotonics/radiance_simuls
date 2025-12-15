#!/usr/bin/env python3
"""
combine_emitters_to_umol.py  — safe replacer
Converts per-channel RAD files from W/sr/m^2 to µmol/sr/m^2/s by dividing each
'3 R G B' line by E_channel (J/µmol), as defined in ppfd_constants.env.
"""
from __future__ import annotations
import re, sys
from pathlib import Path

ROOT = Path("ies_sources")
ENV  = ROOT / "ppfd_constants.env"
CHANNELS = ["WW","CW","R","B","FR","C","UV"]

def parse_env(p: Path) -> dict[str,float]:
    vals={}
    for ln in p.read_text().splitlines():
        ln=ln.strip()
        if not ln or ln.startswith("#") or "=" not in ln: continue
        k,v = ln.split("=",1)
        try: vals[k.strip()] = float(v.strip())
        except: pass
    return vals

def scale_text(ch: str, E: float, txt: str) -> str:
    out=[]
    i=0
    lines = txt.splitlines()
    while i < len(lines):
        L = lines[i]
        out.append(L); i += 1
        if L.startswith("void light "):
            # Expect: next two lines "0" and "0", then a '3 r g b' line
            if i+2 <= len(lines):
                out.append(lines[i]); out.append(lines[i+1]); i += 2
            if i < len(lines):
                m = re.match(r"\s*3\s+([+\-0-9.eE]+)\s+([+\-0-9.eE]+)\s+([+\-0-9.eE]+)\s*$", lines[i])
                if not m:
                    out.append(lines[i]); i += 1
                else:
                    r,g,b = (float(m.group(k)) for k in (1,2,3))
                    s = 1.0/E
                    out.append(f"3 {r*s:.6f} {g*s:.6f} {b*s:.6f}")
                    i += 1
    return "\n".join(out) + "\n"

def main():
    if not ENV.exists():
        sys.exit("ERROR: ies_sources/ppfd_constants.env not found (run generate_emitters_smd.py first).")
    consts = parse_env(ENV)

    pieces=[]
    for ch in CHANNELS:
        f = ROOT / f"emitters_smd_{ch}.rad"
        if not f.exists(): continue
        E = consts.get(f"E_{ch}", 0.0)
        if E <= 0:
            sys.exit(f"ERROR: E_{ch} missing/invalid in {ENV}")
        pieces.append(f"# --- BEGIN {ch} (÷ {E:.6g} J/µmol) ---\n")
        pieces.append(scale_text(ch, E, f.read_text()))
        pieces.append(f"# --- END {ch} ---\n")
    if not pieces:
        sys.exit("ERROR: no per-channel emitter files found in ies_sources/")

    out = ROOT / "emitters_smd_ALL_umol.rad"
    out.write_text(
        "# Combined SMD emitters (pre-scaled to µmol units)\n"
        "# Each '3 R G B' replaced in-place; units are now µmol/sr/m^2/s per channel.\n\n"
        + "".join(pieces)
    )
    print(f"✔ Wrote {out}")

if __name__ == "__main__":
    main()
