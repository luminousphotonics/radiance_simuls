#!/usr/bin/env python3
# optimize_ring_powers.py — robust search (random/LH), scale-invariant, ring0==ring1, hard-bounds, verified writes
import argparse, json, math, os, re, shutil, subprocess, sys, time, random
from pathlib import Path
from collections import OrderedDict
import numpy as np

HERE     = Path(__file__).resolve().parent
RUN_SH   = (HERE / "run_simulation_smd.sh").resolve()
GEN_FILE = (HERE / "generate_emitters_smd.py").resolve()
PPFD_TXT = (HERE / "ppfd_map.txt").resolve()
SUMMARY  = (HERE / "ies_sources" / "smd_summary.txt").resolve()
BACKUP   = (HERE / "generate_emitters_smd.py.bak").resolve()

# Defaults
INIT = OrderedDict({0:55.0,1:55.0,2:65.0,3:65.0,4:55.0,5:65.0,6:40.0,7:80.0})
BOUNDS = {k:(40.0,80.0) for k in range(8)}          # W/module hard limits
RING_COUNTS = [1,4,8,12,16,20,24,28]

# ---------- Files ----------
def read_text(p: Path) -> str: return p.read_text(encoding="utf-8")
def write_text(p: Path, s: str) -> None: p.write_text(s, encoding="utf-8")

# Regex supports typing.Dict[...] OR built-in dict[...]
PAT_RPW = re.compile(
    r"RING_POWER_W\s*:\s*(?:Dict|dict)\s*\[\s*int\s*,\s*float\s*\]\s*=\s*\{.*?\}",
    re.DOTALL
)

def patch_ring_powers(powers: OrderedDict):
    s = read_text(GEN_FILE)
    body = ", ".join(f"{k}: {float(v):.6f}" for k, v in powers.items())
    repl = f"RING_POWER_W: Dict[int, float] = {{{body}}}"
    if not PAT_RPW.search(s):
        pat2 = re.compile(r"RING_POWER_W\s*=\s*\{.*?\}", re.DOTALL)
        if not pat2.search(s):
            raise RuntimeError("RING_POWER_W dict not found in generate_emitters_smd.py")
        write_text(GEN_FILE, pat2.sub(f"RING_POWER_W = {{{body}}}", s, count=1))
    else:
        write_text(GEN_FILE, PAT_RPW.sub(repl, s, count=1))

def parse_summary_per_ring() -> list[float]:
    if not SUMMARY.exists():
        return []
    per = [None]*8
    for line in read_text(SUMMARY).splitlines():
        m = re.search(r"ring\s+(\d+):\s*([0-9.]+)\s*W\b", line)
        if m:
            i = int(m.group(1)); w = float(m.group(2))
            if 0 <= i < 8: per[i] = w
    return [float(x) if x is not None else float("nan") for x in per]

def run_pipeline(mode="instant", optics="none", subpatch=3, osamp=2, py="python3", verbose=False) -> float:
    if PPFD_TXT.exists(): PPFD_TXT.unlink()
    env = os.environ.copy()
    env.update({"MODE": mode, "OPTICS": optics, "SUBPATCH_GRID": str(subpatch), "OS": str(osamp), "PY": py})
    cmd = [str(RUN_SH)] if (RUN_SH.exists() and os.access(RUN_SH, os.X_OK)) else ["/bin/bash", str(RUN_SH)]
    t0 = time.time()
    out = None if verbose else subprocess.PIPE
    subprocess.run(cmd, cwd=str(HERE), env=env, stdout=out, stderr=subprocess.STDOUT, check=True)
    return time.time() - t0

def load_ppfd() -> tuple[float, float, float, float]:
    vals=[]
    with PPFD_TXT.open("r", encoding="utf-8") as f:
        for line in f:
            sp=line.split()
            if len(sp)>=4:
                try: v=float(sp[3])
                except: continue
                if math.isfinite(v): vals.append(v)
    if not vals: raise RuntimeError("No PPFD values found")
    arr=np.array(vals, float)
    mean=float(np.nanmean(arr)); std=float(np.nanstd(arr))
    cv=float(std/mean) if mean>0 else float("inf")
    dou=float((1.0-cv)*100.0) if math.isfinite(cv) else 0.0
    return mean,std,cv,dou

# ---------- Shapes (7 dof; ring0==ring1) ----------
def shape_from_x(x7: np.ndarray, ref_w: float, tau: float, wmin: float) -> np.ndarray:
    # groups = [(0&1), 2,3,4,5,6,7]  -> 7 weights
    z = np.asarray(x7, float) / max(tau, 1e-6)          # temperature softmax
    z -= np.max(z)
    w = np.exp(z); w /= np.sum(w)

    if wmin > 0.0:
        # project to simplex with lower bounds: w' = max(w, wmin); renormalize
        w = np.maximum(w, wmin)
        w /= np.sum(w)

    G = w * (7.0 * ref_w)                               # per-group Watts (unscaled)
    P = np.zeros(8, float)
    P[0] = P[1] = G[0]
    P[2], P[3], P[4], P[5], P[6], P[7] = G[1], G[2], G[3], G[4], G[5], G[6]
    return P

def x_from_init(init: OrderedDict[int,float], ref_w: float) -> np.ndarray:
    g = np.array([0.5*(init[0]+init[1]), init[2],init[3],init[4],init[5],init[6],init[7]], float)
    g = g/np.sum(g)
    z=np.log(np.maximum(g,1e-12))
    return z-z[-1]

def feasible_scale_interval(P_ref: np.ndarray) -> tuple[float,float]:
    s_lo=0.0; s_hi=float("inf")
    for i in range(8):
        if P_ref[i]<=0: continue
        lo,hi = BOUNDS[i]
        s_lo = max(s_lo, lo/P_ref[i])
        s_hi = min(s_hi, hi/P_ref[i])
    return s_lo, s_hi

# ---------- Evaluation ----------
def evaluate_shape(
    x7: np.ndarray,
    target_ppfd: float,
    mode: str,
    optics: str,
    subpatch: int,
    osamp: int,
    py: str,
    ref_w: float,
    tau: float,
    wmin: float,
    closest_scale: bool,
    verbose: bool=False
):
    # ratios -> per-ring (unscaled)
    P_ref = shape_from_x(x7, ref_w, tau=tau, wmin=wmin)
    # bounds for global scale
    s_lo, s_hi = feasible_scale_interval(P_ref)
    if not (np.isfinite(s_hi) and s_hi>0 and s_lo<s_hi):
        return dict(ok=False, reason="empty_bounds")

    # pick neutral scale to measure CV
    s0 = math.sqrt(s_lo*s_hi) if (s_lo>0 and np.isfinite(s_hi)) else max(1.0, s_lo)
    P_eval = P_ref*s0; P_eval[1]=P_eval[0]

    # enforce hard bounds at eval
    for i in range(8):
        lo,hi = BOUNDS[i]
        if P_eval[i]<lo-1e-6 or P_eval[i]>hi+1e-6:
            return dict(ok=False, reason="pre_run_oob")

    # Patch, run, verify
    patch_ring_powers(OrderedDict({i: float(P_eval[i]) for i in range(8)}))
    sim_t = run_pipeline(mode, optics, subpatch, osamp, py, verbose=verbose)

    seen = parse_summary_per_ring()
    if not seen or any(not math.isfinite(v) for v in seen):
        return dict(ok=False, reason="no_summary")
    for i in range(8):
        if abs(seen[i]-P_eval[i]) > 0.15:
            return dict(ok=False, reason=f"mismatch_ring_{i}", seen=seen, asked=list(P_eval))

    mean,std,cv,dou = load_ppfd()

    # exact scale to target (PPFD ∝ power)
    s_star = (target_ppfd / max(mean,1e-9)) * s0
    if not (s_star >= s_lo and s_star <= s_hi):
        if closest_scale:
            s_hat = min(max(s_star, s_lo), s_hi)
            P_hat = P_ref * s_hat; P_hat[1] = P_hat[0]
            return dict(ok=True, cv=cv, dou=dou, mean=mean, std=std,
                        s0=s0, s_star=s_hat, s_lo=s_lo, s_hi=s_hi,
                        sim_s=sim_t, P_target=P_hat, note="closest-scale", x7=x7)
        return dict(ok=False, reason="scale_to_target_oob",
                    rec=dict(mean=mean,std=std,cv=cv,dou=dou,s0=s0,s_lo=s_lo,s_hi=s_hi,s_star=s_star,sim_s=sim_t))

    # Success: exact scale feasible
    P_target = P_ref * s_star; P_target[1] = P_target[0]
    return dict(ok=True, cv=cv, dou=dou, mean=mean, std=std,
                s0=s0, s_star=s_star, s_lo=s_lo, s_hi=s_hi, sim_s=sim_t,
                P_target=P_target, x7=x7)

# ---------- Search ----------
def sample_x_dirichlet(n: int, alpha: float, rng: random.Random) -> np.ndarray:
    # 7-group Dirichlet -> logits x7
    X = []
    for _ in range(n):
        y = np.array([rng.gammavariate(alpha, 1.0) for _ in range(7)], float)
        y /= np.sum(y)
        z = np.log(np.maximum(y, 1e-12))
        X.append(z - z[-1])
    return np.vstack(X) if n>0 else np.zeros((0,7), float)

def main():
    ap = argparse.ArgumentParser(description="Ring optimizer (random/LH search; scale-invariant; ring0==ring1; bounds-safe).")
    ap.add_argument("--target-ppfd", type=float, required=True)
    ap.add_argument("--mode", default="instant", choices=["instant","fast","quality","direct"])
    ap.add_argument("--optics", default="none", choices=["none","lens"])
    ap.add_argument("--subpatch", type=int, default=3)
    ap.add_argument("--osamp", type=int, default=2)
    ap.add_argument("--py", default="python3")
    ap.add_argument("--ref-w", type=float, default=60.0)
    ap.add_argument("--samples", type=int, default=80, help="LH/random samples to try")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--init", type=str, default=None, help="JSON of initial per-ring powers (seed)")
    ap.add_argument("--init2", type=str, default=None, help="Second seed (optional)")
    ap.add_argument("--dirichlet-alpha", type=float, default=2.0)
    ap.add_argument("--tau", type=float, default=2.0, help="softmax temperature (>1 flattens)")
    ap.add_argument("--wmin", type=float, default=0.05, help="min group weight per group (0..1)")
    ap.add_argument("--closest-scale", action="store_true", help="if s* out of bounds, validate at nearest feasible scale")
    ap.add_argument("--polish", type=int, default=20, help="local random perturbation steps around best")

    args = ap.parse_args()

    if not RUN_SH.exists(): sys.exit(f"ERROR: {RUN_SH} not found.")
    if not GEN_FILE.exists(): sys.exit(f"ERROR: {GEN_FILE} not found.")

    rng = random.Random(args.seed)

    # backup
    if not BACKUP.exists():
        shutil.copy2(GEN_FILE, BACKUP)

    # seeds
    seeds_x = []
    init = OrderedDict(INIT)
    if args.init:
        j=json.loads(args.init); init=OrderedDict({int(k):float(v) for k,v in j.items()})
    init[1]=init[0]
    seeds_x.append(x_from_init(init, args.ref_w))
    if args.init2:
        j2=json.loads(args.init2); j2=OrderedDict({int(k):float(v) for k,v in j2.items()})
        j2[1]=j2[0]; seeds_x.append(x_from_init(j2, args.ref_w))

    # LH/random exploration
    X = sample_x_dirichlet(max(0, args.samples - len(seeds_x)), args.dirichlet_alpha, rng)
    if seeds_x:
        X = np.vstack([np.array(seeds_x, float), X])

    best = None  # (cv, info)
    try:
        print(f">>> search | target={args.target_ppfd} | samples={len(X)} | mode={args.mode} | bounds=40..80 W")

        for k, x in enumerate(X, 1):
            info = evaluate_shape(
                x7=x, target_ppfd=args.target_ppfd, mode=args.mode, optics=args.optics,
                subpatch=args.subpatch, osamp=args.osamp, py=args.py, ref_w=args.ref_w,
                tau=args.tau, wmin=args.wmin, closest_scale=args.closest_scale, verbose=False
            )

            if not info.get("ok", False):
                reason = info.get("reason","")
                rec = info.get("rec")
                if rec:
                    print(f"[{k:04d}] infeas: {reason:>20s} | mean={rec['mean']:.1f} CV={rec['cv']*100:.2f}% s*={rec['s_star']:.3f} in [{rec['s_lo']:.3f},{rec['s_hi']:.3f}]")
                else:
                    print(f"[{k:04d}] infeas: {reason:>20s}")
                continue

            cv = info["cv"]; dou=info["dou"]; mean=info["mean"]
            print(f"[{k:04d}]  OK  : CV={cv*100:.2f}% (DOU={100*(1-cv):.2f}%) | mean@eval={mean:.1f} | s*={info['s_star']:.3f} in [{info['s_lo']:.3f},{info['s_hi']:.3f}]")
            if (best is None) or (cv < best[0] - 1e-6):
                best = (cv, info)
                print(f"       ↳ new best (feasible)")

        # Optional local polish near the current best (small jitters in logit space)
        if best is not None and args.polish > 0:
            cv_best, info_best = best
            x_center = info_best.get("x7", None)
            if x_center is not None:
                for j in range(args.polish):
                    x_try = x_center + np.random.normal(scale=0.25, size=7)
                    info = evaluate_shape(
                        x7=x_try, target_ppfd=args.target_ppfd, mode=args.mode, optics=args.optics,
                        subpatch=args.subpatch, osamp=args.osamp, py=args.py, ref_w=args.ref_w,
                        tau=args.tau, wmin=args.wmin, closest_scale=args.closest_scale, verbose=False
                    )
                    if info.get("ok", False) and info["cv"] + 1e-6 < cv_best:
                        best = (info["cv"], info)
                        cv_best, info_best = best
                        print(f"[polish] improved CV -> {cv_best*100:.2f}%")

        if best is None:
            print("No feasible candidate hit the target within bounds. Try higher target, more samples, or widen bounds.")
            # still show baseline for reference
            patch_ring_powers(OrderedDict(INIT))
            sim_t = run_pipeline(args.mode, args.optics, args.subpatch, args.osamp, args.py, verbose=True)
            mean,std,cv,dou = load_ppfd()
            print(f"\nBaseline CV={cv*100:.2f}% @ mean={mean:.1f} | DOU={dou:.2f}% | sim={sim_t:.1f}s")
            return

        # Validate the winner at target (powers already at s* in result)
        P_final = best[1]["P_target"].copy()
        for i in range(8):
            lo,hi = BOUNDS[i]; P_final[i] = min(max(P_final[i], lo), hi)
        P_final[1]=P_final[0]

        patch_ring_powers(OrderedDict({i: float(P_final[i]) for i in range(8)}))
        sim_t = run_pipeline(args.mode, args.optics, args.subpatch, args.osamp, args.py, verbose=True)
        mean,std,cv,dou = load_ppfd()

        print("\n=== FINAL (best feasible @ target, validated) ===")
        for i in range(8):
            print(f"  ring {i}: {P_final[i]:.2f}" + (" (tie)" if i==1 else ""))
        print(f"Validation mean={mean:.2f}, std={std:.2f}, CV={cv*100:.2f}%, DOU={100*(1-cv):.2f}%, sim={sim_t:.1f}s\n")
        print("JSON:")
        print(json.dumps({str(i): round(float(P_final[i]),3) for i in range(8)}, indent=2))

    finally:
        if BACKUP.exists():
            shutil.copy2(BACKUP, GEN_FILE)
        print("\n(generate_emitters_smd.py restored from backup)")

if __name__ == "__main__":
    main()
