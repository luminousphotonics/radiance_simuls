#!/usr/bin/env python3
# solve_uniformity.py
# Compute per-ring powers w for target PPFD using basis_A and a smoothed LSQ.

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear, linprog

from ppfd_metrics import compute_ppfd_metrics, format_ppfd_metrics_line


def parse_args():
    ap = argparse.ArgumentParser(
        description="Ring-wise Radiance photonic density uniformity solver"
    )
    ap.add_argument("--basis", default="basis_A.npy",
                    help="Basis matrix file (npy or csv)")
    ap.add_argument("--target-ppfd", type=float, default=1200.0,
                    help="Target mean PPFD (µmol/m²/s)")
    ap.add_argument("--w-min", type=float, default=10.0,
                    help="Lower bound on per-ring power scale (W per module)")
    ap.add_argument("--w-max", type=float, default=100.0,
                    help="Upper bound on per-ring power scale (W per module)")
    ap.add_argument("--lambda-s", type=float, nargs="+", default=[0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 30.0],
                    help="Smoothing weights to try (L2 on ring differences)")
    ap.add_argument("--lambda-r", type=float, nargs="+", default=[0.0, 1e-3, 1e-2, 1e-1],
                    help="Ridge weights to try (L2 on absolute ring powers)")
    ap.add_argument("--ridge-weights", type=float, nargs="*", default=None,
                    help="Optional per-variable ridge weights (len=K, or a single scalar to broadcast).")
    ap.add_argument("--lambda-mean", type=float, default=10.0,
                    help="Weight on enforcing the target mean PPFD")
    ap.add_argument("--smooth-groups", type=int, nargs="*", default=None,
                    help="Optional group sizes for smoothing (build D per group, e.g. '13 13' for 2 channels).")
    ap.add_argument("--use-chebyshev", action="store_true", default=False,
                    help="Also try a Chebyshev (minimax) solve to shrink max error")
    ap.add_argument("--tol-mean", type=float, default=0.005,
                    help="Relative tolerance on mean PPFD after rescale (0.005 = 0.5%)")
    ap.add_argument("--out-json", default="ring_powers_optimized.json",
                    help="Output JSON with ring powers and metrics")
    ap.add_argument("--legacy-metrics", action="store_true", default=False,
                    help="Also compute/log legacy std/CV/DOU metrics")
    return ap.parse_args()


def load_basis(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    # assume csv/tsv
    return np.loadtxt(path, delimiter=",")


def load_basis_manifest(basis_path: Path) -> dict | None:
    # Prefer explicit BASIS_MANIFEST if set; otherwise look next to basis file.
    manifest_env = os.environ.get("BASIS_MANIFEST", "").strip()
    if manifest_env:
        p = Path(manifest_env)
    else:
        p = basis_path.parent / "basis_manifest.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def build_D(K: int) -> np.ndarray:
    """Finite difference matrix for ring-to-ring smoothing: (K-1)×K."""
    D = np.zeros((K - 1, K), dtype=float)
    for i in range(K - 1):
        D[i, i] = 1.0
        D[i, i + 1] = -1.0
    return D


def build_D_groups(groups: list[int], K: int) -> np.ndarray:
    """Block-diagonal D built per group (each group uses first differences)."""
    if not groups:
        return build_D(K)
    if any(int(g) < 0 for g in groups):
        raise ValueError("smooth group sizes must be non-negative")
    if sum(groups) != K:
        raise ValueError(f"smooth groups sum={sum(groups)} must equal K={K}")

    n_rows = sum(max(0, int(g) - 1) for g in groups)
    D = np.zeros((n_rows, K), dtype=float)
    row0 = 0
    col0 = 0
    for g in groups:
        g = int(g)
        if g >= 2:
            for i in range(g - 1):
                D[row0 + i, col0 + i] = 1.0
                D[row0 + i, col0 + i + 1] = -1.0
            row0 += g - 1
        col0 += g
    return D


def metrics_from_field(p: np.ndarray, *, setpoint_ppfd: float | None = None, legacy_metrics: bool = False) -> dict:
    return compute_ppfd_metrics(p, setpoint_ppfd=setpoint_ppfd, legacy_metrics=legacy_metrics)


def main():
    args = parse_args()

    basis_path = Path(args.basis)
    if not basis_path.exists():
        raise SystemExit(f"Basis file not found: {basis_path}")

    A = load_basis(basis_path)
    n_points, K = A.shape
    print(f"Loaded A with shape (n_points={n_points}, n_rings={K})")
    manifest = load_basis_manifest(basis_path)
    var_mode = None
    ring_group = None
    outer_indices = []
    outer_ring_index = None
    ring_indices = None
    if isinstance(manifest, dict):
        var_mode = manifest.get("variables")
        ring_indices = manifest.get("ring_indices")
        outer_indices = list(manifest.get("outer_ring_indices") or [])
        outer_ring_index = manifest.get("outer_ring_index")
        groups = manifest.get("variable_groups") or {}
        try:
            ring_group = int(groups.get("rings")) if groups.get("rings") is not None else None
        except Exception:
            ring_group = None
    if var_mode == "ring_plus_outer_modules" and ring_group is not None:
        if ring_group + len(outer_indices) != K:
            var_mode = None
            ring_group = None

    target_mu = float(args.target_ppfd)
    w_min = float(args.w_min)
    w_max = float(args.w_max)

    lambdas_s = [float(l) for l in args.lambda_s]
    lambdas_r = [float(l) for l in args.lambda_r]
    lambda_mean = float(args.lambda_mean)
    print("Trying lambda_s values:", lambdas_s)
    print("Trying lambda_r values:", lambdas_r)
    print(f"Mean weight lambda_mean={lambda_mean}")

    # Optional per-variable ridge weights (defaults to 1.0 for all).
    ridge_w = None
    if args.ridge_weights is None or len(args.ridge_weights) == 0:
        ridge_w = np.ones((K,), dtype=float)
    elif len(args.ridge_weights) == 1:
        ridge_w = np.full((K,), float(args.ridge_weights[0]), dtype=float)
    else:
        if len(args.ridge_weights) != K:
            raise SystemExit(f"--ridge-weights length {len(args.ridge_weights)} must match K={K}")
        ridge_w = np.asarray([float(x) for x in args.ridge_weights], dtype=float)

    # Optional smoothing groups (defaults to full vector).
    smooth_groups = None
    if args.smooth_groups is not None and len(args.smooth_groups) > 0:
        smooth_groups = [int(x) for x in args.smooth_groups]

    best = None  # (score, lambda_s, w, metrics)

    # Precompute ones vector
    ones = np.ones((n_points,), dtype=float)

    # Precompute mean row to anchor the overall average PPFD
    mean_row = A.mean(axis=0, keepdims=True)  # shape (1, K)

    for lam_s in lambdas_s:
        for lam_r in lambdas_r:
            print(f"\n--- lambda_s = {lam_s}, lambda_r = {lam_r} ---")

            aug_rows = [A]
            aug_rhs = [target_mu * ones]

            if lam_s > 0.0:
                D = build_D_groups(smooth_groups, K) if smooth_groups is not None else build_D(K)
                aug_rows.append(np.sqrt(lam_s) * D)
                aug_rhs.append(np.zeros((D.shape[0],), dtype=float))

            if lam_r > 0.0:
                # Weighted ridge: penalize some variables more than others.
                W = np.diag(np.sqrt(ridge_w))
                aug_rows.append(np.sqrt(lam_r) * W)
                aug_rhs.append(np.zeros((K,), dtype=float))

            if lambda_mean > 0.0:
                aug_rows.append(np.sqrt(lambda_mean) * mean_row)
                aug_rhs.append(np.array([np.sqrt(lambda_mean) * target_mu], dtype=float))

            A_aug = np.vstack(aug_rows)
            b_aug = np.concatenate(aug_rhs)

            res = lsq_linear(
                A_aug,
                b_aug,
                bounds=(w_min, w_max),
                method="trf",
            )


            if not res.success:
                print(f"  lsq_linear did not converge: {res.message}")
                continue

            w_raw = res.x
            p_raw = A @ w_raw
            m_raw = float(p_raw.mean())

            print(f"  raw mean={m_raw:.3f}, target={target_mu:.3f}")

            # Rescale to hit target mean (approximately), then clip to bounds
            if m_raw > 0:
                alpha = target_mu / m_raw
            else:
                alpha = 1.0

            w_scaled = np.clip(alpha * w_raw, w_min, w_max)
            p_scaled = A @ w_scaled
            m_scaled = float(p_scaled.mean())

            print(f"  scaled mean={m_scaled:.3f} (alpha={alpha:.4f})")

            # Compute metrics on scaled field
            m = metrics_from_field(p_scaled, setpoint_ppfd=target_mu, legacy_metrics=args.legacy_metrics)
            print(" ", format_ppfd_metrics_line(m))

            # Score: prioritize low CV; tie-break with smoother ring steps and higher Min/Avg
            if ring_group is not None and ring_group >= 2:
                diffs = np.diff(w_scaled[:ring_group])
            else:
                diffs = np.diff(w_scaled)
            smooth_pen = np.std(diffs) / max(1e-9, np.mean(w_scaled))
            legacy = m.get("legacy") or {}
            score = (float(legacy.get("cv_percent", float("inf"))), smooth_pen, -float(legacy.get("min_over_avg", 0.0)))

            if best is None or score < best[0]:
                best = (score, (lam_s, lam_r), w_scaled, m)

    # Optional Chebyshev (minimax) solve to minimize worst-case error
    if args.use_chebyshev:
        print("\n--- Chebyshev (minimax) solve ---")
        # Decision variables: w[0..K-1], t
        c = np.zeros(K + 1, dtype=float)
        c[-1] = 1.0  # minimize t

        A_ub = np.vstack([
            np.hstack([A, -np.ones((n_points, 1), dtype=float)]),
            np.hstack([-A, -np.ones((n_points, 1), dtype=float)]),
        ])
        b_ub = np.concatenate([
            target_mu * np.ones((n_points,), dtype=float),
            -target_mu * np.ones((n_points,), dtype=float),
        ])

        A_eq = np.hstack([mean_row, np.zeros((1, 1), dtype=float)])
        b_eq = np.array([target_mu], dtype=float)

        bounds = [(w_min, w_max)] * K + [(0.0, None)]

        res_lp = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )

        if res_lp.success:
            w_lp = res_lp.x[:K]
            p_lp = A @ w_lp
            m_lp = metrics_from_field(p_lp, setpoint_ppfd=target_mu, legacy_metrics=args.legacy_metrics)
            print(
                f"  minimax t={res_lp.fun:.3f}  "
                + format_ppfd_metrics_line(m_lp)
            )
            legacy_lp = m_lp.get("legacy") or {}
            score_lp = (float(legacy_lp.get("cv_percent", float("inf"))), -float(legacy_lp.get("min_over_avg", 0.0)))
            if best is None or score_lp < best[0]:
                best = (score_lp, ("chebyshev", None), w_lp, m_lp)
        else:
            print(f"  Chebyshev solve failed: {res_lp.message}")

    if best is None:
        raise SystemExit("No successful solution found.")

    score, lam_pair_best, w_best, m_best = best
    print("\n=== Best solution ===")
    if lam_pair_best[0] == "chebyshev":
        print("strategy=chebyshev (minimax)")
    else:
        print(f"lambda_s={lam_pair_best[0]}, lambda_r={lam_pair_best[1]}, lambda_mean={lambda_mean}")
    if var_mode == "ring_plus_outer_modules" and ring_group is not None:
        print("ring powers (W per module):")
        use_indices = ring_indices if isinstance(ring_indices, list) and len(ring_indices) == ring_group else list(range(ring_group))
        for i, wi in zip(use_indices, w_best[:ring_group]):
            print(f"  ring {int(i)}: {wi:.3f} W")
        print(f"outer ring modules: {len(w_best[ring_group:])} variables")
    else:
        print("ring powers (W per module):")
        for i, wi in enumerate(w_best):
            print(f"  ring {i}: {wi:.3f} W")

    print(format_ppfd_metrics_line(m_best))

    out = {
        "target_ppfd": target_mu,
        "strategy": "chebyshev" if lam_pair_best[0] == "chebyshev" else "lsq",
        "lambda_s_best": None if lam_pair_best[0] == "chebyshev" else lam_pair_best[0],
        "lambda_r_best": None if lam_pair_best[0] == "chebyshev" else lam_pair_best[1],
        "lambda_mean": lambda_mean,
        "metrics": m_best,
        "basis_file": str(basis_path),
        "n_points": int(n_points),
    }
    if var_mode == "ring_plus_outer_modules" and ring_group is not None:
        use_indices = ring_indices if isinstance(ring_indices, list) and len(ring_indices) == ring_group else list(range(ring_group))
        out.update({
            "variables": "ring_plus_outer_modules",
            "ring_indices": [int(x) for x in use_indices],
            "ring_powers_W_per_module": [float(x) for x in w_best[:ring_group]],
            "outer_ring_index": int(outer_ring_index) if outer_ring_index is not None else None,
            "outer_ring_indices": [int(x) for x in outer_indices],
            "outer_ring_powers_W_per_module": [float(x) for x in w_best[ring_group:]],
            "variable_groups": {"rings": int(ring_group), "outer_modules": int(len(w_best) - ring_group)},
        })
    else:
        out.update({
            "ring_indices": list(range(K)),
            "ring_powers_W_per_module": [float(x) for x in w_best],
        })
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"\nSaved {args.out_json}")


if __name__ == "__main__":
    main()
