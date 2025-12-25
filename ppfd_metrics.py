#!/usr/bin/env python3
"""ppfd_metrics.py

Centralized PPFD field metrics.

The old "uniformity" log line (std/CV/DOU) is useful, but it does *not*
capture the practical control problem you highlighted:

If the canopy has a maximum photosynthetic setpoint (a hard PPFD ceiling),
the operator must dim globally until the *peak* is at or below that ceiling.
What matters then is how much average PPFD (and therefore total usable photons)
is still delivered after that peak-capping dim.

This module exposes:
  - "cap / usable photons" metrics (default)
  - legacy std/CV/DOU metrics (optional)

All inputs are in µmol/m^2/s.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _as_1d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a.ravel()


def compute_ppfd_metrics(
    ppfd: np.ndarray,
    *,
    setpoint_ppfd: float | None = None,
    canopy_area_m2: float | None = None,
    total_input_watts: float | None = None,
    emitted_ppf_umol_s: float | None = None,
    legacy_metrics: bool = False,
    coverage_fracs: tuple[float, ...] = (0.90, 0.95),
) -> dict[str, Any]:
    """Compute metrics for a PPFD field.

    Args:
        ppfd: array of PPFD samples (µmol/m²/s).
        setpoint_ppfd: optional canopy PPFD ceiling/setpoint (µmol/m²/s).
            If provided, we compute the global dim factor needed so max<=setpoint,
            and report the average PPFD remaining under that constraint.
        legacy_metrics: if True, include std/CV/DOU/RMSE-like metrics.
        coverage_fracs: fractions of the setpoint to report coverage for, after
            applying the peak-capping dim factor.

    Returns:
        dict with fields:
          mean, min, max, p05, p50, p95
          peak_over_mean, min_over_mean, mean_over_peak
          (if total_input_watts is set)
            watts_in
          (if emitted_ppf_umol_s is set)
            ppf_emitted, capture_frac
          (if setpoint_ppfd is set)
            setpoint_ppfd, cap_scale, mean_at_cap, min_at_cap, p05_at_cap,
            utilization_at_cap (mean_at_cap/setpoint), dim_penalty,
            coverage_ge_{int(frac*100)} (fraction of points >= frac*setpoint)
          (if both setpoint_ppfd and total_input_watts are set, and canopy_area_m2 is set)
            watts_at_cap, deuc_elec (also aliased as deuc)
          (if setpoint_ppfd and canopy_area_m2 are set)
            ppf_ge_{int(frac*100)}_at_cap (PPF from points >= frac*setpoint, after peak-capping)
          (if setpoint_ppfd and canopy_area_m2 and total_input_watts are set)
            deuc_ge_{int(frac*100)}_at_cap (efficacy of those "in-spec" photons, µmol/J)
          (if legacy_metrics)
            legacy: {std, cv_percent, dou_percent, rmse, mad, ...}
    """

    p = _as_1d(ppfd)
    if p.size == 0:
        raise ValueError("ppfd array is empty")

    mean = float(np.mean(p))
    pmin = float(np.min(p))
    pmax = float(np.max(p))

    # Percentiles help show tail behavior (hotspots/corners) better than std alone.
    p05 = float(np.percentile(p, 5))
    p50 = float(np.percentile(p, 50))
    p95 = float(np.percentile(p, 95))

    eps = 1e-12
    peak_over_mean = float(pmax / max(eps, mean))
    min_over_mean = float(pmin / max(eps, mean))
    mean_over_peak = float(mean / max(eps, pmax))

    out: dict[str, Any] = {
        "mean": mean,
        "min": pmin,
        "max": pmax,
        "p05": p05,
        "p50": p50,
        "p95": p95,
        "peak_over_mean": peak_over_mean,
        "min_over_mean": min_over_mean,
        "mean_over_peak": mean_over_peak,
    }

    # Optional: convert plane-average PPFD to total PPF over the sampled footprint.
    # PPF_out (µmol/s) = mean PPFD (µmol/m²/s) × area (m²)
    if canopy_area_m2 is not None and float(canopy_area_m2) > 0:
        A = float(canopy_area_m2)
        out["ppf_out"] = mean * A

    if total_input_watts is not None:
        w_in = float(total_input_watts)
        if w_in > 0:
            out["watts_in"] = w_in

    if emitted_ppf_umol_s is not None:
        ppf_emit = float(emitted_ppf_umol_s)
        if ppf_emit > 0:
            out["ppf_emitted"] = ppf_emit
            if "ppf_out" in out:
                out["capture_frac"] = float(out["ppf_out"]) / ppf_emit

    if setpoint_ppfd is not None and float(setpoint_ppfd) > 0:
        s = float(setpoint_ppfd)
        # Global dim factor required so that the brightest point is at (or under) setpoint.
        cap_scale = min(1.0, s / max(eps, pmax))
        p_cap = p * cap_scale

        mean_at_cap = float(np.mean(p_cap))
        min_at_cap = float(np.min(p_cap))
        p05_at_cap = float(np.percentile(p_cap, 5))

        utilization_at_cap = float(mean_at_cap / s)  # fraction of setpoint you still deliver on average
        dim_penalty = float(1.0 - cap_scale)          # how much global dimming you had to give up to respect the ceiling

        out.update({
            "setpoint_ppfd": s,
            "cap_scale": cap_scale,
            "dim_penalty": dim_penalty,
            "mean_at_cap": mean_at_cap,
            "min_at_cap": min_at_cap,
            "p05_at_cap": p05_at_cap,
            "utilization_at_cap": utilization_at_cap,
        })

        if "ppf_out" in out:
            A = float(canopy_area_m2)  # safe: only set if >0
            out["ppf_at_cap"] = mean_at_cap * A
            if "watts_in" in out:
                w_at_cap = float(out["watts_in"]) * cap_scale
                out["watts_at_cap"] = w_at_cap
                # Delivered efficacy under cap (DEUC) should reflect the uniformity penalty:
                # we evaluate delivered photons after peak-capping, but keep the *original*
                # input watts in the denominator (otherwise cap_scale cancels).
                w_in = float(out["watts_in"])
                if w_in > 0:
                    out["deuc_elec"] = float(out["ppf_at_cap"]) / w_in
                    out["deuc"] = out["deuc_elec"]

        # Coverage after peak-capping (this is what the grower actually sees if they cap peaks).
        for frac in coverage_fracs:
            if frac <= 0:
                continue
            thr = frac * s
            mask = p_cap >= thr
            cov = float(np.mean(mask))
            out[f"coverage_ge_{int(round(frac * 100))}"] = cov
            if canopy_area_m2 is not None and float(canopy_area_m2) > 0:
                # PPF delivered from points that are "in spec" (>= frac*setpoint),
                # assuming equal-area samples.
                A = float(canopy_area_m2)
                if cov > 0:
                    mean_in_spec = float(np.mean(p_cap[mask]))
                    ppf_in_spec = mean_in_spec * (A * cov)
                else:
                    ppf_in_spec = 0.0
                out[f"ppf_ge_{int(round(frac * 100))}_at_cap"] = ppf_in_spec
                if "watts_at_cap" in out:
                    w_at_cap = float(out["watts_at_cap"])
                    if w_at_cap > 0:
                        out[f"deuc_ge_{int(round(frac * 100))}_at_cap"] = ppf_in_spec / w_at_cap

    if legacy_metrics:
        std = float(np.std(p, ddof=0))
        cv = float(100.0 * std / max(eps, mean))
        # With your definition, RMSE about the mean equals std.
        rmse = std
        mad = float(np.mean(np.abs(p - mean)))
        dou = float(100.0 * (1.0 - rmse / max(eps, mean)))
        out["legacy"] = {
            "std": std,
            "cv_percent": cv,
            "dou_percent": dou,
            "rmse": rmse,
            "mad": mad,
            "min_over_avg": float(pmin / max(eps, mean)),
            "min_over_max": float(pmin / max(eps, pmax)),
        }

    return out


def format_ppfd_metrics_line(m: dict[str, Any]) -> str:
    """Human-readable multi-line block for logs."""

    lines: list[str] = []
    lines.append(
        "stats: "
        f"mean={m['mean']:.2f} min={m['min']:.2f} max={m['max']:.2f} "
        f"p05={m['p05']:.2f} p95={m['p95']:.2f}"
    )
    lines.append(
        "ratios: "
        f"peak/mean={m['peak_over_mean']:.3f} min/mean={m['min_over_mean']:.3f}"
    )

    if "ppf_out" in m:
        lines.append(f"ppf: out={m['ppf_out']:.1f} umol/s")

    if "setpoint_ppfd" in m:
        s = float(m["setpoint_ppfd"])
        cap_parts = [
            f"cap={s:.0f}",
            f"cap_scale={m['cap_scale']:.3f}",
            f"mean@cap={m['mean_at_cap']:.2f}",
            f"util@cap={100.0*m['utilization_at_cap']:.1f}%",
        ]
        if "ppf_at_cap" in m:
            cap_parts.append(f"ppf@cap={m['ppf_at_cap']:.1f} umol/s")
        if "deuc_elec" in m:
            cap_parts.append(f"DEUC_elec={m['deuc_elec']:.3f} umol/J")
        elif "deuc" in m:
            cap_parts.append(f"DEUC_elec={m['deuc']:.3f} umol/J")
        for k in ("coverage_ge_90", "coverage_ge_95"):
            if k in m:
                cap_parts.append(f"{k.replace('coverage_ge_', 'cov>=')}={100.0*m[k]:.1f}%")
        lines.append("cap: " + " ".join(cap_parts))

    # Legacy is optional; only show if caller asked for it.
    if isinstance(m.get("legacy"), dict):
        L = m["legacy"]
        lines.append(
            "legacy: "
            f"std={L['std']:.2f} CV={L['cv_percent']:.2f}% DOU={L['dou_percent']:.2f}%"
        )

    return "\n".join(lines)
