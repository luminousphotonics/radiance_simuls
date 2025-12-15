from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


def _finite_1d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float).ravel()
    return a[np.isfinite(a)]


def compute_ppfd_metrics(
    ppfd: np.ndarray,
    *,
    setpoint_ppfd: Optional[float] = None,
    legacy_metrics: bool = False,
) -> Dict[str, Any]:
    """
    Compute PPFD summary metrics.

    If `setpoint_ppfd` is provided and > 0, also compute "setpoint constrained usable photons" metrics.
    If `legacy_metrics` is True, include std/CV/DOU and min ratios under `legacy`.
    """
    v = _finite_1d(ppfd)
    out: Dict[str, Any] = {"n": int(v.size)}
    if v.size == 0:
        out.update(
            {
                "min": float("nan"),
                "max": float("nan"),
                "mean": float("nan"),
            }
        )
        if setpoint_ppfd is not None:
            out["setpoint_ppfd"] = float(setpoint_ppfd)
        if legacy_metrics:
            out["legacy"] = {
                "std": float("nan"),
                "cv_percent": float("nan"),
                "dou_percent": float("nan"),
                "min_over_avg": float("nan"),
                "min_over_max": float("nan"),
            }
        return out

    p_min = float(np.min(v))
    p_max = float(np.max(v))
    p_mean = float(np.mean(v))
    out.update({"min": p_min, "max": p_max, "mean": p_mean})

    S = None
    if setpoint_ppfd is not None:
        try:
            S = float(setpoint_ppfd)
        except Exception:
            S = None
        if S is not None and S > 0:
            capped = np.minimum(v, S)
            usable_mean = float(np.mean(capped))
            usable_fraction = float(usable_mean / S) if S > 0 else float("nan")
            waste_num = float(np.sum(np.maximum(v - S, 0.0)))
            delivered = float(np.sum(v))
            waste_fraction = float(waste_num / delivered) if delivered > 0 else float("nan")
            deficit_fraction = float(1.0 - usable_fraction) if math.isfinite(usable_fraction) else float("nan")

            global_scale = float(S / p_max) if p_max > 0 else 0.0
            mean_after_global_cap = float(p_mean * global_scale)
            avg_over_max = float(p_mean / p_max) if p_max > 0 else 0.0

            frac_above = float(np.mean(v > S))
            frac_near = float(np.mean((v >= 0.95 * S) & (v <= S)))

            out.update(
                {
                    "setpoint_ppfd": float(S),
                    "usable_mean": usable_mean,
                    "usable_fraction": usable_fraction,
                    "waste_fraction": waste_fraction,
                    "deficit_fraction": deficit_fraction,
                    "global_scale": global_scale,
                    "mean_after_global_cap": mean_after_global_cap,
                    "avg_over_max": avg_over_max,
                    "frac_above": frac_above,
                    "frac_near": frac_near,
                }
            )
        else:
            out["setpoint_ppfd"] = float(setpoint_ppfd)

    if legacy_metrics:
        std = float(np.std(v))
        cv = (std / p_mean) if p_mean > 0 else float("inf")
        out["legacy"] = {
            "std": std,
            "cv_percent": float(100.0 * cv) if math.isfinite(cv) else float("inf"),
            "dou_percent": float(100.0 * (1.0 - cv)) if (p_mean > 0 and math.isfinite(cv)) else 0.0,
            "min_over_avg": float(p_min / p_mean) if p_mean > 0 else 0.0,
            "min_over_max": float(p_min / p_max) if p_max > 0 else 0.0,
        }

    return out


def format_ppfd_metrics_line(metrics: Dict[str, Any], *, emphasize_setpoint: bool = True) -> str:
    """
    Format a single-line log string.

    If setpoint metrics exist, they are emphasized (unless emphasize_setpoint=False).
    """
    n = int(metrics.get("n", 0) or 0)
    mean = metrics.get("mean", float("nan"))
    pmin = metrics.get("min", float("nan"))
    pmax = metrics.get("max", float("nan"))

    parts = [f"mean={mean:.2f}", f"min={pmin:.2f}", f"max={pmax:.2f}", f"n={n}"]

    S = metrics.get("setpoint_ppfd", None)
    has_setpoint_metrics = emphasize_setpoint and all(
        k in metrics
        for k in (
            "usable_mean",
            "usable_fraction",
            "waste_fraction",
            "global_scale",
            "mean_after_global_cap",
            "frac_above",
        )
    )
    if has_setpoint_metrics:
        usable_mean = float(metrics["usable_mean"])
        usable_fraction = float(metrics["usable_fraction"])
        waste_fraction = float(metrics["waste_fraction"])
        global_scale = float(metrics["global_scale"])
        mean_after_global_cap = float(metrics["mean_after_global_cap"])
        frac_above = float(metrics["frac_above"])
        frac_near = metrics.get("frac_near", None)

        parts.append("|")
        parts.append(f"S={float(S):.2f}")
        parts.append(f"usable_mean={usable_mean:.2f}")
        parts.append(f"usable={usable_fraction*100.0:.2f}%")
        parts.append(f"waste={waste_fraction*100.0:.2f}%")
        parts.append(f"global_cap_mean={mean_after_global_cap:.2f}")
        parts.append(f"(scale={global_scale:.6f})")
        parts.append(f"frac_above={frac_above*100.0:.2f}%")
        if frac_near is not None:
            parts.append(f"frac_near={float(frac_near)*100.0:.2f}%")

    legacy = metrics.get("legacy")
    if legacy:
        std = legacy.get("std", float("nan"))
        cvp = legacy.get("cv_percent", float("nan"))
        doup = legacy.get("dou_percent", float("nan"))
        parts.append(f"| std={float(std):.2f} CV={float(cvp):.2f}% DOU={float(doup):.2f}%")

    return " ".join(parts)

