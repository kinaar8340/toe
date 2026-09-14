"""Cheap invariants for the homolog_flywheel Model analog.

MODEL, not theorem.
Geometry and arithmetic stay on QGA / flux_hopf_lib.
The alkane / CH2 / carbene language is an analogy for a discrete insertion step.
n=1 is one flywheel at quaternion identity (methane slot).
n=2 is one extra published step: one extra flywheel XOR one extra rotor insertion.
Do not emit “proves”, “element”, “periodic table identity”, or “carbene is a flywheel”.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from homolog_flywheel.analog import AXIS_RULE
from homolog_flywheel.insert import q_conj, q_mult, q_normalize, unit_axis
from homolog_flywheel.seed import IDENTITY_Q
from homolog_flywheel.state import HomologState

CSV_COLUMNS = [
    "n",
    "alias",
    "step_mode",
    "axis_rule",
    "q_w",
    "q_x",
    "q_y",
    "q_z",
    "axis_x",
    "axis_y",
    "axis_z",
    "identity_overlap",
    "unit_norm_error",
    "step_geodesic_rad",
    "axis_drift_rad",
    "identity_preservation",
]


def quaternion_geodesic_rad(q_prev: np.ndarray, q_now: np.ndarray) -> float:
    """Rotation-angle geodesic of q_now · conj(q_prev) on the analog chain."""
    q_rel = q_mult(q_normalize(q_now), q_conj(q_normalize(q_prev)))
    w = float(np.clip(abs(float(q_rel[0])), 0.0, 1.0))
    return float(2.0 * np.arccos(w))


def axis_drift_rad(axis_prev: np.ndarray, axis_now: np.ndarray) -> float:
    """S² geodesic between successive insertion axes. analog: axis rule, not n as Z."""
    a = unit_axis(axis_prev)
    b = unit_axis(axis_now)
    return float(np.arccos(np.clip(float(np.dot(a, b)), -1.0, 1.0)))


def measure(
    state: HomologState,
    prev: HomologState | None = None,
    *,
    slow: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    """Return a flat dict suitable for CSV. Optional PDE/conduit only if slow=True."""
    q = np.asarray(state.q, dtype=float).reshape(4)
    identity_overlap = float(abs(np.dot(q, IDENTITY_Q)))
    unit_norm_error = float(abs(np.linalg.norm(q) - 1.0))
    if prev is None or state.n == 1:
        step_geodesic_rad = float("nan")
        drift = float("nan")
    else:
        step_geodesic_rad = quaternion_geodesic_rad(prev.q, q)
        drift = axis_drift_rad(prev.insertion_axis, state.insertion_axis)

    axis = unit_axis(state.insertion_axis)
    row: dict[str, Any] = {
        "n": int(state.n),
        "alias": state.alias,
        "step_mode": state.step_mode,
        "axis_rule": AXIS_RULE.get(state.step_mode, ""),
        "q_w": float(q[0]),
        "q_x": float(q[1]),
        "q_y": float(q[2]),
        "q_z": float(q[3]),
        "axis_x": float(axis[0]),
        "axis_y": float(axis[1]),
        "axis_z": float(axis[2]),
        "identity_overlap": identity_overlap,
        "unit_norm_error": unit_norm_error,
        "step_geodesic_rad": step_geodesic_rad,
        "axis_drift_rad": drift,
        # analog: z-map vocabulary, cheap overlap, not the 300-frame map
        "identity_preservation": identity_overlap,
        "Z": int(state.Z),
        "n_flywheels": len(state.flywheels),
    }
    if slow:
        row.update(_slow_invariants(seed=seed))
    return row


def _slow_invariants(*, seed: int) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    try:
        from flux_hopf_lib.simulation import simulate_twist_pde_survival

        result = simulate_twist_pde_survival(nx=20, seed=seed)
        extra["lt_survival"] = float(result["survival"]["mean_survival"])
    except Exception as exc:  # pragma: no cover - optional --slow path
        extra["lt_survival"] = float("nan")
        extra["lt_survival_note"] = f"analog: skipped ({type(exc).__name__})"
    try:
        from conduit import RubikConeConduit

        conduit = RubikConeConduit(embed_dim=384, num_polarizations=3)
        stats = conduit.monitor_topological_winding(n_samples=5, pol_ref=0)
        extra["geometric_winding"] = float(stats.get("geometric_winding", float("nan")))
    except Exception as exc:  # pragma: no cover
        extra["geometric_winding"] = float("nan")
        extra["winding_note"] = f"analog: skipped ({type(exc).__name__})"
    return extra


def hypothesis_verdict(rows: list[dict[str, Any]]) -> str:
    """Record whether this run is consistent with the insertion-stability hypothesis.

    Never phrased as a QGA refutation.
    """
    if not rows:
        return "hypothesis: not supported by this run"
    norms = [float(r["unit_norm_error"]) for r in rows]
    if any(n > 1e-6 for n in norms):
        return "hypothesis: not supported by this run"
    geos = [
        float(r["step_geodesic_rad"])
        for r in rows
        if r["n"] >= 2 and not math.isnan(float(r["step_geodesic_rad"]))
    ]
    if geos:
        g0 = geos[0]
        if g0 > 1e-12 and any(abs(g - g0) > 0.05 * max(g0, 1e-12) + 1e-6 for g in geos):
            return "hypothesis: not supported by this run"
    by_n = {int(r["n"]): float(r["identity_overlap"]) for r in rows}
    if 2 in by_n and by_n[2] < 1e-3:
        return "hypothesis: not supported by this run"
    return "hypothesis: consistent with this run (analog only; not a QGA result)"
