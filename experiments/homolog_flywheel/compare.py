"""Fixed n-max mode comparison for the homolog_flywheel Model analog.

MODEL, not theorem.
Geometry and arithmetic stay on QGA / flux_hopf_lib.
The alkane / CH2 / carbene language is an analogy for a discrete insertion step.
n=1 is one flywheel at quaternion identity (methane slot).
n=2 is one extra published step: one extra flywheel XOR one extra rotor insertion.
Do not emit “proves”, “element”, “periodic table identity”, or “carbene is a flywheel”.

Compares rotor vs flywheel vs published at the same n-max by axis rule:
geodesic step, axis drift, identity overlap. Aliases stay display labels.
n is not Z.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from homolog_flywheel.analog import FROZEN_Z, NOTES_PREFIX, STEP_MODES
from homolog_flywheel.insert import (
    BAKE_X,
    axis_for_mode,
    insert,
    q_mult,
    q_normalize,
    rodrigues,
    small_rotor,
    unit_axis,
)
from homolog_flywheel.measure import axis_drift_rad, measure
from homolog_flywheel.seed import IDENTITY_Q, identity_seed
from homolog_flywheel.state import HomologState

# analog: CLI axis with an x-component, so published walk leaves the yz-plane.
OFF_YZ_AXIS = np.array([1.0, 0.0, 1.0], dtype=float)

COMPARE_COLUMNS = [
    "n",
    "alias",
    "step_mode",
    "axis_rule",
    "axis_x",
    "axis_y",
    "axis_z",
    "axis_drift_rad",
    "walk_phase_rad",
    "step_geodesic_rad",
    "identity_overlap",
    "Z",
]

SUMMARY_COLUMNS = [
    "step_mode",
    "axis_rule",
    "n_max",
    "alias",
    "axis_drift_mean",
    "walk_phase_mean",
    "step_geodesic_mean",
    "identity_overlap_at_nmax",
    "axis_x",
    "axis_y",
    "axis_z",
    "Z",
]


def run_chain(
    n_max: int,
    step_mode: str,
    angle_rad: float,
    axis: np.ndarray,
    *,
    slow: bool = False,
    seed: int = 0,
    Z: int = FROZEN_Z,
) -> list[HomologState]:
    states = [
        identity_seed(
            step_mode=step_mode,
            insertion_axis=axis,
            insertion_angle_rad=angle_rad,
            Z=Z,
        )
    ]
    for _ in range(1, n_max):
        states.append(
            insert(
                states[-1],
                step_mode=step_mode,
                angle_rad=angle_rad,
                axis=axis,
            )
        )
    for i, state in enumerate(states):
        prev = states[i - 1] if i else None
        state.invariants = measure(state, prev, slow=slow, seed=seed)
    return states


def _finite_mean(values: list[float]) -> float:
    xs = [float(v) for v in values if not math.isnan(float(v))]
    if not xs:
        return float("nan")
    return float(np.mean(xs))


def summarize_mode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    last = rows[-1]
    return {
        "step_mode": last["step_mode"],
        "axis_rule": last["axis_rule"],
        "n_max": int(last["n"]),
        "alias": last["alias"],
        "axis_drift_mean": _finite_mean([float(r["axis_drift_rad"]) for r in rows]),
        "walk_phase_mean": _finite_mean([float(r["walk_phase_rad"]) for r in rows]),
        "step_geodesic_mean": _finite_mean([float(r["step_geodesic_rad"]) for r in rows]),
        "identity_overlap_at_nmax": float(last["identity_overlap"]),
        "axis_x": float(last["axis_x"]),
        "axis_y": float(last["axis_y"]),
        "axis_z": float(last["axis_z"]),
        "Z": int(last["Z"]),
    }


def mode_comparison(
    n_max: int,
    angle_rad: float,
    axis: np.ndarray,
    *,
    slow: bool = False,
    seed: int = 0,
    Z: int = FROZEN_Z,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[HomologState]]]:
    """Run all three modes at the same n-max. Z stays frozen; n is chain index."""
    chains: dict[str, list[HomologState]] = {}
    long_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for mode in STEP_MODES:
        states = run_chain(
            n_max,
            mode,
            angle_rad,
            axis,
            slow=slow,
            seed=seed,
            Z=Z,
        )
        chains[mode] = states
        rows = [s.invariants for s in states]
        long_rows.extend(rows)
        summaries.append(summarize_mode(rows))
    return long_rows, summaries, chains


def comparison_verdict(long_rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> str:
    """Axis-rule check. Never a QGA result. Aliases are labels only."""
    if not long_rows or not summaries:
        return "hypothesis: not supported by this run"
    if any(int(r["Z"]) != FROZEN_Z for r in long_rows):
        return "hypothesis: not supported by this run"

    by_n: dict[int, set[str]] = {}
    for row in long_rows:
        by_n.setdefault(int(row["n"]), set()).add(str(row["alias"]))
    if any(len(aliases) != 1 for aliases in by_n.values()):
        return "hypothesis: not supported by this run"

    rules = {str(s["axis_rule"]) for s in summaries}
    axes = {
        (round(float(s["axis_x"]), 6), round(float(s["axis_y"]), 6), round(float(s["axis_z"]), 6))
        for s in summaries
    }
    if len(rules) < 3 or len(axes) < 2:
        return "hypothesis: not supported by this run"
    return (
        f"{NOTES_PREFIX} mode comparison: aliases match at each n; "
        "modes differ by axis rule (not by alkane labels); Z frozen, n is not Z."
    )


def rotor_single_axis_overlap(n: int, angle_rad: float) -> float:
    """|cos((n-1)θ/2)|. analog: fixed-axis subgroup, not an alkane invariant."""
    return float(abs(math.cos((n - 1) * angle_rad / 2.0)))


def expected_s2_drift(cli_axis: np.ndarray, angle_rad: float) -> float:
    """S² chord of one Rodrigues step of CLI about bake-x. Equals θ iff CLI ⟂ x."""
    v = unit_axis(cli_axis)
    w = unit_axis(rodrigues(v, BAKE_X, float(angle_rad)))
    return axis_drift_rad(v, w)


def compose_published_qs(
    n_max: int,
    angle_rad: float,
    cli_axis: np.ndarray,
    *,
    side: str = "left",
) -> list[np.ndarray]:
    """Independent product of walking-axis rotors. Does not use aliases or Z."""
    q = IDENTITY_Q.copy()
    out = [q.copy()]
    cli = unit_axis(cli_axis)
    for step_index in range(1, n_max):
        ax = axis_for_mode(
            "published",
            step_index=step_index,
            cli_axis=cli,
            angle_rad=angle_rad,
        )
        rotor = small_rotor(angle_rad, ax)
        q = q_normalize(q_mult(rotor, q) if side == "left" else q_mult(q, rotor))
        out.append(np.asarray(q, dtype=float).copy())
    return out


def published_axis_probe(
    n_max: int,
    angle_rad: float,
    axis_a: np.ndarray,
    axis_b: np.ndarray,
    *,
    seed: int = 0,
    Z: int = FROZEN_Z,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Two published chains at the same n-max; only CLI axis differs."""
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for axis in (axis_a, axis_b):
        states = run_chain(
            n_max,
            "published",
            angle_rad,
            unit_axis(axis),
            seed=seed,
            Z=Z,
        )
        chain_rows = [s.invariants for s in states]
        for row in chain_rows:
            row["cli_axis"] = ",".join(f"{x:.6f}" for x in unit_axis(axis))
        rows.extend(chain_rows)
        summaries.append(summarize_mode(chain_rows))
        summaries[-1]["cli_axis"] = ",".join(f"{x:.6f}" for x in unit_axis(axis))
    return rows, summaries
