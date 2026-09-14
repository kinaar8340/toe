"""n=1 identity flywheel seed (methane-slot analog).

MODEL, not theorem.
Geometry and arithmetic stay on QGA / flux_hopf_lib.
The alkane / CH2 / carbene language is an analogy for a discrete insertion step.
n=1 is one flywheel at quaternion identity (methane slot).
n=2 is one extra published step: one extra flywheel XOR one extra rotor insertion.
Do not emit “proves”, “element”, “periodic table identity”, or “carbene is a flywheel”.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from homolog_flywheel.analog import (
    AXIS_RULE,
    DEFAULT_ALIAS_FAMILY,
    DEFAULT_STEP_MODE,
    FROZEN_Z,
    NOTES_PREFIX,
    alias_for,
)
from homolog_flywheel.insert import DEFAULT_ANGLE_RAD, axis_for_mode, make_flywheel, unit_axis
from homolog_flywheel.state import HomologState

Array = NDArray[np.floating]

IDENTITY_Q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


def identity_seed(
    *,
    step_mode: str = DEFAULT_STEP_MODE,
    insertion_axis: Array | None = None,
    insertion_angle_rad: float | None = None,
    Z: int = FROZEN_Z,
    alias_family: str = DEFAULT_ALIAS_FAMILY,
    group_id: str = "",
) -> HomologState:
    """n=1: one flywheel at quaternion identity. analog: methane slot."""
    cli = unit_axis(insertion_axis)
    angle = DEFAULT_ANGLE_RAD if insertion_angle_rad is None else float(insertion_angle_rad)
    ax = axis_for_mode(step_mode, step_index=0, cli_axis=cli, angle_rad=angle)
    q = IDENTITY_Q.copy()
    rule = AXIS_RULE.get(step_mode, "cli_fixed")
    alias = alias_for(1, family=alias_family)
    return HomologState(
        n=1,
        alias=alias,
        q=q,
        flywheels=[make_flywheel(quaternion=q)],
        step_mode=step_mode,
        insertion_axis=ax,
        insertion_angle_rad=angle,
        invariants={},
        notes=(
            f"{NOTES_PREFIX} n=1 identity flywheel seed; alias={alias} (display label); "
            f"family={alias_family}; frozen Z={int(Z)} is not n; axis_rule={rule}."
        ),
        Z=int(Z),
        cli_axis=cli,
        alias_family=alias_family,
        group_id=group_id,
    )
