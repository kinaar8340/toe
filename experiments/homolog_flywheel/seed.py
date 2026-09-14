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

from homolog_flywheel.analog import DEFAULT_STEP_MODE, FROZEN_Z, NOTES_PREFIX, alias_for
from homolog_flywheel.insert import DEFAULT_ANGLE_RAD, make_flywheel, unit_axis
from homolog_flywheel.state import HomologState

Array = NDArray[np.floating]

IDENTITY_Q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


def identity_seed(
    *,
    step_mode: str = DEFAULT_STEP_MODE,
    insertion_axis: Array | None = None,
    insertion_angle_rad: float | None = None,
    Z: int = FROZEN_Z,
) -> HomologState:
    """n=1: one flywheel at quaternion identity. analog: methane slot."""
    ax = unit_axis(insertion_axis)
    angle = DEFAULT_ANGLE_RAD if insertion_angle_rad is None else float(insertion_angle_rad)
    q = IDENTITY_Q.copy()
    return HomologState(
        n=1,
        alias=alias_for(1),
        q=q,
        flywheels=[make_flywheel(quaternion=q)],
        step_mode=step_mode,
        insertion_axis=ax,
        insertion_angle_rad=angle,
        invariants={},
        notes=(
            f"{NOTES_PREFIX} n=1 identity flywheel seed; alias=methan (display label); "
            f"frozen Z={int(Z)} is not n."
        ),
        Z=int(Z),
    )
