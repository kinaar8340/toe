"""HomologState dataclass for the homolog_flywheel Model analog.

MODEL, not theorem.
Geometry and arithmetic stay on QGA / flux_hopf_lib.
The alkane / CH2 / carbene language is an analogy for a discrete insertion step.
n=1 is one flywheel at quaternion identity (methane slot).
n=2 is one extra published step: one extra flywheel XOR one extra rotor insertion.
Do not emit “proves”, “element”, “periodic table identity”, or “carbene is a flywheel”.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from homolog_flywheel.analog import FROZEN_Z, NOTES_PREFIX, alias_for


@dataclass
class HomologState:
    """Discrete n-step analog state. Not a chemical identity."""

    n: int
    alias: str
    q: NDArray[np.floating]
    flywheels: list[Any]
    step_mode: str
    insertion_axis: NDArray[np.floating]
    insertion_angle_rad: float
    invariants: dict[str, Any] = field(default_factory=dict)
    notes: str = NOTES_PREFIX
    Z: int = FROZEN_Z  # frozen element-analog index; insert must not change this
    cli_axis: NDArray[np.floating] | None = None  # user/CLI reference; mode rule maps this

    def __post_init__(self) -> None:
        self.q = np.asarray(self.q, dtype=float).reshape(4)
        self.insertion_axis = np.asarray(self.insertion_axis, dtype=float).reshape(3)
        if self.cli_axis is None:
            self.cli_axis = np.array(self.insertion_axis, dtype=float, copy=True)
        else:
            self.cli_axis = np.asarray(self.cli_axis, dtype=float).reshape(3)
        if not self.notes.startswith(NOTES_PREFIX):
            self.notes = f"{NOTES_PREFIX} {self.notes}"
        if self.alias == "":
            self.alias = alias_for(self.n)
