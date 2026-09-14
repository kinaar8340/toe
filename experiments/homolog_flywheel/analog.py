"""Name table and labels for the homolog_flywheel Model analog.

MODEL, not theorem.
Geometry and arithmetic stay on QGA / flux_hopf_lib.
The alkane / CH2 / carbene language is an analogy for a discrete insertion step.
n=1 is one flywheel at quaternion identity (methane slot).
n=2 is one extra published step: one extra flywheel XOR one extra rotor insertion.
Do not emit “proves”, “element”, “periodic table identity”, or “carbene is a flywheel”.

German alkane strings are display aliases only, not chemical identities.
"""

from __future__ import annotations

DISCLAIMER = (
    "MODEL, not theorem.\n"
    "Geometry and arithmetic stay on QGA / flux_hopf_lib.\n"
    "The alkane / CH2 / carbene language is an analogy for a discrete insertion step.\n"
    "n=1 is one flywheel at quaternion identity (methane slot).\n"
    "n=2 is one extra published step: one extra flywheel XOR one extra rotor insertion.\n"
    "Do not emit “proves”, “element”, “periodic table identity”, or “carbene is a flywheel”.\n"
    "group ids are insertion words; molecular names are alias families."
)

# Display aliases from the source figure. Labels only.
ALKANE_ALIAS: dict[int, str] = {
    1: "methan",
    2: "ethan",
    3: "propan",
    4: "butan",
    5: "pentan",
    6: "hexan",
    7: "heptan",
    8: "octan",
    9: "nonan",
    10: "decan",
}

STEP_MODES = ("rotor", "flywheel", "published")
DEFAULT_STEP_MODE = "rotor"
N_MIN = 1
N_MAX = 10
# Frozen Z from scripts/z_flywheel_map.py first demo value. Not n.
FROZEN_Z = 2
NOTES_PREFIX = "MODEL analog:"

# analog: how each mode chooses an insertion axis. Not a chemical identity.
# rotor     — CLI / default z, fixed
# flywheel  — conduit epoch_synchronous_bake rotor axis (1,0,0)
# published — golden-angle rotate the CLI axis about bake-x
AXIS_RULE = {
    "rotor": "cli_fixed",
    "flywheel": "bake_x",
    "published": "golden_rotate_cli",
}
BAKE_X_AXIS = (1.0, 0.0, 0.0)


ALIAS_FAMILIES: dict[str, dict[int, str]] = {
    "alkane": ALKANE_ALIAS,
    "isoalkane": {n: f"iso-{name}" for n, name in ALKANE_ALIAS.items()},
    "cyclo": {n: f"cyclo-{name}" for n, name in ALKANE_ALIAS.items()},
}
DEFAULT_ALIAS_FAMILY = "alkane"


def alias_for(n: int, family: str = DEFAULT_ALIAS_FAMILY) -> str:
    """Display alias for (family, n) only. Identical across step modes. Not an identity."""
    table = ALIAS_FAMILIES.get(family)
    if table is None:
        raise ValueError(f"analog: unknown alias family {family!r}")
    if n not in table:
        raise ValueError(f"analog: n={n} is outside the alias table 1..{N_MAX} for {family}")
    return table[n]
