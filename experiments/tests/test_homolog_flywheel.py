"""Fast tests for the homolog_flywheel Model analog.

MODEL, not theorem.
Geometry and arithmetic stay on QGA / flux_hopf_lib.
The alkane / CH2 / carbene language is an analogy for a discrete insertion step.
n=1 is one flywheel at quaternion identity (methane slot).
n=2 is one extra published step: one extra flywheel XOR one extra rotor insertion.
Do not emit “proves”, “element”, “periodic table identity”, or “carbene is a flywheel”.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from homolog_flywheel.analog import FROZEN_Z, STEP_MODES, alias_for
from homolog_flywheel.compare import mode_comparison, run_chain
from homolog_flywheel.insert import DEFAULT_AXIS, axis_for_mode, insert
from homolog_flywheel.measure import measure
from homolog_flywheel.seed import IDENTITY_Q, identity_seed


def test_n1_is_identity():
    state = identity_seed()
    assert state.n == 1
    assert state.alias == "methan"
    assert np.allclose(state.q, IDENTITY_Q, atol=1e-6)
    assert len(state.flywheels) == 1


def test_n2_alias_ethan():
    state = insert(identity_seed(step_mode="rotor"), step_mode="rotor")
    assert state.n == 2
    assert state.alias == "ethan"


def test_rotor_step_preserves_unit_norm():
    state = identity_seed(step_mode="rotor")
    for n in range(1, 6):
        assert state.n == n
        assert np.allclose(np.linalg.norm(state.q), 1.0, atol=1e-6)
        if n < 5:
            state = insert(state, step_mode="rotor")


def test_one_step_not_three():
    rotor_seed = identity_seed(step_mode="rotor")
    rotor_n2 = insert(rotor_seed, step_mode="rotor")
    assert len(rotor_seed.flywheels) == 1
    assert len(rotor_n2.flywheels) == 1

    fly_seed = identity_seed(step_mode="flywheel")
    fly_n2 = insert(fly_seed, step_mode="flywheel")
    assert len(fly_seed.flywheels) == 1
    assert len(fly_n2.flywheels) == 2


def test_z_is_not_n():
    state = identity_seed(Z=2)
    assert state.Z == 2
    assert state.n == 1
    while state.n < 5:
        prev_n = state.n
        state = insert(state, step_mode="published")
        assert state.Z == 2
        assert state.n == prev_n + 1
    assert state.n == 5
    assert state.n != state.Z


def test_readme_disclaimer_present():
    readme = Path(__file__).resolve().parents[1] / "README.md"
    text = readme.read_text(encoding="utf-8")
    assert "MODEL, not theorem" in text


def test_geodesic_defined_for_n_ge_2():
    s1 = identity_seed(step_mode="rotor")
    s2 = insert(s1, step_mode="rotor")
    row1 = measure(s1, prev=None)
    row2 = measure(s2, prev=s1)
    assert math.isnan(float(row1["step_geodesic_rad"]))
    assert not math.isnan(float(row2["step_geodesic_rad"]))
    assert float(row2["step_geodesic_rad"]) > 0.0
    assert math.isnan(float(row1["axis_drift_rad"]))
    assert not math.isnan(float(row2["axis_drift_rad"]))


def test_aliases_identical_across_modes():
    n_max = 4
    aliases = {}
    for mode in STEP_MODES:
        states = run_chain(n_max, mode, angle_rad=0.5, axis=DEFAULT_AXIS)
        aliases[mode] = [s.alias for s in states]
        assert aliases[mode] == [alias_for(n) for n in range(1, n_max + 1)]
    assert aliases["rotor"] == aliases["flywheel"] == aliases["published"]


def test_axis_rules_differ_across_modes():
    cli = DEFAULT_AXIS
    rotor = axis_for_mode("rotor", step_index=1, cli_axis=cli)
    fly = axis_for_mode("flywheel", step_index=1, cli_axis=cli)
    pub = axis_for_mode("published", step_index=1, cli_axis=cli)
    assert np.allclose(rotor, cli, atol=1e-6)
    assert np.allclose(fly, np.array([1.0, 0.0, 0.0]), atol=1e-6)
    assert not np.allclose(pub, rotor, atol=1e-3)
    assert not np.allclose(pub, fly, atol=1e-3)
    assert np.allclose(axis_for_mode("rotor", step_index=3, cli_axis=cli), rotor)


def test_compare_modes_does_not_promote_n_to_Z():
    long_rows, summaries, chains = mode_comparison(
        n_max=4,
        angle_rad=0.5,
        axis=DEFAULT_AXIS,
    )
    assert set(chains) == set(STEP_MODES)
    assert all(int(r["Z"]) == FROZEN_Z for r in long_rows)
    assert all(int(s["Z"]) == FROZEN_Z for s in summaries)
    n_vals = {int(r["n"]) for r in long_rows}
    assert n_vals == {1, 2, 3, 4}
    assert FROZEN_Z in n_vals  # n=2 exists; Z is still 2, not assigned from n
    for mode in STEP_MODES:
        zs = {int(s.Z) for s in chains[mode]}
        ns = [int(s.n) for s in chains[mode]]
        assert zs == {FROZEN_Z}
        assert ns == [1, 2, 3, 4]
