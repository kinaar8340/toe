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
import pytest

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from homolog_flywheel.analog import FROZEN_Z, STEP_MODES, alias_for
from homolog_flywheel.compare import (
    OFF_YZ_AXIS,
    compose_published_qs,
    expected_s2_drift,
    mode_comparison,
    published_axis_probe,
    rotor_single_axis_overlap,
    run_chain,
)
from homolog_flywheel.insert import DEFAULT_ANGLE_RAD, DEFAULT_AXIS, axis_for_mode, insert
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


def test_rotor_overlap_is_single_axis_cosine():
    theta = DEFAULT_ANGLE_RAD
    states = run_chain(4, "rotor", theta, DEFAULT_AXIS)
    for state in states:
        expected = rotor_single_axis_overlap(state.n, theta)
        assert state.invariants["identity_overlap"] == pytest.approx(expected, abs=1e-6)


def test_published_overlaps_match_independent_composition():
    theta = DEFAULT_ANGLE_RAD
    states = run_chain(4, "published", theta, DEFAULT_AXIS)
    left = compose_published_qs(4, theta, DEFAULT_AXIS, side="left")
    right = compose_published_qs(4, theta, DEFAULT_AXIS, side="right")
    for state, q_l, q_r in zip(states, left, right, strict=True):
        got = float(state.invariants["identity_overlap"])
        assert got == pytest.approx(abs(float(q_l[0])), abs=1e-6)
        assert got == pytest.approx(abs(float(q_r[0])), abs=1e-6)
        assert abs(float(q_l[0])) == pytest.approx(abs(float(q_r[0])), abs=1e-6)


def test_default_published_axis_stays_in_yz():
    states = run_chain(4, "published", DEFAULT_ANGLE_RAD, DEFAULT_AXIS)
    for state in states:
        assert abs(float(state.invariants["axis_x"])) < 1e-6
    drifts = [float(s.invariants["axis_drift_rad"]) for s in states if s.n >= 2]
    assert drifts
    assert all(abs(d - DEFAULT_ANGLE_RAD) < 1e-6 for d in drifts)


def test_published_off_yz_walk_phase_stays_theta_overlap_changes():
    theta = DEFAULT_ANGLE_RAD
    rows, summaries = published_axis_probe(4, theta, DEFAULT_AXIS, OFF_YZ_AXIS)
    assert all(int(r["Z"]) == FROZEN_Z for r in rows)
    n4 = [r for r in rows if int(r["n"]) == 4]
    assert {r["alias"] for r in n4} == {"butan"}
    assert len(summaries) == 2
    ov_z, ov_off = (float(s["identity_overlap_at_nmax"]) for s in summaries)
    assert abs(ov_z - ov_off) > 1e-3
    for row in rows:
        if int(row["n"]) >= 2:
            assert float(row["walk_phase_rad"]) == pytest.approx(theta, abs=1e-6)
            assert float(row["step_geodesic_rad"]) == pytest.approx(theta, abs=1e-6)
    off_rows = [r for r in rows if int(r["n"]) >= 2 and abs(float(r["axis_x"])) > 1e-6]
    assert off_rows
    expected_off = expected_s2_drift(OFF_YZ_AXIS, theta)
    for row in off_rows:
        assert float(row["axis_drift_rad"]) == pytest.approx(expected_off, abs=1e-6)


def _s2_chord_formula(cli_axis: np.ndarray, theta: float) -> tuple[float, float]:
    """Independent S² chord: arccos(cos²φ + sin²φ cosθ), φ = angle(v, bake-x).

    analog: this is the embedding of walk_phase on S², not walk_phase itself.
    """
    bake_x = np.array([1.0, 0.0, 0.0])
    v = np.asarray(cli_axis, dtype=float)
    v = v / np.linalg.norm(v)
    phi = float(np.arccos(np.clip(float(np.dot(v, bake_x)), -1.0, 1.0)))
    chord = float(math.acos(math.cos(phi) ** 2 + math.sin(phi) ** 2 * math.cos(theta)))
    return phi, chord


def test_s2_drift_is_chord_formula_not_walk_phase():
    """Probe axes: axis_drift matches the chord formula; it is not walk_phase."""
    theta = DEFAULT_ANGLE_RAD
    rows, _summaries = published_axis_probe(4, theta, DEFAULT_AXIS, OFF_YZ_AXIS)
    for cli in (DEFAULT_AXIS, OFF_YZ_AXIS):
        v = np.asarray(cli, dtype=float)
        v = v / np.linalg.norm(v)
        key = ",".join(f"{x:.6f}" for x in v)
        phi, chord = _s2_chord_formula(cli, theta)
        subset = [r for r in rows if r.get("cli_axis") == key and int(r["n"]) >= 2]
        assert subset
        for row in subset:
            assert float(row["walk_phase_rad"]) == pytest.approx(theta, abs=1e-6)
            assert float(row["axis_drift_rad"]) == pytest.approx(chord, abs=1e-6)
        if abs(phi - math.pi / 2) < 1e-6:
            assert chord == pytest.approx(theta, abs=1e-6)
        else:
            assert abs(chord - theta) > 1e-3


def test_disclaimer_names_insertion_words():
    from homolog_flywheel.analog import DISCLAIMER

    assert "group ids are insertion words; molecular names are alias families." in DISCLAIMER


def test_catalog_z_frozen_and_aliases_by_family():
    from homolog_flywheel.catalog import run_catalog

    rows, meta = run_catalog()
    assert int(meta["z_frozen"]) == FROZEN_Z
    assert all(int(r["Z"]) == FROZEN_Z for r in rows)
    alkane = [r for r in rows if r["alias_family"] == "alkane" and r["n"] == 4]
    iso = [r for r in rows if r["alias_family"] == "isoalkane" and r["n"] == 4]
    cyclo = [r for r in rows if r["alias_family"] == "cyclo" and r["n"] == 4]
    assert alkane
    assert {r["alias"] for r in alkane} == {"butan"}
    assert iso and {r["alias"] for r in iso} == {"iso-butan"}
    assert cyclo and {r["alias"] for r in cyclo} == {"cyclo-butan"}
    modes = {r["step_mode"] for r in alkane}
    assert "rotor" in modes and "published" in modes


def test_branch_commutator_nonzero():
    from homolog_flywheel.catalog import run_catalog

    rows, _meta = run_catalog()
    branch = [r for r in rows if r["group_id"] == "branch_yz"]
    assert branch
    for row in branch:
        assert float(row["commutator_norm"]) > 1e-6
        assert int(row["Z"]) == FROZEN_Z


def test_ring4_closure_is_not_tuned_to_zero():
    from homolog_flywheel.catalog import run_catalog

    rows, _meta = run_catalog()
    ring = [r for r in rows if r["group_id"] == "ring4_rotor" and r["n"] == 4]
    assert len(ring) == 1
    # analog: golden θ does not close a square; do not curve-fit to cycloalkane
    assert float(ring[0]["closure_rad"]) > 1e-3
    assert ring[0]["alias"] == "cyclo-butan"
