"""Thin wrapper so `pytest tests/` collects the homolog_flywheel experiment tests.

MODEL, not theorem.
Geometry and arithmetic stay on QGA / flux_hopf_lib.
The alkane / CH2 / carbene language is an analogy for a discrete insertion step.
n=1 is one flywheel at quaternion identity (methane slot).
n=2 is one extra published step: one extra flywheel XOR one extra rotor insertion.
Do not emit “proves”, “element”, “periodic table identity”, or “carbene is a flywheel”.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_EXP = _REPO / "experiments"
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

_path = _EXP / "tests" / "test_homolog_flywheel.py"
_spec = importlib.util.spec_from_file_location("test_homolog_flywheel_impl", _path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"analog: cannot load experiment tests from {_path}")
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

test_n1_is_identity = _mod.test_n1_is_identity
test_n2_alias_ethan = _mod.test_n2_alias_ethan
test_rotor_step_preserves_unit_norm = _mod.test_rotor_step_preserves_unit_norm
test_one_step_not_three = _mod.test_one_step_not_three
test_z_is_not_n = _mod.test_z_is_not_n
test_readme_disclaimer_present = _mod.test_readme_disclaimer_present
test_geodesic_defined_for_n_ge_2 = _mod.test_geodesic_defined_for_n_ge_2
test_aliases_identical_across_modes = _mod.test_aliases_identical_across_modes
test_axis_rules_differ_across_modes = _mod.test_axis_rules_differ_across_modes
test_compare_modes_does_not_promote_n_to_Z = _mod.test_compare_modes_does_not_promote_n_to_Z
test_rotor_overlap_is_single_axis_cosine = _mod.test_rotor_overlap_is_single_axis_cosine
test_published_overlaps_match_independent_composition = (
    _mod.test_published_overlaps_match_independent_composition
)
test_default_published_axis_stays_in_yz = _mod.test_default_published_axis_stays_in_yz
test_published_off_yz_walk_phase_stays_theta_overlap_changes = (
    _mod.test_published_off_yz_walk_phase_stays_theta_overlap_changes
)
test_s2_drift_is_chord_formula_not_walk_phase = _mod.test_s2_drift_is_chord_formula_not_walk_phase
test_disclaimer_names_insertion_words = _mod.test_disclaimer_names_insertion_words
test_catalog_z_frozen_and_aliases_by_family = _mod.test_catalog_z_frozen_and_aliases_by_family
test_branch_commutator_nonzero = _mod.test_branch_commutator_nonzero
test_ring4_closure_is_not_tuned_to_zero = _mod.test_ring4_closure_is_not_tuned_to_zero
test_catalog_shards_partition_groups = _mod.test_catalog_shards_partition_groups
test_ring4_keeps_n_max_when_cli_override_is_8 = _mod.test_ring4_keeps_n_max_when_cli_override_is_8
