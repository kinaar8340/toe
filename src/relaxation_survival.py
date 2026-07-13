"""
relaxation_survival.py
======================
Thin compatibility shim — canonical implementation lives in flux_hopf_lib.

Prefer::

    from flux_hopf_lib.simulation import (
        simulate_twist_pde_survival,
        evolve_gauged_twist_survival,
        compare_to_analogs,
        steps_for_lambda_t,
    )

This module re-exports the same public API so existing toe / mystery call sites
(``from relaxation_survival import …``) keep working during the migration.

Normalize simulation time / accumulated twist to dimensionless λt = 2.
Mean-field gauge -κ θ̄ ⇒ λ ≈ κ; at λt = 2 survival tracks e^{-2} ≈ 0.135335.
"""

from __future__ import annotations

from flux_hopf_lib.constants import (
    E_INV2,
    GOLDEN_ANGLE_DEG,
    GOLDEN_ANGLE_FRACTION,
    PHI,
    PHI_INV2,
    PI,
    R_RESIDUAL,
    E,
)
from flux_hopf_lib.simulation import (
    LambdaTNormalization,
    SurvivalAnalogs,
    compare_to_analogs,
    evolve_gauged_twist_survival,
    simulate_twist_pde_survival,
    steps_for_lambda_t,
)
from flux_hopf_lib.simulation.relaxation import simulate_twist_pde, twist_pde_step

__all__ = [
    "E_INV2",
    "GOLDEN_ANGLE_DEG",
    "GOLDEN_ANGLE_FRACTION",
    "PHI",
    "PHI_INV2",
    "PI",
    "R_RESIDUAL",
    "E",
    "LambdaTNormalization",
    "SurvivalAnalogs",
    "compare_to_analogs",
    "evolve_gauged_twist_survival",
    "simulate_twist_pde",
    "simulate_twist_pde_survival",
    "steps_for_lambda_t",
    "twist_pde_step",
]
