"""Unit tests for λt normalization and survival analog comparison."""

import pytest

from relaxation_survival import (
    E_INV2,
    GOLDEN_ANGLE_FRACTION,
    R_RESIDUAL,
    compare_to_analogs,
    steps_for_lambda_t,
)


def test_steps_for_lambda_t_at_kappa_doc():
    """κ=0.85, dt=0.001, λt=2 → n_steps=2353 (t_phys=2/κ)."""
    assert steps_for_lambda_t(lambda_t_target=2.0, kappa=0.85, dt=0.001) == 2353


@pytest.mark.parametrize(
    "kappa,expected",
    [
        (0.80, 2500),
        (0.85, 2353),
        (0.90, 2222),
    ],
)
def test_steps_for_lambda_t_scales_inversely_with_kappa(kappa: float, expected: int):
    assert steps_for_lambda_t(2.0, kappa, 0.001) == expected


def test_compare_to_analogs_hybrid_score_near_r():
    """Measured ≈ R should score well on hybrid rotational+dissipative metric."""
    result = compare_to_analogs(R_RESIDUAL, "test")
    assert result["best_match"] in {"R_phi_e_pi", "golden_angle_over_1000"}
    assert result["hybrid_score"] > 0.9
    assert result["hybrid_delta_pct"] < 5.0


def test_compare_to_analogs_includes_reference_constants():
    result = compare_to_analogs(E_INV2, "e_inv2_test")
    assert result["measured"] == pytest.approx(E_INV2)
    assert result["candidates"]["R_phi_e_pi"] == pytest.approx(R_RESIDUAL)
    assert result["candidates"]["golden_angle_over_1000"] == pytest.approx(GOLDEN_ANGLE_FRACTION)
