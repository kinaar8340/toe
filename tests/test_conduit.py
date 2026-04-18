"""Unit tests for the core conduit module."""

import pytest
import torch
from conduit import (
    qmul,
    qnormalize,
    # add any other helpers/classes you want to test
    RubikConeConduit,
    # ...
)


def test_quaternion_basic_operations():
    """Test pure quaternion math functions (fast & deterministic)."""
    q1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    q2 = torch.tensor([0.0, 1.0, 0.0, 0.0])

    result = qmul(q1, q2)
    expected = torch.tensor([0.0, 1.0, 0.0, 0.0])

    assert torch.allclose(result, expected, atol=1e-6)


def test_rubik_cone_conduit_initialization():
    """Basic smoke test for the main conduit class."""
    conduit = RubikConeConduit(...)  # pass minimal valid config
    assert conduit is not None
    # add more assertions on invariants, etc.


@pytest.mark.slow
def test_heavy_simulation():
    """Mark expensive tests so they can be skipped in quick CI runs."""
    # ...
    pass