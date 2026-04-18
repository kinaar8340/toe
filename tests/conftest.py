"""Pytest fixtures for the TOE conduit test suite."""

import pytest
import torch
from conduit import RubikConeConduit, CubeChain, RingConeChain


@pytest.fixture(scope="session")
def device():
    """Use CUDA if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def minimal_conduit():
    """Use parameters that match the current RubikConeConduit implementation."""
    return RubikConeConduit(
        embed_dim=384,
        twist_rate=12.5,
        max_depth=56.0,
        num_polarizations=3,   # ← critical: internal buffers are sized to 3
    )


@pytest.fixture
def cube_chain(device):
    return CubeChain(num_cubes=12, device=device)


@pytest.fixture
def ring_cone_chain(device):
    return RingConeChain(embed_dim=384, device=device)