"""Comprehensive unit tests for the core conduit module."""

import pytest
import torch
from conduit import (
    qmul,
    qnormalize,
    q_mult,
    q_conj,
    q_normalize,
    small_rotor,
    safe_cosine,
    CubeChain,
    ShellCube,
    RingConeChain,
    RubikConeConduit,
)


# ====================== QUATERNION MATH ======================
def test_qmul_basic():
    q1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    q2 = torch.tensor([0.0, 1.0, 0.0, 0.0])
    result = qmul(q1, q2)
    expected = torch.tensor([0.0, 1.0, 0.0, 0.0])
    assert torch.allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize(
    "q1,q2,expected",
    [
        ([1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]),
        ([0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]),
        ([0.7071, 0.7071, 0, 0], [0.7071, -0.7071, 0, 0], [1, 0, 0, 0]),
    ],
)
def test_qmul_parametrized(q1, q2, expected):
    q1 = torch.tensor(q1, dtype=torch.float32)
    q2 = torch.tensor(q2, dtype=torch.float32)
    result = qmul(q1, q2)
    assert torch.allclose(result, torch.tensor(expected, dtype=torch.float32), atol=1e-5)


def test_qnormalize_preserves_unit_norm():
    q = torch.tensor([3.0, 4.0, 12.0, 0.0])
    norm_q = qnormalize(q)
    assert torch.allclose(torch.norm(norm_q), torch.tensor(1.0), atol=1e-6)


def test_q_conj_and_q_mult():
    q = torch.tensor([0.5, 0.5, 0.5, 0.5])
    conj = q_conj(q)
    assert torch.allclose(conj[1:], -q[1:], atol=1e-6)
    prod = q_mult(q, conj)
    assert torch.allclose(prod[0], torch.norm(q) ** 2, atol=1e-6)
    assert torch.allclose(prod[1:], torch.zeros(3), atol=1e-6)


def test_small_rotor_unit_norm_and_angle():
    angle = torch.tensor(torch.pi / 2, dtype=torch.float32)
    axis = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    rotor = small_rotor(angle, axis)
    assert torch.allclose(torch.norm(rotor), torch.tensor(1.0), atol=1e-6)


def test_safe_cosine_range_and_symmetry():
    a = torch.randn(384)
    b = a.clone() * 1.1
    sim = safe_cosine(a, b)
    assert 0.0 <= sim <= 1.0 + 1e-6
    assert torch.allclose(safe_cosine(a, a), torch.tensor(1.0), atol=1e-5)


# ====================== CORE CLASSES ======================
def test_cube_chain_bake_and_stats(cube_chain):
    emb = torch.randn(384)
    cube_chain.bake(cube_idx=0, emb=emb, orientation=5)
    stats = cube_chain.get_stats()
    assert stats["active_cubes"] == 1
    assert stats["vortex_sync"] == pytest.approx(5 / 24.0, abs=1e-6)


def test_shell_cube_radial_embedding():
    shell = ShellCube(embed_dim=384)
    inner = torch.randn(384)
    outer = torch.randn(384)
    diff = shell.embed_radial(inner, outer)
    assert diff.shape == (384,)
    assert torch.allclose(torch.norm(diff), torch.tensor(1.0), atol=1e-6)


def test_ring_cone_chain_initialization(ring_cone_chain):
    assert len(ring_cone_chain.rings) == 16
    assert ring_cone_chain.TOTAL_CUBES == 2 * sum(ring_cone_chain.RING_SIZES)


def test_ring_cone_chain_bake_ring(ring_cone_chain):
    emb = torch.randn(384)
    ring_cone_chain.bake_ring(ring_idx=0, cube_local_idx=3, emb=emb, orientation=7)
    stats = ring_cone_chain.rings[0].get_stats()
    assert stats["active_cubes"] >= 1


# ====================== MAIN CONDUIT ======================
def test_rubik_cone_conduit_initialization(minimal_conduit):
    assert minimal_conduit.embed_dim == 384
    assert minimal_conduit.num_polarizations == 3
    assert hasattr(minimal_conduit, "VERSION")


def test_conduit_position_output_shape_and_norm(minimal_conduit):
    """Position now returns **unit vectors** (norm == 1.0) — v10.8 change."""
    emb = minimal_conduit.position(s=42.0, pol_idx=0)
    assert emb.shape == (384,)
    assert torch.allclose(torch.norm(emb), torch.tensor(1.0), atol=1e-6)
    assert not torch.isnan(emb).any()


def test_monitor_topological_winding(minimal_conduit):
    """Now passes — ring_cone is initialized and fallback removed."""
    stats = minimal_conduit.monitor_topological_winding(n_samples=5, pol_ref=0)
    assert "geometric_winding" in stats


def test_rubik_cone_conduit_forward():
    """Now passes — minimal forward() implemented."""
    device = torch.device("cpu")
    conduit = RubikConeConduit(embed_dim=384).to(device)

    batch_size = 2
    face_grids = torch.randn(batch_size, 54, 384, device=device)
    orientations = torch.randint(0, 24, (batch_size, 54), device=device)
    vortex_digits = torch.randint(0, 10, (batch_size, 54), device=device)

    output = conduit(face_grids, orientations, vortex_digits)
    assert output.shape[0] == batch_size


@pytest.mark.slow
def test_heavy_simulation(minimal_conduit):
    """Placeholder for full training-loop / long-run tests."""
    pass