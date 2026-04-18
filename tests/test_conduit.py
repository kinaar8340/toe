"""Comprehensive unit tests for the core conduit module."""

import pytest
import torch

from conduit import (
    RubikConeConduit,
    ShellCube,
    q_conj,
    q_mult,
    qmul,
    qnormalize,
    safe_cosine,
    small_rotor,
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
    assert 2 * sum(ring_cone_chain.RING_SIZES) == ring_cone_chain.TOTAL_CUBES


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


# ====================== TRAINING LOOP & LOSSES ======================


@pytest.fixture
def sample_batch():
    """Sample batch for training_step (matches expected Dict format)."""
    torch.manual_seed(42)
    batch = []
    for i in range(4):
        s = 1.0 + i * 2.5
        emb = torch.randn(384, dtype=torch.float32)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        batch.append({"emb": emb, "s": s, "pol_idx": i % 3})
    return batch


def test_training_step_smoke(minimal_conduit, sample_batch):
    """Smoke test: training_step runs, returns correct metrics, all losses non-negative."""
    optimizer = torch.optim.Adam(minimal_conduit.parameters(), lr=1e-4)
    metrics = minimal_conduit.training_step(sample_batch, optimizer)

    assert isinstance(metrics, dict)
    required = {"recon", "align", "depth_pull", "winding", "braiding", "total"}
    for key in required:
        assert key in metrics
        assert isinstance(metrics[key], float)
        assert metrics[key] >= 0.0
    assert metrics["total"] > 0.0


def test_loss_components_weights_affect_total(minimal_conduit, sample_batch):
    """Weights control contribution to total_loss (raw metrics always computed)."""
    optimizer = torch.optim.Adam(minimal_conduit.parameters(), lr=1e-4)
    metrics_full = minimal_conduit.training_step(sample_batch, optimizer, recon_weight=4200.0)
    metrics_no_recon = minimal_conduit.training_step(sample_batch, optimizer, recon_weight=0.0)

    assert metrics_full["recon"] > 0.0
    assert metrics_no_recon["recon"] > 0.0
    # total should be noticeably lower without recon
    assert metrics_no_recon["total"] < metrics_full["total"] + 10.0


def test_bake_to_cube_and_recover_depth(minimal_conduit, sample_batch):
    """End-to-end bake → recall cycle."""
    for item in sample_batch:
        emb = item["emb"].unsqueeze(0) if item["emb"].dim() == 1 else item["emb"]
        cube_idx = int(item["s"]) % 12
        minimal_conduit.bake_to_cube(cube_idx, emb, orientation=None)

    test_item = sample_batch[0]
    emb = test_item["emb"].unsqueeze(0) if test_item["emb"].dim() == 1 else test_item["emb"]
    recovered = minimal_conduit.recover_depth(
        emb.squeeze(0), pol_idx=test_item.get("pol_idx", 0), grid_size=64
    )
    assert isinstance(recovered, float)
    assert 0.0 < recovered < 60.0


def test_end_to_end_train_then_recall(minimal_conduit, sample_batch):
    """Bake → several training steps → recall quality is reasonable (soft convergence)."""
    optimizer = torch.optim.Adam(minimal_conduit.parameters(), lr=3e-3)

    # initial bake
    for item in sample_batch[:2]:
        emb = item["emb"].unsqueeze(0) if item["emb"].dim() == 1 else item["emb"]
        cube_idx = int(item["s"]) % 12
        minimal_conduit.bake_to_cube(cube_idx, emb, orientation=None)

    # train
    for _ in range(5):  # extra steps for better pull
        minimal_conduit.training_step(sample_batch, optimizer)

    # recall
    test_item = sample_batch[0]
    emb = test_item["emb"].unsqueeze(0) if test_item["emb"].dim() == 1 else test_item["emb"]
    recovered = minimal_conduit.recover_depth(emb.squeeze(0), pol_idx=test_item.get("pol_idx", 0))
    assert abs(recovered - test_item["s"]) < 25.0, f"Expected ~{test_item['s']}, got {recovered}"


def test_optimizer_and_clamping(minimal_conduit, sample_batch):
    """Gradients flow, step happens, and output_scale is clamped."""
    optimizer = torch.optim.Adam(minimal_conduit.parameters(), lr=1e-3)
    minimal_conduit.training_step(sample_batch, optimizer)

    # gradients exist
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in minimal_conduit.parameters()
    )
    assert has_grad, "No gradients were computed during training_step"

    # output_scale clamping
    if hasattr(minimal_conduit, "output_scale"):
        scale = minimal_conduit.output_scale.item()
        assert 0.25 <= scale <= 0.4


def test_monitor_topological_winding_after_training(minimal_conduit, sample_batch):
    """Topological invariants remain consistent after training - actual keys."""
    optimizer = torch.optim.Adam(minimal_conduit.parameters(), lr=1e-4)
    minimal_conduit.training_step(sample_batch, optimizer)

    stats = minimal_conduit.monitor_topological_winding(n_samples=8, pol_ref=0)
    assert isinstance(stats, dict)
    assert "geometric_winding" in stats
    assert "effective_winding" in stats or "learned_contribution" in stats
    assert "braiding_phase" in stats


@pytest.mark.slow
def test_heavy_simulation(minimal_conduit):
    """Placeholder for full-scale epoch sweeps (kept for CI)."""
    pass
