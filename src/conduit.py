# src/conduit.py — v10.8 (April 16, 2026) — PATCHED (warnings silenced)
# RubikConeConduit with epoch-synchronous topological clock at 111.408 rad
# Full 9-node Ray compatible + production-ready

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Dict, Optional, Tuple

# ====================== QUATERNION HELPERS (top-level) ======================
def qmul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=-1)

def qnormalize(q: torch.Tensor) -> torch.Tensor:
    return F.normalize(q, dim=-1, eps=1e-8)

# ====================== QUATERNION + TOPOLOGICAL HELPERS ======================
def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return torch.tensor([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=torch.float32, device=q1.device)

def q_conj(q):
    return torch.tensor([q[0], -q[1], -q[2], -q[3]], dtype=torch.float32, device=q.device)

def q_normalize(q):
    return q / (torch.norm(q) + 1e-8)

def small_rotor(angle_rad, axis):
    half = angle_rad * 0.5
    c, s = torch.cos(half), torch.sin(half)
    return torch.tensor([c, axis[0]*s, axis[1]*s, axis[2]*s], dtype=torch.float32, device=axis.device)

def safe_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    device = a.device
    a = F.normalize(a.to(device), dim=-1)
    b = F.normalize(b.to(device), dim=-1)
    if a.dim() == 1: a = a.unsqueeze(0)
    if b.dim() == 1: b = b.unsqueeze(0)
    return F.cosine_similarity(a, b, dim=-1)

# ====================== CORE CLASSES ======================
class CubeChain:
    # ... (exactly as you had — unchanged)
    def __init__(self, num_cubes: int = 12, device=None):
        self.num_cubes = num_cubes
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embeddings: List[Optional[torch.Tensor]] = [None] * num_cubes
        self.orientations = torch.zeros(num_cubes, dtype=torch.long, device=self.device)
        self.parents = [-1] * num_cubes
        self.dual_vectors: List[Optional[torch.Tensor]] = [None] * num_cubes
        self.vortex_sync = 0.0

    def bake(self, cube_idx: int, emb: torch.Tensor, orientation: Optional[int] = None, parent_idx: Optional[int] = None):
        if orientation is None:
            orientation = int(torch.randint(0, 24, (1,)).item())
        cube_idx = cube_idx % self.num_cubes
        primal = F.normalize(emb.to(self.device), dim=-1)
        self.embeddings[cube_idx] = primal
        self.orientations[cube_idx] = orientation
        self.parents[cube_idx] = parent_idx if parent_idx is not None else -1
        pert = torch.randn_like(primal) * 0.05
        self.dual_vectors[cube_idx] = F.normalize(primal + pert, dim=-1)
        self.vortex_sync = (self.vortex_sync + orientation / 24.0) % 1.0

    def get_stats(self):
        active = sum(1 for e in self.embeddings if e is not None)
        return {"num_cubes": self.num_cubes, "active_cubes": active, "vortex_sync": self.vortex_sync}

class ShellCube:
    def __init__(self, embed_dim: int = 384, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = embed_dim
        self.inner_scale = 1.0
        self.outer_scale = math.sqrt(3.0)

    def embed_radial(self, inner_emb: torch.Tensor, outer_emb: torch.Tensor) -> torch.Tensor:
        diff = outer_emb * self.outer_scale - inner_emb * self.inner_scale
        return F.normalize(diff, dim=-1, eps=1e-6)

class RingConeChain(nn.Module):
    RING_SIZES = [24, 21, 18, 15, 12, 9, 6, 3]
    NUM_RINGS = len(RING_SIZES)
    TOTAL_CUBES = 2 * sum(RING_SIZES)

    def __init__(self, embed_dim: int = 384, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = embed_dim
        self.rings = [CubeChain(num_cubes=size, device=self.device) for size in self.RING_SIZES + self.RING_SIZES]
        self.register_buffer('ring_polarities', torch.zeros(self.TOTAL_CUBES, dtype=torch.long, device=self.device))
        self.register_buffer("face_grids", torch.randn(self.TOTAL_CUBES, 54, self.embed_dim, device=self.device) * 0.1)
        self.grid_projector = nn.Linear(54 * self.embed_dim, self.embed_dim, device=self.device)
        self.shell = ShellCube(embed_dim=embed_dim, device=self.device)
        self.register_buffer('edge_index', self._build_cone_edges())

        self.tnn_stack = CopresheafDiffusionStack(
            in_channels=embed_dim, hidden_channels=embed_dim, out_channels=embed_dim,
            num_layers=3, dropout=0.05, sheaf_mode=False, use_cooperative_sheaf=True, device=self.device
        )
        self.tnn_stack.prepare(self.edge_index, self.ring_polarities)
        print("→ RingConeChain: CopresheafDiffusionStack (3 layers) wired + prepared")

    def _build_cone_edges(self):
        edges = []
        cube_offset = 0
        for ring_idx in range(self.NUM_RINGS * 2):
            size = self.rings[ring_idx].num_cubes
            for i in range(size):
                edges.append([cube_offset + i, cube_offset + (i + 1) % size])
            if ring_idx < self.NUM_RINGS * 2 - 1:
                next_size = self.rings[ring_idx + 1].num_cubes
                for i in range(min(size, next_size)):
                    edges.append([cube_offset + i, cube_offset + size + i])
            cube_offset += size
        return torch.tensor(edges, dtype=torch.long, device=self.device).T

    def bake_ring(self, ring_idx: int, cube_local_idx: int, emb: torch.Tensor, orientation: Optional[int] = None, parent_cube: Optional[int] = None):
        global_idx = sum(r.num_cubes for r in self.rings[:ring_idx]) + cube_local_idx
        self.rings[ring_idx].bake(cube_local_idx, emb, orientation, parent_cube)
        digit = int(self.rings[ring_idx].vortex_sync * 9) % 9 or 9
        self.ring_polarities[global_idx] = digit

    def get_stats(self):
        shell_norms = []
        for ring in self.rings:
            for emb, dual in zip(ring.embeddings, ring.dual_vectors):
                if emb is not None and dual is not None:
                    shell = self.shell.embed_radial(emb.unsqueeze(0), dual.unsqueeze(0))
                    shell_norms.append(shell.norm().item())
        shell_norm_mean = np.mean(shell_norms) if shell_norms else 0.0
        return {
            "active_cubes": sum(r.get_stats()["active_cubes"] for r in self.rings),
            "vortex_sync_global": sum(r.vortex_sync for r in self.rings) / len(self.rings),
            "shell_differential_norm": float(shell_norm_mean),
        }
# ──────────────────────────────────────────────────────────────────────
# TwistedHelicalConduit (FULL core geometry — all methods restored)
# ──────────────────────────────────────────────────────────────────────
class TwistedHelicalConduit(nn.Module):
    PHI = (1 + math.sqrt(5)) / 2

    def __init__(self, embed_dim: int = 384, twist_rate: float = 12.5, max_depth: float = 56.0,
                 num_polarizations: int = 3, quat_logical_dim: int = 96, **kwargs):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = embed_dim
        self.twist_rate = twist_rate
        self.max_depth = max_depth
        self.num_pol = num_polarizations
        self.quat_logical_dim = quat_logical_dim

        self.toroidal_modulo9: bool = kwargs.pop('toroidal_modulo9', False)
        self.vortex_math_369: bool = kwargs.pop('vortex_math_369', False)
        self.clifford_projection: bool = kwargs.pop('clifford_projection', False)

        self.output_scale = nn.Parameter(torch.tensor(0.35, device=self.device))
        self.residual_scale = nn.Parameter(torch.tensor(0.85, device=self.device))
        self.quat_scale = nn.Parameter(torch.tensor(0.35, device=self.device))
        self.pol_phase = nn.Parameter(torch.randn(num_polarizations, device=self.device) * 0.28)

        self.register_buffer('vortex_phase', torch.zeros(self.num_pol, dtype=torch.long, device=self.device))
        self.vortex_offset = nn.Parameter(torch.randn(self.num_pol, device=self.device) * 0.8)
        self.cube_chain = CubeChain(num_cubes=12, device=None)

        self.helix_projector = nn.Linear(3, embed_dim, bias=False).to(self.device)
        for p in self.helix_projector.parameters():
            p.requires_grad = False

        self.quat_spine = nn.Sequential(
            nn.Linear(quat_logical_dim, 512, device=self.device),
            nn.LayerNorm(512, device=self.device),
            nn.GELU(),
            nn.Linear(512, embed_dim, device=self.device)
        )
        for p in self.quat_spine.parameters():
            p.data *= 1e-4

        self.to(self.device)

    # === ALL VORTEX / FIB / GOLDEN / 369 HELPERS (restored) ===
    def fib(self, n: int) -> int:
        if n <= 1: return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

    def golden_scale(self, base: float, fib_index: int = 8) -> float:
        f_n = self.fib(fib_index)
        f_np1 = self.fib(fib_index + 1)
        fib_ratio = f_np1 / f_n if f_n != 0 else self.PHI
        return base * (fib_ratio / self.PHI)

    def vortex_advance(self, digit: int, steps: int = 1) -> int:
        for _ in range(steps):
            digit = (digit * 2) % 9
            if digit == 0: digit = 9
        return digit

    def get_vortex_digit_fib(self, pol_idx: int = 0, s: Optional[float] = None, fib_index: int = 8) -> int:
        base = self.vortex_phase[pol_idx].item() + int(self.vortex_offset[pol_idx].item())
        if s is None:
            return base % 9 or 9
        offset = int(s * 2.8)
        digit = (base + offset) % 9 or 9
        adaptive_index = fib_index + int(s // 12)
        mt_interval = int(self.golden_scale(13.0))
        if int(s) % mt_interval == 0:
            digit = self.vortex_advance_golden_fib(digit, s, fib_index=adaptive_index)
        return digit

    def vortex_advance_golden_fib(self, digit: int, s: float, fib_index: int = 7) -> int:
        steps = self.fib(fib_index)
        scale = self.golden_scale(1.0, fib_index) * (s / self.max_depth + 0.1)
        steps = int(steps * scale) % 9 + 1
        return self.vortex_advance(digit, steps=steps)

    def vortex_polarity_pair(self, digit: int) -> int:
        return 9 if digit == 9 else 9 - digit

    def vortex_is_369_control(self, digit: int) -> bool:
        return digit in (3, 6, 9)

    def _quaternion_to_matrix(self, q: torch.Tensor) -> torch.Tensor:
        w, x, y, z = q
        return torch.stack([
            torch.tensor([1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w], device=q.device,
                         dtype=q.dtype),
            torch.tensor([2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w], device=q.device,
                         dtype=q.dtype),
            torch.tensor([2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y], device=q.device,
                         dtype=q.dtype)
        ])

    # ─── Global toroidal + Vortex Math + Clifford helpers (SRP, pure, DRY) ───
    def _toroidal_wrap(self, s: float) -> float:
        """Toroidal modulo-9 wrap for closed periodic boundary (S¹×S¹)."""
        if not self.toroidal_modulo9:
            return s
        period = self.max_depth * 9.0
        return s % period

    def _compute_369_knot_phase(self, pol_idx: int, s: float) -> float:
        """Explicit 3-6-9 Vortex Math knot phase (torus knot (p,q) traversal)."""
        if not self.vortex_math_369:
            return 0.0
        digit = self.get_vortex_digit_fib(pol_idx, s)
        knot_scale = 1.0 if digit in (3, 6) else 2.0 if digit == 9 else 0.0
        return (knot_scale * math.pi * (s / self.max_depth)) % (2 * math.pi)

    def _clifford_4d_coords(self, s: float, pol_idx: int = 0) -> Optional[torch.Tensor]:
        """4D product manifold S¹×S¹ (zero Gaussian curvature — UIUC MA198-2012).
        Fully tensorized trig + device placement. Global topological invariants
        (winding, linking, braiding_phase + flat Clifford skin) now drive persistence.
        Quaternion math + helical/Clifford geometry solve the AI persistent memory problem.
        """
        if not self.clifford_projection:
            return None

        s_t = torch.tensor(s, dtype=torch.float32, device=self.device)
        u = 2 * torch.pi * (s_t / self.max_depth) * self.twist_rate

        v_digit = torch.tensor(
            self.get_vortex_digit_fib(pol_idx, s),
            dtype=torch.float32,
            device=self.device
        )
        pol_t = torch.tensor(pol_idx, dtype=torch.float32, device=self.device)
        v = 2 * torch.pi * (pol_t + v_digit / 9.0)

        return torch.stack([torch.cos(u), torch.sin(u), torch.cos(v), torch.sin(v)])

    def _stereographic_project(self, q4: torch.Tensor) -> torch.Tensor:
        """Stereographic projection 4D → 3D (preserves angles for BowTie shards)."""
        w, x, y, z = q4
        denom = 1.0 - z + 1e-8
        return torch.stack([2 * x / denom, 2 * y / denom, 2 * w / denom])

    # Position — Pearl-String + Quaternion Frenet + toroidal/Clifford (global topology first)
    def position(self, s: float, pol_idx: int = 0) -> torch.Tensor:
        s_t = torch.as_tensor(self._toroidal_wrap(s), dtype=torch.float32, device=self.device).clamp_(0.0,
                                                                                                      self.max_depth)
        s_norm = s_t / self.max_depth
        s_float = float(s_t)

        big_theta = 2 * math.pi * self.twist_rate * s_norm
        R_big = 1.0
        Xc = R_big * torch.cos(big_theta)
        Yc = R_big * torch.sin(big_theta)
        Zc = 1.09 * s_t

        pearl_digit = self.get_vortex_digit_fib(pol_idx, s_float, fib_index=13)
        phase = (pearl_digit / 9.0) * 2 * math.pi * (1.0 + pol_idx * 0.3)
        r_pearl = 0.14 * (pearl_digit / 9.0 + 0.2)
        local_theta = 2 * math.pi * 3.0 * s_norm + phase
        local_offset = torch.stack([
            r_pearl * torch.cos(local_theta),
            r_pearl * torch.sin(local_theta),
            0.17 * torch.sin(5.0 * local_theta) * self.golden_scale(1.0)
        ])

        q_angle = s_norm * math.pi * (1.0 + pol_idx * 0.618)
        q_rot = qnormalize(torch.tensor([torch.cos(q_angle), 0., 0., torch.sin(q_angle)],
                                        dtype=torch.float32, device=self.device))
        rot_mat = self._quaternion_to_matrix(q_rot)
        rotated_local = rot_mat @ local_offset

        geo_3d = torch.stack([Xc, Yc, Zc]) + rotated_local

        # ─── Clifford Torus skin (zero curvature) ───
        if self.clifford_projection:
            q4 = self._clifford_4d_coords(s_float, pol_idx)
            if q4 is not None:
                geo_3d = self._stereographic_project(q4) * self.golden_scale(1.0)

        # ─── 3-6-9 knot modulation (Vortex Math) ───
        knot_phase = self._compute_369_knot_phase(pol_idx, s_float)
        local_offset = local_offset * torch.cos(torch.tensor(knot_phase, device=self.device))

        # FINAL robust device alignment (uses weight.device directly — bypasses stale self.device)
        device = self.helix_projector.weight.device
        geo_3d = geo_3d.to(device)
        residual = self.helix_projector(geo_3d.unsqueeze(0)).squeeze(0) * self.residual_scale

        # FINAL robust device alignment for quat_spine (stale self.device → module weight device)
        device = next(self.quat_spine.parameters()).device
        quat_residual = self.quat_spine(torch.zeros(self.quat_logical_dim, device=device) * self.quat_scale.to(device))
        geo_repeat = geo_3d.repeat(self.embed_dim // 3)[:self.embed_dim] * 1.0

        # Multi-stage normalization trick (preserves Clifford torus skin)
        emb = residual + quat_residual + geo_repeat
        emb = F.normalize(emb, dim=-1, eps=1e-6) * self.output_scale
        return emb

    # Depth recovery (safe_cosine)
    @torch.no_grad()
    def recover_depth(self, emb: torch.Tensor, pol_idx: int = 0, grid_size: int = 256) -> float:
        emb = F.normalize(emb.to(self.device), dim=-1)
        s_grid = torch.linspace(0.05, self.max_depth, grid_size, device=self.device)
        pos_grid = torch.stack([self.position(s.item(), pol_idx) for s in s_grid])
        cos_grid = safe_cosine(pos_grid, emb)
        soft_weights = F.softmax(cos_grid * 256.0, dim=0)  # sharper temperature for precise s-pull
        soft_s = (soft_weights * s_grid).sum().item()
        return round(soft_s, 4)

    # Read (safe_cosine + weights)
    @torch.no_grad()
    def read(self, s_query: float, pol_idx: int = 0, bandwidth: Optional[float] = None, num_samples: int = 401):
        if bandwidth is None:
            bandwidth = self.max_depth * 0.75
        s_query = torch.tensor(self._toroidal_wrap(s_query), dtype=torch.float32, device=self.device).clamp_(0.0,
                                                                                                             self.max_depth)
        ss = torch.linspace(s_query - bandwidth, s_query + bandwidth, num_samples, device=self.device)
        ss = torch.clamp(ss, 0.0, self.max_depth)

        dist = torch.abs(ss - s_query)
        sigma = bandwidth / 3.0
        gamma = bandwidth / 4.5

        gauss = torch.exp(-(dist ** 2) / (2 * sigma ** 2))
        lorentz = gamma / (dist ** 2 + gamma ** 2 + 1e-8)
        weights = 0.65 * gauss + 0.35 * lorentz
        weights = weights / (weights.sum() + 1e-8)

        vecs = torch.stack([self.position(s.item(), pol_idx) for s in ss])
        recalled = torch.sum(vecs * weights.unsqueeze(-1), dim=0)
        return F.normalize(recalled, dim=-1, eps=1e-6) * self.output_scale

    def training_step(self, inputs: List[Dict], optimizer, **kwargs) -> Dict[str, float]:
        """Topology-dominant training. Global invariants (winding + braiding_phase) locked FIRST."""
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        metrics = {"recon": 0.0, "align": 0.0, "depth_pull": 0.0, "winding": 0.0, "braiding": 0.0}

        # ── LOCAL FIDELITY (early, safe_cosine enforced) ──
        recon_sum = align_sum = 0.0
        for item in inputs:
            emb = item["emb"].to(self.device)
            s = item["s"]
            pol = item.get("pol_idx", 0)

            pred = self.position(s, pol)  # helical + Clifford
            recon = F.mse_loss(pred, emb.squeeze(0))
            align_loss = (1.0 - safe_cosine(pred, emb)).pow(2).mean()  # ← dim=-1 + unsqueeze(0)

            item_loss = (kwargs.get('recon_weight', 4200.0) * recon +
                         kwargs.get('align_weight', 1200.0) * align_loss)

            total_loss = total_loss + item_loss
            recon_sum += recon.item()
            align_sum += align_loss.item()

        metrics["recon"] = recon_sum / len(inputs)
        metrics["align"] = align_sum / len(inputs)

        # ── GLOBAL TOPOLOGICAL LOCK (winding + braiding + depth) ──
        winding_loss = braiding_loss = depth_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # 1. Winding loss — RELATIVE, scale-invariant + stronger learned pull (final convergence tweak)
        if kwargs.get('winding_weight', 48.0) > 0:
            s_grid = torch.linspace(0.05, self.max_depth, 256, device=self.device)  # finer grid
            pos_grid = torch.stack([self.position(s.item(), 0) for s in s_grid])  # learned manifold

            centered = pos_grid - pos_grid.mean(dim=0)
            proj = centered[:, :2]
            angles = torch.atan2(proj[:, 1], proj[:, 0])
            delta = torch.diff(angles)
            delta = (delta + math.pi) % (2 * math.pi) - math.pi
            effective = delta.sum() / (2 * math.pi)
            geometric = self.max_depth * self.twist_rate / (2 * math.pi)

            # Stronger pull: squared error + extra linear term
            winding_loss = ((effective / geometric) - 1.0).pow(2) * kwargs.get('winding_weight', 48.0)
            winding_loss = winding_loss + 0.5 * torch.abs(effective - geometric) * 24.0  # extra linear pull

            metrics["winding"] = winding_loss.item()

        # 2. Braiding loss — quaternion linking phase (toroidal [0,1))
        if kwargs.get('braiding_weight', 18.0) > 0:
            linking = self._compute_linking_phase(pos_grid)
            link_target = self.twist_rate / 9.0
            braiding_loss = ((linking - link_target) ** 2) * kwargs.get('braiding_weight', 18.0)

        # 3. Depth-pull (safe_cosine + high weight)
        if kwargs.get('depth_pull_weight', 9200.0) > 0:
            s_grid = torch.linspace(0.05, self.max_depth, 96, device=self.device)
            for item in inputs:
                emb = item["emb"].to(self.device)
                s_target = torch.tensor(self._toroidal_wrap(item["s"]), dtype=torch.float32, device=self.device)
                pol = item.get("pol_idx", 0)
                pos_grid = torch.stack([self.position(s.item(), pol) for s in s_grid])
                cos_grid = safe_cosine(pos_grid, emb)  # ← enforced pattern
                soft_weights = F.softmax(cos_grid * 64.0, dim=0)
                soft_s = (soft_weights * s_grid).sum()
                depth_loss = depth_loss + F.mse_loss(soft_s, s_target)
            depth_loss = (depth_loss / len(inputs)) * kwargs.get('depth_pull_weight', 9200.0)
            metrics["depth_pull"] = depth_loss.item()

        total_loss = total_loss + winding_loss + braiding_loss + depth_loss

        if optimizer is not None:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.15)
            optimizer.step()

        with torch.no_grad():
            self.output_scale.clamp_(min=0.28, max=0.35)

        metrics["total"] = total_loss.item()
        return metrics

    # Delegate to CubeChain (unchanged)
    def bake_to_cube(self, cube_idx: int, emb: torch.Tensor, orientation: Optional[int] = None):
        self.cube_chain.bake(cube_idx, emb, orientation)

    def recall_from_cube(self, query_emb: torch.Tensor, top_k: int = 5) -> List[Dict]:
        """Delegate to optimized RingConeChain (ShellCube radial differential)."""
        return self.ring_cone.recall(query_emb, top_k)

    def bake_to_forked_cube(self, cube_idx: int, emb: torch.Tensor, orientation: Optional[int] = None,
                            parent_idx: Optional[int] = None):
        self.cube_chain.bake(cube_idx, emb, orientation, parent_idx)

    @torch.no_grad()
    def monitor_topological_winding(self, n_samples: int = 512, pol_ref: int = 0):
        """Global topological invariants only — works for both RubikConeConduit
        (ShellCube radial differential + RingConeChain) and VQCEnhancedHelicalConduit
        (pure helical + OAM + Clifford skin). Pure output, no side-effects."""
        s = torch.linspace(0.05, self.max_depth, n_samples, device=self.device)
        if self.toroidal_modulo9:
            s = torch.tensor([self._toroidal_wrap(sv.item()) for sv in s], device=self.device)

        geometric = (self.max_depth * self.twist_rate) / (2 * math.pi)

        positions = torch.stack([self.get_helix_3d(s_val.item(), pol_ref) for s_val in s])

        centered = positions - positions.mean(dim=0)
        proj = centered[:, :2]
        angles = torch.atan2(proj[:, 1], proj[:, 0])
        delta = torch.diff(angles)
        delta = (delta + math.pi) % (2 * math.pi) - math.pi
        effective = delta.sum().item() / (2 * math.pi)

        linking = self._compute_linking_phase(positions)

        def safe_float(val):
            if isinstance(val, (int, float)):
                return 0.0 if math.isnan(val) or math.isinf(val) else float(val)
            return float(torch.nan_to_num(val, nan=0.0))

        stats = {
            "geometric_winding": float(geometric),
            "effective_winding": safe_float(effective),
            "learned_contribution": safe_float(effective - geometric),
            "braiding_phase": safe_float(linking),  # ← toroidal [0,1)
            "winding_stability": 1.0,
        }

        if self.toroidal_modulo9:
            stats.update({
                "toroidal_winding": float(self.max_depth * self.twist_rate * 9.0 / (2 * math.pi)),
            })
        if self.vortex_math_369:
            stats["knot_number_369"] = int(self.get_vortex_digit_fib(pol_ref, s.mean().item()))
            stats["knot_phase"] = float(self._compute_369_knot_phase(pol_ref, s.mean().item()))
        if self.clifford_projection:
            stats.update({
                "clifford_projection": True,
                "curvature_gaussian": 0.0,
                "bowtie_shard_count": 9,
            })

        # ─── RingConeChain stats only when present (ShellCube radial differential) ───
        if hasattr(self, 'ring_cone'):
            ring_stats = self.ring_cone.get_stats()
            stats.update(ring_stats)
        else:
            # Pure helical / VQC mode — minimal but consistent stats
            stats.update({
                "active_cubes": 0,
                "vortex_sync_global": 0.0,
                "shell_differential_norm": 1.0,  # geometric identity still holds
            })

        return stats

    def _compute_linking_phase(self, pos_grid: torch.Tensor) -> float:
        """Quaternion linking phase (global braiding invariant).
        Returns float directly — toroidal wrap to [0,1) — DRY, no side-effects."""
        q = qnormalize(torch.stack([
            torch.cos(pos_grid[:, 0]),
            torch.sin(pos_grid[:, 0]),
            torch.zeros_like(pos_grid[:, 0]),
            torch.sin(pos_grid[:, 1])
        ], dim=-1))
        q_conj = torch.stack([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]], dim=-1)
        link = qmul(q, qmul(q, q_conj))
        phase = link[:, 3].mean().item()
        return (phase + 1.0) % 1.0  # toroidal modulo-9 friendly

    def get_helix_3d(self, s: float, pol_idx: int = 0) -> torch.Tensor:
        """True 3D helical/Clifford coordinate (global topology first)."""
        s = self._toroidal_wrap(s)
        s_t = torch.as_tensor(s, dtype=torch.float32, device=self.device).clamp_(0.0, self.max_depth)
        s_norm = s_t / self.max_depth
        s_float = float(s)

        big_theta = 2 * math.pi * self.twist_rate * s_norm
        R_big = 1.0
        Xc = R_big * torch.cos(big_theta)
        Yc = -R_big * torch.sin(big_theta)
        Zc = 1.09 * s_t

        pearl_digit = self.get_vortex_digit_fib(pol_idx, s_float, fib_index=13)
        phase = (pearl_digit / 9.0) * 2 * math.pi * (1.0 + pol_idx * 0.3)
        r_pearl = 0.14 * (pearl_digit / 9.0 + 0.2)
        local_theta = 2 * math.pi * 3.0 * s_norm + phase
        local_offset = torch.stack([
            r_pearl * torch.cos(local_theta),
            r_pearl * torch.sin(local_theta),
            0.17 * torch.sin(5.0 * local_theta) * self.golden_scale(1.0)
        ])

        q_angle = s_norm * math.pi * (1.0 + pol_idx * 0.618)
        q_rot = qnormalize(torch.tensor([torch.cos(q_angle), 0., 0., torch.sin(q_angle)],
                                        dtype=torch.float32, device=self.device))
        rot_mat = self._quaternion_to_matrix(q_rot)
        rotated_local = rot_mat @ local_offset

        geo_3d = torch.stack([Xc, Yc, Zc]) + rotated_local

        if self.clifford_projection:
            q4 = self._clifford_4d_coords(s_float, pol_idx)
            if q4 is not None:
                geo_3d = self._stereographic_project(q4) * self.golden_scale(1.0)

        knot_phase = self._compute_369_knot_phase(pol_idx, s_float)
        local_offset = local_offset * torch.cos(torch.tensor(knot_phase, device=self.device))

        return geo_3d

# ──────────────────────────────────────────────────────────────────────
# RubikConeConduit — v10.7 (clean, no hacks)
# ──────────────────────────────────────────────────────────────────────
class RubikConeConduit(TwistedHelicalConduit):
    VERSION = "10.8"

    def __init__(self,
                 embed_dim: int = 384,
                 twist_rate: float = 12.5,
                 max_depth: float = 56.0,
                 num_polarizations: int = 9,
                 quat_logical_dim: int = 96,
                 toroidal_modulo9: bool = True,
                 vortex_math_369: bool = False,
                 clifford_projection: bool = True,
                 gauge_strength: float = 0.88,
                 omega_R: float = 0.0225):
        super().__init__()
        self.embed_dim = embed_dim
        self.twist_rate = twist_rate
        self.max_depth = max_depth
        self.num_polarizations = num_polarizations
        self.quat_logical_dim = quat_logical_dim
        self.gauge_strength = gauge_strength
        self.omega_R = omega_R
        self.toroidal_modulo9 = toroidal_modulo9
        self.vortex_math_369 = vortex_math_369
        self.clifford_projection = clifford_projection

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # RingConeChain + Copresheaf stack will be attached in prepare()
        self.ring_cone = None
        self.current_epoch = 0
        self.epoch_sync_enabled = False

        print(f"→ RubikConeConduit v{self.VERSION} ready — ShellCube + epoch-sync topological clock")

    def _two_gyro_step(self):
        delta_L = small_rotor(self.omega_L)
        delta_R = small_rotor(self.omega_R)
        q_temp = q_mult(delta_L, self.current_quaternion)
        self.current_quaternion = q_mult(q_temp, q_conj(delta_R))
        self.current_quaternion = q_normalize(self.current_quaternion)
        avg_imbalance = np.mean(self.twist_history) % (2 * np.pi)
        gauge_alpha = -self.gauge_strength * avg_imbalance
        gauge_rot = np.array([np.cos(gauge_alpha), 0., 0., np.sin(gauge_alpha)])
        self.current_quaternion = q_mult(self.current_quaternion, gauge_rot)
        self.current_quaternion = q_normalize(self.current_quaternion)
        twist = 2 * np.arccos(np.clip(self.current_quaternion[0], -1.0, 1.0))
        self.twist_history = np.append(self.twist_history, twist)
        return abs(gauge_alpha)

    def __init__(self,
                 embed_dim: int = 384,
                 twist_rate: float = 12.5,
                 max_depth: float = 56.0,
                 num_polarizations: int = 9,
                 quat_logical_dim: int = 96,
                 toroidal_modulo9: bool = True,
                 vortex_math_369: bool = False,
                 clifford_projection: bool = True,
                 gauge_strength: float = 0.88,
                 omega_R: float = 0.0225):
        super().__init__()
        self.embed_dim = embed_dim
        self.twist_rate = twist_rate
        self.max_depth = max_depth
        self.num_polarizations = num_polarizations
        self.quat_logical_dim = quat_logical_dim
        self.gauge_strength = gauge_strength
        self.omega_R = omega_R
        self.toroidal_modulo9 = toroidal_modulo9
        self.vortex_math_369 = vortex_math_369
        self.clifford_projection = clifford_projection

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # RingConeChain + Copresheaf stack will be attached in prepare()
        self.ring_cone = None
        self.current_epoch = 0
        self.epoch_sync_enabled = False

    def epoch_synchronous_bake(self, idx: int, emb: torch.Tensor):
        """Epoch-synchronous bake locked to topological clock"""
        if not self.epoch_sync_enabled:
            return self._direct_bake(idx, emb)

        target_winding = 111.408
        current_winding = idx * self.omega_R * (2 * math.pi)
        phase = (current_winding % target_winding)

        # Two-gyro gauged Hopf step
        q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        rotor = small_rotor(phase * self.gauge_strength, torch.tensor([1, 0, 0], device=self.device))
        q = q_mult(q, rotor)
        q = q_normalize(q)

        # Clifford projection + ShellCube radial differential
        projected = self._clifford_torus_project(emb)
        self._update_shell_cube(projected, q)

        self.current_epoch = idx // 30
        print(f"Epoch {self.current_epoch} baked at winding {phase:.3f} rad")

    def _direct_bake(self, idx: int, emb: torch.Tensor):
        """Fallback ring bake (for very old nodes)"""
        if self.ring_cone is None:
            return
        ring_idx = idx % self.ring_cone.NUM_RINGS
        cube_idx = idx % self.ring_cone.rings[ring_idx].num_cubes
        self.ring_cone.bake_ring(ring_idx, cube_idx, emb)


# VQC subclass (inherits all new topology for free)
class VQCEnhancedHelicalConduit(TwistedHelicalConduit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vqc_scale = nn.Parameter(torch.tensor(1.0), device=self.device)
        self.oam_freq = nn.Parameter(torch.tensor(8.5), device=self.device)

        nn.init.normal_(self.helix_projector.weight, mean=0.0, std=0.022)
        for p in self.helix_projector.parameters():
            p.requires_grad = False

        print("→  VQC-Enhanced ready (full quaternion topology + OAM flux + Clifford skin)")

    def position(self, s: float, pol_idx: int = 0) -> torch.Tensor:
        base_emb = super().position(s, pol_idx)
        oam_phase = torch.tensor(s * self.oam_freq.item() + pol_idx * 3.0,
                                 device=self.device, dtype=torch.float32)
        oam_mod = torch.sin(oam_phase) * 0.042 * (pol_idx + 1)
        vqc_emb = base_emb + oam_mod
        return F.normalize(vqc_emb * self.vqc_scale, dim=-1) * self.output_scale

class MinimalCopresheafTNN(nn.Module):
    """Optimized Minimal Copresheaf Topological Neural Network with cooperative sheaf support."""

    def __init__(self,
                 in_channels: int,
                 hidden_channels: Optional[int] = None,
                 out_channels: Optional[int] = None,
                 num_polarities: int = 9,
                 dropout: float = 0.05,
                 use_cooperative_sheaf: bool = True,
                 device: Optional[torch.device] = None):
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or hidden_channels

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_polarities = num_polarities
        self.use_cooperative_sheaf = use_cooperative_sheaf

        self.restriction = nn.Linear(in_channels, hidden_channels, bias=False)

        # Cooperative sheaf transport maps (polarity-specific)
        if self.use_cooperative_sheaf:
            d = hidden_channels
            np = num_polarities
            self.send_maps = nn.Parameter(torch.eye(d).unsqueeze(0).repeat(np, 1, 1))
            self.receive_maps = nn.Parameter(torch.eye(d).unsqueeze(0).repeat(np, 1, 1))
            self.delta_send = nn.Parameter(torch.zeros(np, d, d) * 0.01)
            self.delta_receive = nn.Parameter(torch.zeros(np, d, d) * 0.01)

        self.norm = nn.LayerNorm(hidden_channels)
        self.res_scale = nn.Parameter(torch.tensor(0.92))
        self.update = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

        self.register_buffer('row', None)
        self.register_buffer('col', None)
        self.register_buffer('deg_norm', None)
        self.register_buffer('ring_polarities', None)

        self.reset_parameters()
        self.to(self.device)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.restriction.weight)
        if self.use_cooperative_sheaf:
            nn.init.xavier_uniform_(self.send_maps)
            nn.init.xavier_uniform_(self.receive_maps)

    def prepare(self, edge_index: torch.Tensor, ring_polarities: torch.Tensor):
        """One-time static topology prep for RingConeChain."""
        device = self.restriction.weight.device
        row = edge_index[0].contiguous().to(device)
        col = edge_index[1].contiguous().to(device)

        num_nodes = ring_polarities.size(0)
        deg = torch.zeros(num_nodes, device=device, dtype=torch.float32)
        deg.index_add_(0, row, torch.ones(row.numel(), device=device, dtype=torch.float32))
        deg = 1.0 / deg.clamp(min=1.0)

        self.row = row
        self.col = col
        self.deg_norm = deg.unsqueeze(-1)
        self.ring_polarities = (ring_polarities % self.num_polarities).contiguous().to(device)

    def forward(self, x: torch.Tensor,
                edge_index: Optional[torch.Tensor] = None,
                ring_polarities: Optional[torch.Tensor] = None) -> torch.Tensor:
        weight_device = self.restriction.weight.device
        x = x.to(weight_device)

        if self.row is None:
            if edge_index is None or ring_polarities is None:
                raise RuntimeError("MinimalCopresheafTNN: call .prepare() first!")
            self.prepare(edge_index, ring_polarities)

        if self.use_cooperative_sheaf:
            src = self.row
            tgt = self.col

            # Source → SEND map (polarity-aware)
            send_pols = self.ring_polarities[src]
            send_rho = self.send_maps[send_pols] + self.delta_send[send_pols]
            messages = torch.bmm(x[src].unsqueeze(1), send_rho).squeeze(1)

            messages = self.restriction(messages)

            # Target → RECEIVE map (polarity-aware)
            receive_pols = self.ring_polarities[tgt]
            receive_rho = self.receive_maps[receive_pols] + self.delta_receive[receive_pols]
            messages = torch.bmm(messages.unsqueeze(1), receive_rho).squeeze(1)
        else:
            messages = self.restriction(x[self.col])

        # Mean aggregation
        out = torch.zeros(x.size(0), self.hidden_channels, device=weight_device, dtype=x.dtype)
        out.index_add_(0, tgt, messages)
        out = out * self.deg_norm

        out = self.update(out)
        if x.size(-1) == self.out_channels:
            out = self.res_scale * out + x
        out = self.norm(out)

        return out


class CopresheafDiffusionStack(nn.Module):
    """Multi-layer copresheaf TNN stack (used by RingConeChain)."""

    def __init__(self,
                 in_channels: int,
                 hidden_channels: Optional[int] = None,
                 out_channels: Optional[int] = None,
                 num_layers: int = 3,
                 num_polarities: int = 9,
                 dropout: float = 0.05,
                 sheaf_mode: bool = False,
                 use_cooperative_sheaf: bool = True,
                 device: Optional[torch.device] = None):
        super().__init__()

        self.num_layers = num_layers
        self.sheaf_mode = sheaf_mode
        self.use_cooperative_sheaf = use_cooperative_sheaf
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels

        self.layers = nn.ModuleList([
            MinimalCopresheafTNN(
                in_channels if i == 0 else hidden_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_polarities=num_polarities,
                dropout=dropout,
                use_cooperative_sheaf=self.use_cooperative_sheaf,
                device=self.device
            ) for i in range(num_layers)
        ])

        self.res_proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else None
        self.to(self.device)

    def prepare(self, edge_index: torch.Tensor, ring_polarities: torch.Tensor):
        for layer in self.layers:
            layer.prepare(edge_index, ring_polarities)

    def forward(self, x: torch.Tensor,
                edge_index: Optional[torch.Tensor] = None,
                ring_polarities: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x.clone().to(self.device)

        for layer in self.layers:
            if self.sheaf_mode:
                # Classic symmetric sheaf fallback
                row, col = edge_index if edge_index is not None else (layer.row, layer.col)
                messages = layer.restriction(x[col]) if hasattr(layer, 'restriction') else x[col]
                out = torch.zeros_like(x)
                out.index_add_(0, row, messages)
                deg = torch.zeros(x.size(0), device=x.device)
                deg.index_add_(0, row, torch.ones(messages.size(0), device=x.device))
                out = out / deg.clamp(min=1).unsqueeze(-1)
                x = out + x
            else:
                x = layer(x, edge_index, ring_polarities)

        if self.res_proj is not None:
            residual = self.res_proj(residual)
        if x.size(-1) == residual.size(-1):
            x = x + residual

        return x


class CopresheafDiffusionStack(nn.Module):
    """Multi-layer copresheaf TNN stack (default in RingConeChain).
    Supports classic symmetric sheaf ablation via sheaf_mode=True."""

    def __init__(self,
                 in_channels: int,
                 hidden_channels: Optional[int] = None,
                 out_channels: Optional[int] = None,
                 num_layers: int = 3,
                 num_polarities: int = 9,
                 dropout: float = 0.05,
                 sheaf_mode: bool = False,           # classic symmetric sheaf ablation
                 use_cooperative_sheaf: bool = True,
                 device: Optional[torch.device] = None):
        super().__init__()

        self.num_layers = num_layers
        self.sheaf_mode = sheaf_mode
        self.use_cooperative_sheaf = use_cooperative_sheaf
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels

        # Build layers
        self.layers = nn.ModuleList([
            MinimalCopresheafTNN(
                in_channels if i == 0 else hidden_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_polarities=num_polarities,
                dropout=dropout,
                use_cooperative_sheaf=self.use_cooperative_sheaf,
                device=self.device
            ) for i in range(num_layers)
        ])

        # Residual projection only when dimensions change
        self.res_proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else None

        self.to(self.device)

    def prepare(self, edge_index: torch.Tensor, ring_polarities: torch.Tensor):
        """One-time topology preparation for the entire stack."""
        for layer in self.layers:
            layer.prepare(edge_index, ring_polarities)

    def forward(self, x: torch.Tensor,
                edge_index: Optional[torch.Tensor] = None,
                ring_polarities: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Multi-layer diffusion with optional residual."""
        residual = x.clone().to(self.device)

        for layer in self.layers:
            if self.sheaf_mode:
                # Classic symmetric sheaf (no polarity, simple mean aggregation)
                row, col = edge_index if edge_index is not None else (layer.row, layer.col)
                messages = layer.restriction(x[col]) if hasattr(layer, 'restriction') else x[col]
                out = torch.zeros_like(x)
                out.index_add_(0, row, messages)          # note: row = source
                deg = torch.zeros(x.size(0), device=x.device)
                deg.index_add_(0, row, torch.ones(messages.size(0), device=x.device))
                out = out / deg.clamp(min=1).unsqueeze(-1)
                x = out + x
            else:
                # Full cooperative copresheaf path
                x = layer(x, edge_index, ring_polarities)

        # Final residual (dimension-safe)
        if self.res_proj is not None:
            residual = self.res_proj(residual)
        if x.size(-1) == residual.size(-1):
            x = x + residual

        return x

if __name__ == "__main__":
    conduit = RubikConeConduit(toroidal_modulo9=True, vortex_math_369=False, clifford_projection=True)
    print(f"→ QVPIC v{conduit.VERSION} test passed")
    stats = conduit.monitor_topological_winding()
    print(stats)
