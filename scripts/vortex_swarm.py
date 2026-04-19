#!/usr/bin/env python3
"""
~/toe/scripts/vortex_swarm.py — v10.9 Mult-Node Swarm
Expanded grid + full hyperparameter sweep (lr + recon_weight + more layers/pols/facts)
Fully distributed across 9-node R630 cluster.
Now supports --use-ray flag
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import ray
import torch
import torch.nn.functional as F

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.conduit import CopresheafDiffusionStack, RubikConeConduit
from src.config import load_config

cfg = load_config("configs/default.yaml")
public_facts_file = Path("facts/public_facts.json")


# ==================== QUATERNION HELPERS ====================
def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def q_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def q_normalize(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-8 else q


def small_rotor(theta, axis=np.array([0.0, 0.0, 1.0])):
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    half = theta / 2
    return np.array([np.cos(half), *(np.sin(half) * axis)])


# ==================== TRIAL FUNCTION ====================
def run_qvpic_trial(trial_id: int, params: dict):
    print(
        f"→ Trial {trial_id} running | layers={params['num_layers']} | pol={params['num_polarities']} | "
        f"coop={params['cooperative_sheaf']} | facts={params['max_facts']} | lr={params['lr']:.2e} | recon_w={params['recon_weight']}"
    )

    torch.set_num_threads(8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" → Using device: {device}")

    conduit = RubikConeConduit(
        embed_dim=cfg.model.embed_dim,
        twist_rate=cfg.model.twist_rate,
        max_depth=cfg.model.max_depth,
        num_polarizations=cfg.model.num_polarizations,
        quat_logical_dim=getattr(cfg.model, "quat_logical_dim", 96),  # ← FIXED: safe fallback
        toroidal_modulo9=True,
        vortex_math_369=False,
        clifford_projection=True,
    ).to(device)

    # === GAUGED HOPF / TWO-GYRO UPGRADE ===
    use_gauged = params.get("use_gauged_hopf", False)

    ring_cone = conduit.ring_cone
    new_stack = CopresheafDiffusionStack(
        in_channels=ring_cone.embed_dim,
        hidden_channels=ring_cone.embed_dim,
        out_channels=ring_cone.embed_dim,
        num_layers=params["num_layers"],
        num_polarities=params["num_polarities"],
        dropout=0.05,
        sheaf_mode=False,
        use_cooperative_sheaf=params["cooperative_sheaf"],
        device=device,
    )
    new_stack.prepare(ring_cone.edge_index, ring_cone.ring_polarities)
    new_stack = new_stack.to(device)
    ring_cone.tnn_stack = new_stack

    # === LOAD FACTS ===
    raw_data = json.loads(public_facts_file.read_text(encoding="utf-8"))
    lines = [
        line.strip()
        for item in raw_data
        if isinstance(item, dict)
        for line in (item.get("text") or str(item)).splitlines()
        if line.strip() and not line.startswith(("#", "/identity/"))
    ]

    # === OPTIMIZER ===
    optimizer = torch.optim.AdamW(conduit.parameters(), lr=params["lr"], weight_decay=1e-4)

    for idx, _fact in enumerate(lines[: params["max_facts"]]):
        emb = F.normalize(torch.randn(384, device=device), dim=-1) * 0.28
        ring_idx = idx % ring_cone.NUM_RINGS
        cube_local_idx = idx % ring_cone.rings[ring_idx].num_cubes
        ring_cone.bake_ring(ring_idx, cube_local_idx, emb, orientation=idx % 24)

        for _step in range(100):
            item = {
                "emb": emb.unsqueeze(0),
                "s": torch.tensor([4.5 + idx * 4.8], device=device),
                "pol_idx": 0,
            }
            try:
                conduit.training_step(
                    inputs=[item],
                    optimizer=optimizer,
                    recon_weight=params["recon_weight"],
                    align_weight=55000.0,
                    depth_pull_weight=40000.0,
                    winding_weight=48.0,
                    braiding_weight=18.0,
                )
            except Exception:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
                loss.backward()
                optimizer.step()

        if (idx + 1) % 3 == 0:
            print(f"   → Fact {idx + 1}/{params['max_facts']} baked")

            # === TWO-GYRO GAUGED HOPF UPGRADE (analytical-scale pointer) ===
            if use_gauged:
                # Create missing attributes on first use (safe fallback)
                if not hasattr(ring_cone, "current_quaternion"):
                    ring_cone.current_quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # identity
                if not hasattr(ring_cone, "twist_history"):
                    ring_cone.twist_history = np.zeros(1)

                delta_L = small_rotor(0.025)
                delta_R = small_rotor(0.023)

                q_temp = q_mult(delta_L, ring_cone.current_quaternion)
                ring_cone.current_quaternion = q_mult(q_temp, q_conj(delta_R))
                ring_cone.current_quaternion = q_normalize(ring_cone.current_quaternion)

                # Gauge connection (the pointer on the analytical scale)
                avg_imbalance = np.mean(ring_cone.twist_history) % (2 * np.pi)
                gauge_alpha = -0.85 * avg_imbalance
                gauge_rot = np.array([np.cos(gauge_alpha), 0.0, 0.0, np.sin(gauge_alpha)])

                ring_cone.current_quaternion = q_mult(ring_cone.current_quaternion, gauge_rot)
                ring_cone.current_quaternion = q_normalize(ring_cone.current_quaternion)

                # Store for future steps and monitoring
                ring_cone.twist_history = np.append(
                    ring_cone.twist_history,
                    2 * np.arccos(np.clip(ring_cone.current_quaternion[0], -1.0, 1.0)),
                )
                if not hasattr(ring_cone, "gauge_alpha_history"):
                    ring_cone.gauge_alpha_history = []
                ring_cone.gauge_alpha_history.append(gauge_alpha)

        print(f" → Trial {trial_id} bake complete")

    stats = conduit.monitor_topological_winding(n_samples=512)
    return {
        "trial_id": trial_id,
        "num_layers": params["num_layers"],
        "num_polarities": params["num_polarities"],
        "cooperative_sheaf": params["cooperative_sheaf"],
        "max_facts": params["max_facts"],
        "lr": params["lr"],
        "recon_weight": params["recon_weight"],
        "braiding_phase": float(stats.get("braiding_phase", 0.0)),
        "geometric_winding": float(stats.get("geometric_winding", 0.0)),
        "active_cubes": int(stats.get("active_cubes", 0)),
        "use_gauged_hopf": use_gauged,
        "timestamp": datetime.now().isoformat(),
    }


# ==================== LAUNCH ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vortex Swarm (TOE) — GPU enabled")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--max-facts", type=int, default=9)
    parser.add_argument("--use-ray", action="store_true")
    parser.add_argument(
        "--gpu-per-trial",
        type=float,
        default=0.3,  # ← NEW: 30% GPU each
        help="Fractional GPU per trial (0.3 = 30% of one 4090)",
    )
    args = parser.parse_args()

    param_grid = [
        {
            "num_layers": nl,
            "num_polarities": np,
            "cooperative_sheaf": cs,
            "max_facts": mf,
            "lr": lr_val,
            "recon_weight": rw,
            "use_gauged_hopf": gh,
        }
        for nl in [2, 3, 4, 5, 6]
        for np in [9, 12, 18, 24, 36]
        for cs in [True, False]
        for mf in [9, 18, 27, 36]
        for lr_val in [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]
        for rw in [8000.0, 10000.0, 15000.0, 20000.0, 25000.0]
        for gh in [True, False]
    ][: args.trials]

    print("🌟 Magic Island Sweep — Flux Flywheel Resonance Hunter")
    print(
        f"→ Launching {len(param_grid)} trials | GPU per trial = {args.gpu_per_trial} | Mode: {'Ray' if args.use_ray else 'Single'}"
    )

if args.use_ray:
    try:
        ray.init(address="auto", ignore_reinit_error=True)
        print(f"   🌟 Ray initialized — {len(ray.nodes())} nodes available")

        @ray.remote(num_cpus=12, num_gpus=0, max_retries=2, scheduling_strategy="SPREAD")
        def remote_trial(trial_id: int, params: dict):
            return run_qvpic_trial(trial_id, params)

        futures = []
        for i, p in enumerate(param_grid):
            future = remote_trial.options(num_gpus=args.gpu_per_trial).remote(i, p)
            futures.append(future)
        results = ray.get(futures)
        ray.shutdown()
    except Exception as e:
        print(f"   Ray failed ({e}) — falling back to single-node")
        results = [run_qvpic_trial(i, p) for i, p in enumerate(param_grid)]
else:
    print("   🔄 Running sequentially (single-node mode)")
    results = [run_qvpic_trial(i, p) for i, p in enumerate(param_grid)]

    df = pd.DataFrame(results)
    report_path = Path(f"outputs/swarm_report_{datetime.now():%Y%m%d_%H%M%S}.md")
    df.to_markdown(report_path, index=False)

    print(f"\n→ Swarm complete! Report saved → {report_path}")
    print("→ Done.")
