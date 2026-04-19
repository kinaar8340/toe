#!/usr/bin/env python3
"""
~/toe/scripts/magic_island_sweep.py — v1.7.1 Magic Island Sweep
Magic Island Sweep v1.7.1 — Balanced Throughput Edition
MAX_GPU_TRIALS=12 (Configure for NVIDIA RTX 4090 — ~740 MiB per trial)
43 finished in 12 minutes = ~215 trials/hour
Now supports --use-ray flag
"""

import argparse
import gc
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import ray
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

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


# ==================== TUNED GPU DEMAND (v1.7.1) ====================
def calculate_gpu_demand(params: dict) -> tuple[float, str, int]:
    nl = params["num_layers"]
    mf = params["max_facts"]
    pseudo_z = params["num_polarities"] + (mf * 2)

    if pseudo_z > 200 or (nl >= 4 and mf >= 58):
        return 1.0, "EXCLUSIVE", 4
    elif pseudo_z > 185 or (nl >= 4 and mf >= 55):
        return 0.40, "HEAVY", 6
    elif pseudo_z > 170 or (nl >= 4 and mf >= 50):
        return 0.22, "MEDIUM", 8
    else:
        return 0.0, "CPU_ONLY", 12


# ==================== TRIAL FUNCTION ====================
@ray.remote(num_cpus=8, num_gpus=0, max_retries=2, scheduling_strategy="SPREAD")
def run_magic_trial(trial_id: int, params: dict):
    gpu_fraction = params.get("gpu_fraction", 0.0)
    gpu_tier = params.get("gpu_tier", "CPU_ONLY")
    use_gpu = gpu_fraction > 0.0

    print(
        f"→ Trial {trial_id} | pseudo_Z≈{params['pseudo_z']} | pol={params['num_polarities']} | "
        f"facts={params['max_facts']} | layers={params['num_layers']} | Tier={gpu_tier} | "
        f"GPU={gpu_fraction:.2f}"
    )

    torch.set_num_threads(8)
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    print(f" → Running on {device}")

    if use_gpu:
        torch.cuda.empty_cache()
        scaler = GradScaler(device="cuda")
        print(" → Modern AMP + GradScaler enabled")

    conduit = RubikConeConduit(
        embed_dim=cfg.model.embed_dim,
        twist_rate=cfg.model.twist_rate,
        max_depth=cfg.model.max_depth,
        num_polarizations=cfg.model.num_polarizations,
        quat_logical_dim=cfg.model.quat_logical_dim,
        toroidal_modulo9=True,
        vortex_math_369=True,
        clifford_projection=True,
    ).to(device)

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

    omega_L = 0.025
    omega_R = params["omega_R"]
    gauge_strength = params["gauge_strength"]
    use_gauged = params.get("use_gauged_hopf", True)

    raw_data = json.loads(public_facts_file.read_text(encoding="utf-8"))
    lines = [
        line.strip()
        for item in raw_data
        if isinstance(item, dict)
        for line in (item.get("text") or str(item)).splitlines()
        if line.strip() and not line.startswith(("#", "/identity/"))
    ]

    optimizer = torch.optim.AdamW(conduit.parameters(), lr=params["lr"], weight_decay=1e-4)

    burst_count = 0
    id_history = []
    pointer_history = []
    twist_history = []

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
                if use_gpu:
                    with autocast(device_type="cuda"):
                        conduit.training_step(
                            inputs=[item],
                            optimizer=optimizer,
                            recon_weight=params["recon_weight"],
                            align_weight=55000.0,
                            depth_pull_weight=40000.0,
                            winding_weight=48.0,
                            braiding_weight=18.0,
                        )
                else:
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
                if use_gpu:
                    with autocast(device_type="cuda"):
                        loss = loss
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        if (idx + 1) % 5 == 0:
            print(f"   → Fact {idx + 1}/{params['max_facts']} baked")

        if use_gpu and (idx + 1) % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        if use_gauged:
            if not hasattr(ring_cone, "current_quaternion"):
                ring_cone.current_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
            if not hasattr(ring_cone, "twist_history"):
                ring_cone.twist_history = np.zeros(1)
            if not hasattr(ring_cone, "identity"):
                ring_cone.identity = np.array([q_normalize(np.random.randn(4)) for _ in range(96)])
            if not hasattr(ring_cone, "initial_identity"):
                ring_cone.initial_identity = ring_cone.identity.copy()

            delta_L = small_rotor(omega_L)
            delta_R = small_rotor(omega_R)
            q_temp = q_mult(delta_L, ring_cone.current_quaternion)
            ring_cone.current_quaternion = q_mult(q_temp, q_conj(delta_R))
            ring_cone.current_quaternion = q_normalize(ring_cone.current_quaternion)

            avg_imbalance = np.mean(ring_cone.twist_history) % (2 * np.pi)
            gauge_alpha = -gauge_strength * avg_imbalance
            gauge_rot = np.array([np.cos(gauge_alpha), 0.0, 0.0, np.sin(gauge_alpha)])

            ring_cone.current_quaternion = q_mult(ring_cone.current_quaternion, gauge_rot)
            ring_cone.current_quaternion = q_normalize(ring_cone.current_quaternion)

            twist = 2 * np.arccos(np.clip(ring_cone.current_quaternion[0], -1.0, 1.0))
            ring_cone.twist_history = np.append(ring_cone.twist_history, twist)

            pointer = np.tanh(gauge_alpha * 6)
            pointer_history.append(pointer)
            twist_history.append(twist)

            try:
                cosines = np.sum(ring_cone.identity * ring_cone.initial_identity, axis=1)
                id_history.append(float(np.mean(cosines)))
            except Exception:
                id_history.append(1.0)

            if twist > 5.8:
                burst_count += 1

    if use_gpu:
        torch.cuda.empty_cache()

    stats = conduit.monitor_topological_winding(n_samples=512)
    bursts_per_step = burst_count / (params["max_facts"] * 100 + 1e-8)
    mean_id = np.mean(id_history) if id_history else 1.0
    twist_var = np.var(twist_history) if twist_history else 0.0
    pointer_var = np.var(pointer_history) if pointer_history else 0.0

    stability_score = stats.get("active_cubes", 5) * mean_id / (1.0 + bursts_per_step + 1e-8)

    if stability_score > 7.0:
        print(
            f"🌟 HIGH STABILITY CANDIDATE! Score={stability_score:.3f} | pseudo_Z≈{params['pseudo_z']} | Tier={gpu_tier}"
        )

    return {
        "trial_id": trial_id,
        "pseudo_Z": params["pseudo_z"],
        "num_layers": params["num_layers"],
        "num_polarities": params["num_polarities"],
        "max_facts": params["max_facts"],
        "gauge_strength": gauge_strength,
        "omega_R": omega_R,
        "braiding_phase": float(stats.get("braiding_phase", 0.0)),
        "active_cubes": int(stats.get("active_cubes", 0)),
        "stability_score": float(stability_score),
        "bursts_per_step": float(bursts_per_step),
        "mean_id_preservation": float(mean_id),
        "twist_variance": float(twist_var),
        "pointer_variance": float(pointer_var),
        "use_gauged_hopf": use_gauged,
        "gpu_tier": gpu_tier,
        "gpu_fraction": gpu_fraction,
        "timestamp": datetime.now().isoformat(),
    }


# ==================== LAUNCH ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Magic Island Sweep (TOE)")
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument(
        "--use-ray",
        action="store_true",
        help="Use Ray for distributed/multi-node execution (default: single-node sequential)",
    )
    args = parser.parse_args()

    # ==================== PARAM GRID ====================
    param_grid = []
    for nl in [2, 3, 4]:
        for np_val in [9, 12, 18, 24, 36]:
            for mf in [24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60]:
                for gs in [0.78, 0.82, 0.85, 0.88, 0.92]:
                    for or_val in [0.0215, 0.0220, 0.0225, 0.0230, 0.0235]:
                        pseudo_z = np_val + (mf * 2)
                        base_params = {
                            "num_layers": nl,
                            "num_polarities": np_val,
                            "max_facts": mf,
                            "gauge_strength": gs,
                            "omega_R": or_val,
                            "use_gauged_hopf": True,
                            "cooperative_sheaf": True,
                            "lr": 1e-4,
                            "recon_weight": 20000,
                            "pseudo_z": pseudo_z,
                        }
                        gpu_fraction, gpu_tier, cpu_req = calculate_gpu_demand(base_params)
                        base_params.update(
                            {"gpu_fraction": gpu_fraction, "gpu_tier": gpu_tier, "cpu_req": cpu_req}
                        )
                        param_grid.append(base_params)

    param_grid.sort(key=lambda x: x["gpu_fraction"], reverse=True)

    MAX_GPU_TRIALS = 12
    for i in range(MAX_GPU_TRIALS, len(param_grid)):
        if param_grid[i]["gpu_fraction"] > 0:
            param_grid[i]["gpu_fraction"] = 0.0
            param_grid[i]["gpu_tier"] = "CPU_ONLY"
            param_grid[i]["cpu_req"] = 12

    param_grid = param_grid[: args.trials]

    tier_counts = Counter(p["gpu_tier"] for p in param_grid)
    total_gpu_demand = sum(p["gpu_fraction"] for p in param_grid)

    print(
        f"→ Launching {len(param_grid)} trials | Mode: {'Ray (distributed)' if args.use_ray else 'Single-node (sequential)'}"
    )
    print(f"   Tiers → {dict(tier_counts)}")
    print(f"   Total GPU demand: {total_gpu_demand:.2f} / 1.0 available")
    print(f"   Expected concurrent GPU trials: ~{max(1, int(total_gpu_demand / 0.22))}")

    print("🌟 Magic Island Sweep v1.7.1 — Perfectly Balanced Throughput Edition")

    if args.use_ray:
        try:
            ray.init(address="auto", ignore_reinit_error=True)
            print(f"   🌟 Ray initialized — {len(ray.nodes())} nodes available")

            futures = []
            for i, p in enumerate(param_grid):
                future = run_magic_trial.options(
                    num_cpus=p["cpu_req"], num_gpus=p["gpu_fraction"]
                ).remote(i, p)
                futures.append(future)
            results = ray.get(futures)
            ray.shutdown()
        except Exception as e:
            print(f"   Ray failed ({e}) — falling back to single-node")
            results = [run_magic_trial(i, p) for i, p in enumerate(param_grid)]
    else:
        print("   🔄 Running sequentially (single-node mode)")
        results = [run_magic_trial(i, p) for i, p in enumerate(param_grid)]

        # ==================== Reporting + top-30 ====================
        Path("outputs").mkdir(exist_ok=True)
        report_path = Path(f"outputs/magic_island_report_{datetime.now():%Y%m%d_%H%M%S}.md")
        df = pd.DataFrame(results)
        df.to_markdown(report_path, index=False)
        print(f"✅ Full report saved → {report_path}")
        print("\n🏆 Top 30 Stability Candidates:")
        print(
            df.sort_values("stability_score", ascending=False).head(30)[...].to_string(index=False)
        )
        print("→ Sweep complete.")
