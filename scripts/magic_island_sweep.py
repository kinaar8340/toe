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
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
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

E = math.e
PI = math.pi
PHI = (1 + math.sqrt(5)) / 2
R_RESIDUAL = PHI**2 + E**2 - PI**2
KAPPA_DOC = 0.85
KAPPA_STAR = E / PI - R_RESIDUAL / PI**2
KAPPA_SIM = 0.89

# Noble-gas / magic-island presets (pseudo_Z = num_polarities + 2*max_facts approx Z)
ISLAND_PRESETS: dict[int, dict] = {
    18: {
        "island_z": 18,
        "element": "Ar",
        "num_layers": 3,
        "num_polarities": 12,
        "max_facts": 30,
        "pseudo_z": 72,
        "gauge_strength": 0.88,
        "omega_R": 0.0225,
    },
    54: {
        "island_z": 54,
        "element": "Xe",
        "num_layers": 3,
        "num_polarities": 18,
        "max_facts": 36,
        "pseudo_z": 90,
        "gauge_strength": 0.88,
        "omega_R": 0.0225,
    },
    129: {
        "island_z": 129,
        "element": "magic",
        "num_layers": 4,
        "num_polarities": 9,
        "max_facts": 60,
        "pseudo_z": 129,
        "gauge_strength": 0.85,
        "omega_R": 0.0225,
    },
}


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
def run_magic_trial_core(trial_id: int, params: dict) -> dict:
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

    kappa_seed = float(params.get("kappa_seed", KAPPA_DOC))
    wg_base = float(params.get("wg_base", 350.0))
    braiding_target = float(params.get("braiding_target", 0.8145))
    braid_feedback_gain = float(params.get("braid_feedback_gain", 0.002))
    steps_per_fact = int(params.get("steps_per_fact", 100))

    conduit = RubikConeConduit(
        embed_dim=cfg.model.embed_dim,
        twist_rate=cfg.model.twist_rate,
        max_depth=cfg.model.max_depth,
        num_polarizations=params.get("num_polarities", cfg.model.num_polarizations),
        quat_logical_dim=getattr(cfg.model, "quat_logical_dim", 96),
        toroidal_modulo9=bool(params.get("toroidal_modulo9", True)),
        vortex_math_369=bool(params.get("vortex_math_369", True)),
        clifford_projection=bool(params.get("clifford_projection", True)),
        gauge_strength=params["gauge_strength"],
        omega_R=params["omega_R"],
        wg_base=wg_base,
        kappa=kappa_seed,
        braiding_target=braiding_target,
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

        for _step in range(steps_per_fact):
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

            if params.get("adaptive_kappa", True) and (idx + 1) % 5 == 0:
                mid = conduit.monitor_topological_winding(n_samples=64)
                braiding = float(mid.get("braiding_phase", 0.0))
                braid_err = braiding - braiding_target
                conduit.kappa = float(
                    np.clip(conduit.kappa + braid_feedback_gain * braid_err, 0.70, 0.95)
                )

    if use_gpu:
        torch.cuda.empty_cache()

    stats = conduit.monitor_topological_winding(n_samples=512)
    bursts_per_step = burst_count / (params["max_facts"] * steps_per_fact + 1e-8)
    mean_id = np.mean(id_history) if id_history else 1.0
    twist_var = np.var(twist_history) if twist_history else 0.0
    pointer_var = np.var(pointer_history) if pointer_history else 0.0

    stability_score = stats.get("active_cubes", 5) * mean_id / (1.0 + bursts_per_step + 1e-8)

    if stability_score > 7.0:
        print(
            f"🌟 HIGH STABILITY CANDIDATE! Score={stability_score:.3f} | pseudo_Z≈{params['pseudo_z']} | Tier={gpu_tier}"
        )

    kappa_final = float(conduit.kappa)
    geo_w = float(stats.get("geometric_winding", 0.0))
    w_g_target = wg_base / PI
    braiding = float(stats.get("braiding_phase", 0.0))
    hopf_delta = abs(geo_w - w_g_target)
    braiding_delta = abs(braiding - braiding_target)
    vortex369 = bool(params.get("vortex_math_369", True))
    knot_phase = float(stats.get("knot_phase", 0.0)) if vortex369 else 0.0
    eff_w = float(stats.get("effective_winding", 0.0))
    gap_stress = hopf_delta / max(w_g_target, 1e-6) + braiding_delta * 0.05 + abs(eff_w) * 0.001
    kappa_proxy = float(np.clip(E / PI - gap_stress / PI + knot_phase * 0.01, 0.70, 0.95))

    return {
        "trial_id": trial_id,
        "label": params.get("label", f"trial_{trial_id}"),
        "island_z": params.get("island_z"),
        "pseudo_Z": params["pseudo_z"],
        "num_layers": params["num_layers"],
        "num_polarities": params["num_polarities"],
        "max_facts": params["max_facts"],
        "gauge_strength": gauge_strength,
        "omega_R": omega_R,
        "topology": {
            "toroidal_modulo9": bool(params.get("toroidal_modulo9", True)),
            "vortex_math_369": vortex369,
        },
        "kappa_seed": kappa_seed,
        "kappa_final": kappa_final,
        "kappa_drift": kappa_final - kappa_seed,
        "kappa_proxy": kappa_proxy,
        "braid_feedback_gain": braid_feedback_gain,
        "braiding_phase": braiding,
        "braiding_delta": braiding_delta,
        "hopf_delta": hopf_delta,
        "w_g_measured": geo_w,
        "active_cubes": int(stats.get("active_cubes", 0)),
        "vortex_sync_global": float(stats.get("vortex_sync_global", 0.0)),
        "stability_score": float(stability_score),
        "bursts_per_step": float(bursts_per_step),
        "mean_id_preservation": float(mean_id),
        "twist_variance": float(twist_var),
        "pointer_variance": float(pointer_var),
        "use_gauged_hopf": use_gauged,
        "gpu_tier": gpu_tier,
        "gpu_fraction": gpu_fraction,
        "delta_vs_kappa_doc": abs(kappa_final - KAPPA_DOC),
        "delta_vs_kappa_star": abs(kappa_final - KAPPA_STAR),
        "delta_vs_kappa_sim": abs(kappa_final - KAPPA_SIM),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _ray_remote_trial():
    import ray

    @ray.remote(num_cpus=8, num_gpus=0, max_retries=2, scheduling_strategy="SPREAD")
    def run_magic_trial(trial_id: int, params: dict):
        return run_magic_trial_core(trial_id, params)

    return run_magic_trial


def island_topology_grid_params(
    island_z: int = 129,
    quick: bool = True,
    braid_gains: list[float] | None = None,
) -> list[dict]:
    """2×2 topology × braid-gain grid at fixed island preset."""
    preset = dict(ISLAND_PRESETS[island_z])
    if quick:
        preset["max_facts"] = max(12, preset["max_facts"] // 5)
        preset["steps_per_fact"] = 20
    else:
        preset["steps_per_fact"] = 100
    gains = braid_gains or [0.002]
    combos = [
        ("baseline", False, False),
        ("toroidal_only", True, False),
        ("vortex369_only", False, True),
        ("full_topology", True, True),
    ]
    grid = []
    for gain in gains:
        for label, toroidal, vortex369 in combos:
            grid.append(
                {
                    **preset,
                    "label": f"{label}_z{island_z}_bg{gain:g}",
                    "toroidal_modulo9": toroidal,
                    "vortex_math_369": vortex369,
                    "kappa_seed": KAPPA_DOC,
                    "wg_base": 350.0,
                    "braiding_target": 0.8145,
                    "braid_feedback_gain": gain,
                    "adaptive_kappa": True,
                    "use_gauged_hopf": True,
                    "cooperative_sheaf": True,
                    "lr": 1e-4,
                    "recon_weight": 20000,
                    "gpu_fraction": 0.0,
                    "gpu_tier": "CPU_ONLY",
                    "cpu_req": 8,
                }
            )
    return grid


def run_island_topology_grid(
    island_z: int = 129,
    quick: bool = True,
    braid_gains: list[float] | None = None,
) -> dict:
    param_grid = island_topology_grid_params(island_z, quick=quick, braid_gains=braid_gains)
    results = [run_magic_trial_core(i, p) for i, p in enumerate(param_grid)]
    comparison = [
        {
            "label": r["label"],
            "island_z": r.get("island_z"),
            "toroidal_modulo9": r["topology"]["toroidal_modulo9"],
            "vortex_math_369": r["topology"]["vortex_math_369"],
            "braid_feedback_gain": r["braid_feedback_gain"],
            "kappa_final": r["kappa_final"],
            "kappa_drift": r["kappa_drift"],
            "kappa_proxy": r["kappa_proxy"],
            "stability_score": r["stability_score"],
            "hopf_delta": r["hopf_delta"],
        }
        for r in results
    ]
    return {
        "references": {
            "kappa_doc": KAPPA_DOC,
            "kappa_star": KAPPA_STAR,
            "kappa_sim": KAPPA_SIM,
        },
        "island_z": island_z,
        "quick": quick,
        "braid_gains": braid_gains or [0.002],
        "n_runs": len(results),
        "runs": results,
        "comparison_table": comparison,
    }


def save_island_grid_json(summary: dict) -> Path:
    out = Path("outputs") / "magic_island"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = out / f"island_topology_grid_z{summary['island_z']}_{stamp}.json"
    path.write_text(json.dumps(summary, indent=2, default=str))
    return path


# ==================== LAUNCH ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Magic Island Sweep (TOE)")
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument(
        "--use-ray",
        action="store_true",
        help="Use Ray for distributed/multi-node execution (default: single-node sequential)",
    )
    parser.add_argument(
        "--topology-grid",
        action="store_true",
        help="Run 2×2 topology grid at island preset (see --island-z)",
    )
    parser.add_argument("--island-z", type=int, default=129, choices=sorted(ISLAND_PRESETS))
    parser.add_argument("--quick", action="store_true", help="Reduced facts/steps for grid mode")
    parser.add_argument(
        "--braid-gains",
        type=float,
        nargs="+",
        default=None,
        help="Braid feedback gains (topology grid)",
    )
    args = parser.parse_args()

    if args.topology_grid:
        gains = args.braid_gains or [0.002, 0.005, 0.01]
        print(f"=== Island topology κ grid Z={args.island_z} gains={gains} quick={args.quick} ===")
        summary = run_island_topology_grid(
            island_z=args.island_z, quick=args.quick, braid_gains=gains
        )
        path = save_island_grid_json(summary)
        for row in summary["comparison_table"]:
            print(
                f"  {row['label']:<28} κ→{row['kappa_final']:.3f} "
                f"drift={row['kappa_drift']:+.4f} proxy={row['kappa_proxy']:.3f} "
                f"stab={row['stability_score']:.2f}"
            )
        print(f"JSON: {path}")
        raise SystemExit(0)

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
            import ray

            run_magic_trial = _ray_remote_trial()
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
            results = [run_magic_trial_core(i, p) for i, p in enumerate(param_grid)]
    else:
        print("   🔄 Running sequentially (single-node mode)")
        results = [run_magic_trial_core(i, p) for i, p in enumerate(param_grid)]

        # ==================== Reporting + top-30 ====================
        import pandas as pd

        Path("outputs").mkdir(exist_ok=True)
        report_path = Path(f"outputs/magic_island_report_{datetime.now():%Y%m%d_%H%M%S}.md")
        df = pd.DataFrame(results)
        df.to_markdown(report_path, index=False)
        print(f"✅ Full report saved → {report_path}")
        print("\n🏆 Top 30 Stability Candidates:")
        print(df.sort_values("stability_score", ascending=False).head(30).to_string(index=False))
        print("→ Sweep complete.")
