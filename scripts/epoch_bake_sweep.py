#!/usr/bin/env python3
"""
scripts/epoch_bake_sweep.py — epoch bake sweep with topology flags and κ readout.

Modes:
  Default   — parameter grid sweep (legacy CSV output)
  --topology-grid — 2×2 toroidal_modulo9 × vortex_math_369 comparison at fixed W_g
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "epoch_bake"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

E = math.e
PI = math.pi
PHI = (1 + math.sqrt(5)) / 2
R_RESIDUAL = PHI**2 + E**2 - PI**2
KAPPA_DOC = 0.85
KAPPA_STAR = E / PI - R_RESIDUAL / PI**2
KAPPA_SIM = 0.89
W_G_TARGET = 350.0 / PI


def kappa_references() -> dict[str, float]:
    return {
        "kappa_doc": KAPPA_DOC,
        "kappa_star": KAPPA_STAR,
        "kappa_sim": KAPPA_SIM,
        "R_residual": R_RESIDUAL,
        "w_g_target": W_G_TARGET,
    }


def _q_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
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


def _q_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _q_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 1e-8 else q


def _small_rotor_np(theta: float, axis: np.ndarray | None = None) -> np.ndarray:
    axis = np.array([0.0, 0.0, 1.0]) if axis is None else axis / (np.linalg.norm(axis) + 1e-8)
    half = theta / 2
    return np.array([np.cos(half), *(np.sin(half) * axis)])


def _two_gyro_step_np(conduit) -> float:
    """Numpy two-gyro step compatible with RubikConeConduit state."""
    omega_L = float(getattr(conduit, "omega_L", 0.025))
    delta_L = _small_rotor_np(omega_L)
    delta_R = _small_rotor_np(conduit.omega_R)
    q_temp = _q_mult(delta_L, conduit.current_quaternion)
    conduit.current_quaternion = _q_mult(q_temp, _q_conj(delta_R))
    conduit.current_quaternion = _q_normalize(conduit.current_quaternion)
    avg_imbalance = float(np.mean(conduit.twist_history)) % (2 * np.pi)
    gauge_alpha = -conduit.gauge_strength * avg_imbalance
    gauge_rot = np.array([np.cos(gauge_alpha), 0.0, 0.0, np.sin(gauge_alpha)])
    conduit.current_quaternion = _q_mult(conduit.current_quaternion, gauge_rot)
    conduit.current_quaternion = _q_normalize(conduit.current_quaternion)
    twist = 2 * np.arccos(np.clip(conduit.current_quaternion[0], -1.0, 1.0))
    conduit.twist_history = np.append(conduit.twist_history, twist)
    return abs(gauge_alpha)


def _run_bake_loop(
    conduit,
    device: torch.device,
    bake_steps: int,
    wg_base: float,
    adaptive_kappa: bool,
    kappa_trace: list[float],
    gauge_alphas: list[float],
    braid_feedback_gain: float = 0.002,
) -> str:
    """Topology-aware bake: ring updates + two-gyro + helix traversal."""
    w_g_target = wg_base / PI
    try:
        for step in range(bake_steps):
            emb = torch.zeros(1, conduit.embed_dim, device=device)
            conduit._direct_bake(step, emb)
            gauge_alphas.append(_two_gyro_step_np(conduit))

            s = (step + 1) / bake_steps * conduit.max_depth
            pol = step % max(conduit.num_pol, 1)
            with torch.no_grad():
                conduit.get_helix_3d(s, pol_idx=pol)

            if adaptive_kappa and step > 0 and step % 50 == 0:
                mid = conduit.monitor_topological_winding(n_samples=64)
                geo_w = float(mid.get("geometric_winding", w_g_target))
                eff_w = float(mid.get("effective_winding", 0.0))
                braiding = float(mid.get("braiding_phase", 0.0))
                hopf_err = (geo_w - w_g_target) / max(w_g_target, 1e-6)
                braid_err = braiding - float(conduit.braiding_target)
                wind_err = eff_w / max(abs(geo_w), 1e-6)
                delta_k = -0.01 * hopf_err * (E / PI - conduit.kappa)
                delta_k += braid_feedback_gain * braid_err
                delta_k += 0.005 * wind_err * (KAPPA_SIM - conduit.kappa)
                conduit.kappa = float(np.clip(conduit.kappa + delta_k, 0.70, 0.95))
                kappa_trace.append(float(conduit.kappa))
        return "real"
    except Exception as exc:
        return f"fallback:{exc}"


def _safe_stats(stats: dict) -> dict:
    out: dict = {}
    for key, val in stats.items():
        if isinstance(val, (int, float, np.floating, np.integer)):
            out[key] = float(val)
        elif isinstance(val, (bool, np.bool_)):
            out[key] = bool(val)
        else:
            out[key] = str(val)
    return out


def run_epoch_trial(trial_id: int, params: dict) -> dict:
    from src.conduit import RubikConeConduit

    bake_steps = int(params.get("bake_steps", 120))
    kappa_seed = float(params.get("kappa_seed", KAPPA_DOC))
    wg_base = float(params.get("wg_base", 350.0))
    braiding_target = float(params.get("braiding_target", 0.8145))
    toroidal = bool(params.get("toroidal_modulo9", True))
    vortex369 = bool(params.get("vortex_math_369", True))
    adaptive_kappa = bool(params.get("adaptive_kappa", True))
    clifford = bool(params.get("clifford_projection", True))
    braid_feedback_gain = float(params.get("braid_feedback_gain", 0.002))

    label = params.get("label", f"trial_{trial_id}")
    print(
        f"Trial {trial_id} [{label}] toroidal={toroidal} vortex369={vortex369} "
        f"steps={bake_steps} κ_seed={kappa_seed:.3f}"
    )

    conduit = RubikConeConduit(
        num_polarizations=params.get("num_polarities", 18),
        gauge_strength=params["gauge_strength"],
        omega_R=params["omega_R"],
        toroidal_modulo9=toroidal,
        vortex_math_369=vortex369,
        clifford_projection=clifford,
        wg_base=wg_base,
        kappa=kappa_seed,
        braiding_target=braiding_target,
    )
    if hasattr(conduit, "num_layers"):
        conduit.num_layers = int(params.get("num_layers", 3))
    if hasattr(conduit, "max_facts"):
        conduit.max_facts = int(params.get("max_facts", 48))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conduit.to(device)
    print(f"   Conduit v{conduit.VERSION} on {device}")

    stats_before = _safe_stats(conduit.monitor_topological_winding(n_samples=128))
    w_g_target = wg_base / PI

    gauge_alphas: list[float] = []
    kappa_trace: list[float] = [kappa_seed]
    bake_mode = _run_bake_loop(
        conduit,
        device,
        bake_steps,
        wg_base,
        adaptive_kappa,
        kappa_trace,
        gauge_alphas,
        braid_feedback_gain=braid_feedback_gain,
    )
    if bake_mode != "real":
        print(f"   Bake fallback: {bake_mode}")

    stats_after = _safe_stats(conduit.monitor_topological_winding(n_samples=512))
    kappa_final = float(conduit.kappa)
    geo_w = float(stats_after.get("geometric_winding", 0.0))
    braiding = float(stats_after.get("braiding_phase", 0.0))
    hopf_delta = abs(geo_w - w_g_target)
    braiding_delta = abs(braiding - braiding_target)

    # Holonomy-gap κ proxy: invert B(κ)=π²(e/π−κ) using winding+braiding stress
    eff_w = float(stats_after.get("effective_winding", 0.0))
    knot_phase = float(stats_after.get("knot_phase", 0.0)) if vortex369 else 0.0
    gap_stress = hopf_delta / max(w_g_target, 1e-6) + braiding_delta * 0.05 + abs(eff_w) * 0.001
    kappa_proxy = float(np.clip(E / PI - gap_stress / PI + knot_phase * 0.01, 0.70, 0.95))

    ring_stats = {
        k: stats_after[k]
        for k in ("active_cubes", "vortex_sync_global", "stability_score", "bursts_per_step")
        if k in stats_after
    }
    stability_score = float(ring_stats.get("stability_score", 8.0 - hopf_delta * 10))

    result = {
        "trial_id": trial_id,
        "label": label,
        "topology": {
            "toroidal_modulo9": toroidal,
            "vortex_math_369": vortex369,
            "clifford_projection": clifford,
        },
        "kappa_seed": kappa_seed,
        "kappa_final": kappa_final,
        "kappa_drift": kappa_final - kappa_seed,
        "kappa_proxy": kappa_proxy,
        "kappa_trace_last": kappa_trace[-5:],
        "wg_base": wg_base,
        "w_g_target": w_g_target,
        "w_g_measured": geo_w,
        "hopf_delta": hopf_delta,
        "braiding_phase": braiding,
        "braiding_target": braiding_target,
        "braiding_delta": braiding_delta,
        "braid_feedback_gain": braid_feedback_gain,
        "stability_score": round(stability_score, 4),
        "bake_steps": bake_steps,
        "bake_mode": bake_mode,
        "gauge_alpha_mean": float(np.mean(gauge_alphas)) if gauge_alphas else None,
        "gauge_alpha_std": float(np.std(gauge_alphas)) if gauge_alphas else None,
        "delta_vs_kappa_doc": abs(kappa_final - KAPPA_DOC),
        "delta_vs_kappa_star": abs(kappa_final - KAPPA_STAR),
        "delta_vs_kappa_sim": abs(kappa_final - KAPPA_SIM),
        "delta_proxy_vs_kappa_doc": abs(kappa_proxy - KAPPA_DOC),
        "delta_proxy_vs_kappa_star": abs(kappa_proxy - KAPPA_STAR),
        "delta_proxy_vs_kappa_sim": abs(kappa_proxy - KAPPA_SIM),
        "stats_before": stats_before,
        "stats_after": stats_after,
        "version": conduit.VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "params": params,
    }

    print(
        f"   Done | κ {kappa_seed:.3f}→{kappa_final:.3f} (proxy {kappa_proxy:.3f}) "
        f"W_g Δ={hopf_delta:.4f} braid Δ={braiding_delta:.4f}"
    )
    return result


def topology_grid_params(
    bake_steps: int = 500,
    braid_feedback_gain: float = 0.002,
) -> list[dict]:
    """2×2 flag grid at meta-opt sweet-spot hyperparameters."""
    sweet = {
        "num_layers": 3,
        "num_polarities": 18,
        "max_facts": 48,
        "gauge_strength": 0.88,
        "omega_R": 0.0225,
        "wg_base": 350.0,
        "kappa_seed": KAPPA_DOC,
        "braiding_target": 0.8145,
        "bake_steps": bake_steps,
        "adaptive_kappa": True,
        "clifford_projection": True,
        "braid_feedback_gain": braid_feedback_gain,
    }
    combos = [
        ("baseline", False, False),
        ("toroidal_only", True, False),
        ("vortex369_only", False, True),
        ("full_topology", True, True),
    ]
    grid = []
    for label, toroidal, vortex369 in combos:
        grid.append(
            {
                **sweet,
                "label": label,
                "toroidal_modulo9": toroidal,
                "vortex_math_369": vortex369,
            }
        )
    return grid


def run_topology_grid(
    bake_steps: int = 500,
    braid_gains: list[float] | None = None,
) -> dict:
    refs = kappa_references()
    gains = braid_gains or [0.002]
    param_grid: list[dict] = []
    for gain in gains:
        for p in topology_grid_params(bake_steps=bake_steps, braid_feedback_gain=gain):
            p = dict(p)
            p["label"] = f"{p['label']}_bg{gain:g}"
            param_grid.append(p)
    results = [run_epoch_trial(i, p) for i, p in enumerate(param_grid)]

    def nearest_kappa(row: dict) -> str:
        deltas = {
            "kappa_doc": row["delta_vs_kappa_doc"],
            "kappa_star": row["delta_vs_kappa_star"],
            "kappa_sim": row["delta_vs_kappa_sim"],
        }
        return min(deltas, key=deltas.get)

    comparison = []
    for row in results:
        comparison.append(
            {
                "label": row["label"],
                "toroidal_modulo9": row["topology"]["toroidal_modulo9"],
                "vortex_math_369": row["topology"]["vortex_math_369"],
                "kappa_seed": row["kappa_seed"],
                "kappa_final": row["kappa_final"],
                "kappa_drift": row["kappa_drift"],
                "kappa_proxy": row["kappa_proxy"],
                "hopf_delta": row["hopf_delta"],
                "braiding_delta": row["braiding_delta"],
                "braid_feedback_gain": row.get("braid_feedback_gain", 0.002),
                "nearest_kappa": nearest_kappa(row),
            }
        )

    best_drift = min(results, key=lambda r: abs(r["kappa_drift"]))
    best_hopf = min(results, key=lambda r: r["hopf_delta"])

    summary = {
        "references": refs,
        "bake_steps": bake_steps,
        "n_runs": len(results),
        "runs": results,
        "comparison_table": comparison,
        "best_kappa_drift": {
            "label": best_drift["label"],
            "kappa_final": best_drift["kappa_final"],
            "kappa_drift": best_drift["kappa_drift"],
        },
        "best_hopf_lock": {
            "label": best_hopf["label"],
            "hopf_delta": best_hopf["hopf_delta"],
            "w_g_measured": best_hopf["w_g_measured"],
        },
    }
    return summary


def save_topology_json(summary: dict) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"topology_kappa_grid_{stamp}.json"
    path.write_text(json.dumps(summary, indent=2, default=str))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Epoch bake sweep with topology κ readout")
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--use-ray", action="store_true")
    parser.add_argument("--dense", action="store_true", help="High-resolution sweet-spot grid")
    parser.add_argument(
        "--topology-grid",
        action="store_true",
        help="Run 2×2 toroidal×vortex369 comparison (4 trials)",
    )
    parser.add_argument("--bake-steps", type=int, default=500, help="Bake steps (topology grid)")
    parser.add_argument(
        "--braid-gains",
        type=float,
        nargs="+",
        default=None,
        help="Braid feedback gains for topology grid (default: 0.002)",
    )
    args = parser.parse_args()

    if args.topology_grid:
        gains = args.braid_gains or [0.002]
        print(f"=== Topology κ Bake Grid (2×2) braid_gains={gains} ===")
        summary = run_topology_grid(bake_steps=args.bake_steps, braid_gains=gains)
        json_path = save_topology_json(summary)
        print("\n=== Comparison ===")
        for row in summary["comparison_table"]:
            print(
                f"  {row['label']:<16} toroidal={row['toroidal_modulo9']} "
                f"v369={row['vortex_math_369']}  κ {row['kappa_seed']:.3f}→"
                f"{row['kappa_final']:.3f}  drift={row['kappa_drift']:+.4f}  "
                f"nearest={row['nearest_kappa']}"
            )
        print(f"\nBest κ drift: {summary['best_kappa_drift']}")
        print(f"Best Hopf lock: {summary['best_hopf_lock']}")
        print(f"JSON: {json_path}")
        raise SystemExit(0)

    mode_str = "DENSE sweet-spot grid" if args.dense else "Ultra-focused grid"
    print(
        f"   Launching {args.trials} REAL trials | Mode: {mode_str} | "
        f"{'Ray (parallel)' if args.use_ray else 'Single-node (sequential)'}"
    )

    if args.dense:
        gs_values = [0.875, 0.8775, 0.880, 0.8825, 0.885]
        omega_values = [0.02200, 0.02225, 0.02250, 0.02275, 0.02300]
    else:
        gs_values = [0.84, 0.86, 0.88, 0.90]
        omega_values = [0.0215, 0.0220, 0.0225, 0.0230, 0.0235]

    base_grid = []
    for nl in [2, 3, 4]:
        for np_val in [12, 18, 24]:
            for mf in [24, 30, 36, 42, 48]:
                for gs in gs_values:
                    for omega_r in omega_values:
                        base_grid.append(
                            {
                                "num_layers": nl,
                                "num_polarities": np_val,
                                "max_facts": mf,
                                "gauge_strength": gs,
                                "omega_R": omega_r,
                                "toroidal_modulo9": True,
                                "vortex_math_369": True,
                                "wg_base": 350.0,
                                "kappa_seed": KAPPA_DOC,
                                "bake_steps": 120,
                            }
                        )

    if args.trials <= len(base_grid):
        param_grid = base_grid[: args.trials]
    else:
        print(
            f"   Base grid has only {len(base_grid)} unique combos → repeating for {args.trials} trials"
        )
        repeats = (args.trials // len(base_grid)) + 1
        param_grid = base_grid * repeats
        np.random.shuffle(param_grid)
        param_grid = param_grid[: args.trials]

    if args.use_ray:
        try:
            import ray

            ray.init(ignore_reinit_error=True, address="auto")
            print("   Ray initialized successfully - running in parallel")

            @ray.remote
            def remote_trial(trial_id, params):
                return run_epoch_trial(trial_id, params)

            futures = [remote_trial.remote(i, p) for i, p in enumerate(param_grid)]
            results = ray.get(futures)
        except Exception as e:
            print(f"   Ray failed ({e}), falling back to single-node")
            results = [run_epoch_trial(i, p) for i, p in enumerate(param_grid)]
    else:
        print("   Running sequentially (single-node mode)")
        results = [run_epoch_trial(i, p) for i, p in enumerate(param_grid)]

    import pandas as pd

    flat_rows = []
    for row in results:
        flat = {k: v for k, v in row.items() if k not in ("params", "stats_before", "stats_after", "topology", "kappa_trace_last")}
        flat.update(row.get("topology", {}))
        params = row.get("params", {})
        for pk, pv in params.items():
            flat[f"param_{pk}"] = pv
        flat_rows.append(flat)

    df = pd.DataFrame(flat_rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT_DIR / f"epoch_sweep_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    print(f"Sweep complete! Results saved to {csv_path}")
    if "w_g_measured" in df.columns:
        print(f"   W_g mean: {df['w_g_measured'].mean():.3f} (target {W_G_TARGET:.3f})")
    if "kappa_final" in df.columns:
        print(f"   κ final mean: {df['kappa_final'].mean():.4f} (seed {KAPPA_DOC})")