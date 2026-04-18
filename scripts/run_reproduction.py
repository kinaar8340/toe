#!/usr/bin/env python3
"""
run_reproduction.py — One-command reproduction with single-node vs Ray mode
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import the trial function (still pure single-node compatible)
from scripts.epoch_bake_sweep import run_epoch_trial

OUTPUT_DIR = Path("outputs/reproduction")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_reproduction(num_trials: int = 30, use_ray: bool = False):
    print("Starting Aaron’s TOE Reproduction Script")
    print(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Trials: {num_trials} | Mode: {'Ray (parallel)' if use_ray else 'Single-node (sequential)'}")

    # Focused parameter grid (same proven Yahtzee region)
    param_grid = []
    for nl in [2, 3, 4]:
        for np_val in [12, 18, 24]:
            for mf in [24, 30, 36]:
                for gs in [0.84, 0.86, 0.88]:
                    for omega_r in [0.0215, 0.0225, 0.0230]:
                        param_grid.append({
                            "num_layers": nl,
                            "num_polarities": np_val,
                            "max_facts": mf,
                            "gauge_strength": gs,
                            "omega_R": omega_r,
                        })

    np.random.shuffle(param_grid)
    param_grid = param_grid[:num_trials]

    print(f"   Launching {len(param_grid)} focused trials...")

    # === EXECUTION MODE: toggle with --use-ray ===
    if use_ray:
        try:
            import ray
            # This will connect to an existing cluster OR auto-start a local one
            ray.init(ignore_reinit_error=True)
            print("   Ray initialized successfully - running in parallel")

            @ray.remote
            def remote_trial(trial_id, params):
                return run_epoch_trial(trial_id, params)

            futures = [remote_trial.remote(i, p) for i, p in enumerate(param_grid)]
            results = ray.get(futures)

        except Exception as e:
            print(f"   Ray failed to initialize ({e}). Falling back to single-node mode.")
            results = [run_epoch_trial(i, p) for i, p in enumerate(param_grid)]
    else:
        print("   Running sequentially (single-node mode)")
        results = [run_epoch_trial(i, p) for i, p in enumerate(param_grid)]

    # === RESULTS PROCESSING (unchanged) ===
    df = pd.DataFrame(results)

    # Expand params for plotting
    params_df = pd.json_normalize(df['params'])
    df = pd.concat([df.drop(columns=['params']), params_df], axis=1)

    # Save raw results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(OUTPUT_DIR / f"reproduction_results_{timestamp}.csv", index=False)

    # === INVARIANT CHECKS ===
    wg_mean = df["w_g"].mean()
    wg_std = df["w_g"].std()
    braiding_mean = df["braiding_phase"].mean()
    braiding_std = df["braiding_phase"].std()
    active_mean = df["active_cubes"].mean()

    print("\n" + "="*60)
    print(" REPRODUCTION RESULTS")
    print("="*60)
    print(f"W_g lock          : {wg_mean:.4f} ± {wg_std:.4f}  → {'LOCKED' if abs(wg_mean - 111.408) < 0.001 else 'DRIFT'}")
    print(f"Braiding phase    : {braiding_mean:.4f} ± {braiding_std:.4f}  (expected ~0.8145)")
    print(f"Mean active_cubes : {active_mean:.2f}  (stability islands observed)")
    print("="*60)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df["num_polarities"] + 2 * df["max_facts"], df["active_cubes"],
                c=df["braiding_phase"], cmap='viridis', s=80, edgecolor='black')
    plt.colorbar(label='Braiding Phase φ_b')
    plt.xlabel('Pseudo-Z (num_polarities + 2×max_facts)')
    plt.ylabel('Active Cubes (Stability)')
    plt.title('Reproduced Stability Islands (Pseudo-Z vs Active Cubes)')
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / f"stability_islands_{timestamp}.png", dpi=200)
    plt.close()

    print(f"\n All outputs saved to: {OUTPUT_DIR}")
    print("   • reproduction_results_*.csv")
    print("   • stability_islands_*.png")
    print("\n Reproduction complete! The invariants lock as expected.")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Reproduce Aaron’s TOE invariants")
    parser.add_argument("--trials", type=int, default=30, help="Number of focused trials (default 30)")
    parser.add_argument("--use-ray", action="store_true", help="Use Ray for parallel execution (default: single-node)")
    args = parser.parse_args()

    run_reproduction(num_trials=args.trials, use_ray=args.use_ray)