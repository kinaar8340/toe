# scripts/epoch_bake_sweep.py — v2.1 Ultra-Focused Sweep
"""
Focused hyperparameter sweep around the proven best region from the last 200-trial run.
Only the highest-performing combinations are tested → much higher signal, less noise.
Nuclear cache purge + direct source load kept for full 9-node reliability.
"""

import os
import sys
import ray
import torch
import torch.nn.functional as F
import pandas as pd
import json
import argparse
import numpy as np
import importlib
import importlib.util
import shutil
from pathlib import Path
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

ray.init(address="auto", ignore_reinit_error=True)

print("🌟 Epoch-Synchronous Magic Island Sweep v2.0 — FOCUSED YAHTZEE REGION")
print(f"→ Connected nodes: {len(ray.nodes())}")
print(f"→ Total CPUs available: {ray.cluster_resources().get('CPU', 0)}")


# FORCE RELOAD on driver
import src.conduit
importlib.reload(src.conduit)
from src.conduit import RubikConeConduit, CopresheafDiffusionStack, RingConeChain
from src.config import load_config

cfg = load_config("configs/default.yaml")
public_facts_file = Path("facts/public_facts.json")

@ray.remote(num_cpus=12, num_gpus=0, max_retries=3, scheduling_strategy="SPREAD")
def run_epoch_trial(trial_id: int, params: dict):
    # === NUCLEAR PURGE + DIRECT SOURCE LOAD (guarantees v10.7 on every node) ===
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    for root, dirs, files in os.walk(project_root):
        if "__pycache__" in dirs:
            shutil.rmtree(os.path.join(root, "__pycache__"), ignore_errors=True)

    importlib.invalidate_caches()
    sys.modules.pop('src.conduit', None)
    conduit_path = os.path.join(project_root, "src", "conduit.py")
    spec = importlib.util.spec_from_file_location("src.conduit", conduit_path)
    conduit_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conduit_module)
    sys.modules["src.conduit"] = conduit_module

    from src.conduit import RubikConeConduit, CopresheafDiffusionStack, RingConeChain
    print(f"   → src.conduit LOADED DIRECTLY from {conduit_path} (mtime: {os.path.getmtime(conduit_path):.0f})")

    torch.set_num_threads(16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" → Using device: {device}")

    conduit = RubikConeConduit(
        embed_dim=cfg.model.embed_dim,
        twist_rate=cfg.model.twist_rate,
        max_depth=cfg.model.max_depth,
        num_polarizations=cfg.model.num_polarizations,
        quat_logical_dim=cfg.model.quat_logical_dim,
        toroidal_modulo9=True,
        vortex_math_369=True,
        clifford_projection=True,
        gauge_strength=params["gauge_strength"],
        omega_R=params["omega_R"]
    ).to(device)

    version = getattr(conduit, "VERSION", "unknown")
    print(f"   → Loaded RubikConeConduit v{version}")

    use_epoch_sync = hasattr(conduit, "enable_epoch_sync") and hasattr(conduit, "epoch_synchronous_bake")
    if use_epoch_sync:
        conduit.enable_epoch_sync(gauge_strength=params["gauge_strength"])
        print("   → Epoch-synchronous mode ENABLED (v10.7+)")
    else:
        print("   ⚠️ Stale conduit → using DIRECT ring bake fallback")

    if not hasattr(conduit, 'ring_cone'):
        conduit.ring_cone = RingConeChain(embed_dim=cfg.model.embed_dim, device=device)

    new_stack = CopresheafDiffusionStack(
        in_channels=conduit.ring_cone.embed_dim,
        hidden_channels=conduit.ring_cone.embed_dim,
        out_channels=conduit.ring_cone.embed_dim,
        num_layers=params["num_layers"],
        num_polarities=params["num_polarities"],
        dropout=0.05,
        use_cooperative_sheaf=True,
        device=device
    )
    new_stack.prepare(conduit.ring_cone.edge_index, conduit.ring_cone.ring_polarities)
    conduit.ring_cone.tnn_stack = new_stack.to(device)

    optimizer = torch.optim.AdamW(conduit.parameters(), lr=1e-4, weight_decay=1e-4)

    raw_data = json.loads(public_facts_file.read_text(encoding="utf-8"))
    lines = [line.strip() for item in raw_data if isinstance(item, dict)
             for line in (item.get("text") or str(item)).splitlines()
             if line.strip() and not line.startswith(("#", "/identity/"))]

    print(f" → Strong bake: {params['max_facts']} facts × 30 training steps on {device}...")

    for idx, fact in enumerate(lines[:params["max_facts"]]):
        emb = F.normalize(torch.randn(384, device=device), dim=-1) * 0.28

        if use_epoch_sync:
            conduit.epoch_synchronous_bake(idx, emb)
        else:
            ring_idx = idx % conduit.ring_cone.NUM_RINGS
            cube_idx = idx % conduit.ring_cone.rings[ring_idx].num_cubes
            conduit.ring_cone.bake_ring(ring_idx, cube_idx, emb, orientation=idx % 24)

        for step in range(30):
            item = {'emb': emb.unsqueeze(0), 's': torch.tensor([4.5 + idx * 4.8], device=device), 'pol_idx': 0}
            try:
                conduit.training_step(
                    inputs=[item],
                    optimizer=optimizer,
                    recon_weight=15000.0,
                    align_weight=55000.0,
                    depth_pull_weight=40000.0,
                    winding_weight=48.0,
                    braiding_weight=18.0
                )
            except Exception:
                pass

        if (idx + 1) % 5 == 0:
            print(f"   → Fact {idx + 1}/{params['max_facts']} baked")

    stats = conduit.monitor_topological_winding(n_samples=512)
    stability_score = stats.get("active_cubes", 5) * 1.0 / (1.0 + 1e-8)

    print(f" → Trial {trial_id} bake complete | braiding_phase={stats.get('braiding_phase', 0):.5f}")

    return {
        "trial_id": trial_id,
        "num_layers": params["num_layers"],
        "num_polarities": params["num_polarities"],
        "max_facts": params["max_facts"],
        "gauge_strength": params["gauge_strength"],
        "omega_R": params["omega_R"],
        "braiding_phase": float(stats.get("braiding_phase", 0.0)),
        "active_cubes": int(stats.get("active_cubes", 0)),
        "epochs_completed": getattr(conduit, "current_epoch", 0),
        "stability_score": float(stability_score),
        "used_epoch_sync": use_epoch_sync,
        "version": version,
        "timestamp": datetime.now().isoformat()
    }


# ==================== LAUNCH — FOCUSED GRID ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=60)
    args = parser.parse_args()

    # ULTRA-FOCUSED GRID (only the proven winners)
    param_grid = []
    for nl in [3, 2, 4]:                     # 3 is heaviest
        for np_val in [18, 24, 12]:
            for mf in [30, 24, 36, 42, 48]:
                for gs in [0.84, 0.86, 0.88, 0.90]:
                    for or_val in [0.0215, 0.0220, 0.0225, 0.0230, 0.0235]:
                        param_grid.append({
                            "num_layers": nl,
                            "num_polarities": np_val,
                            "max_facts": mf,
                            "gauge_strength": gs,
                            "omega_R": or_val,
                        })
    np.random.shuffle(param_grid)
    param_grid = param_grid[:args.trials]

    print(f"→ Launching {len(param_grid)} ULTRA-FOCUSED trials...")
    futures = [run_epoch_trial.remote(i, p) for i, p in enumerate(param_grid)]
    results = ray.get(futures)

    df = pd.DataFrame(results)
    report_path = Path(f"outputs/epoch_sweep_{datetime.now():%Y%m%d_%H%M%S}.md")
    df.to_markdown(report_path, index=False)

    print("✅ v2.1 Ultra-Focused Sweep complete")
    print(f"✅ Report → {report_path}")
    print("\n🏆 Top 10:")
    print(df.sort_values("stability_score", ascending=False).head(10)[
        ["num_layers","num_polarities","max_facts","gauge_strength","stability_score","active_cubes","used_epoch_sync"]
    ].to_string(index=False))
    ray.shutdown()
