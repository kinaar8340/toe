# TOE — Theory of Everything (RubikConeConduit v10.8)

**Flux Flywheels, Gauged Hopf Lattice, and Emergent Reality**

![Tests](https://github.com/kinaar8340/toe/actions/workflows/ci.yml/badge.svg)
![Coverage](https://codecov.io/gh/kinaar8340/toe/branch/main/graph/badge.svg)
![mypy](https://img.shields.io/badge/mypy-passing-brightgreen)
![Python 3.13](https://img.shields.io/badge/python-3.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)

![Banner](3d_map.png)

A self-consistent Hopf-lattice model in which the periodic table emerges as stable flux-flywheel configurations in a porous vacuum sponge, with observer synchronization explaining current null results while leaving distinctive non-local predictions testable.

## Quick Start (One-Click Reproduction)
1. Clone Repository:
```bash
git clone https://github.com/kinaar8340/toe.git
cd toe
```
2. Create Environment:
```bash
python -m venv venv
source venv/bin/activate
```
3. Install Dependencies:
```bash
# Shared core (editable if developing flux_hopf_lib alongside):
#   pip install -e ../flux_hopf_lib
pip install -r requirements.txt
```
4. Run the Simulation:
```bash
# Single-Node (default 30 trials)
python scripts/epoch_bake_sweep.py
python scripts/run_reproduction.py
```
```bash
# Or Run the Module as a Script
python -m scripts.epoch_bake_sweep
python -m scripts.run_reproduction
```
```bash
# Single-Node (custom)
python scripts/epoch_bake_sweep.py --trials 1000 --dense
python scripts/run_reproduction.py --trials 1000 --dense
```
```bash
# Multi-Node (custom)
python scripts/epoch_bake_sweep.py --use-ray --trials 1000 --dense
python scripts/run_reproduction.py --use-ray --trials 1000 --dense
```
5. Generate Plots:
```bash
python scripts/plot_sweep_results.py
```
6. Tests:
```bash
python -m pytest tests/ -q --cov=conduit
```
7. Python API Example:
```bash
from config import load_config
from conduit import RubikConeConduit

cfg = load_config("configs/default.yaml")
model = RubikConeConduit(
    embed_dim=cfg.model.embed_dim,
    twist_rate=cfg.model.twist_rate,
       # ... other params
    )
loss = model.training_step(inputs, optimizer)
recall = model.recall()
print(f"Final recall: {recall:.4f}")
```

## Full File Structure
```
toe/
├── src/
│   └── conduit.py
├── scripts/
│   ├── run_reproduction.py
│   ├── epoch_bake_sweep.py
│   ├── pde_relaxation.py
│   ├── z_flywheel_map.py
│   └── two_gyro_lattice_demo.py
├── facts/
│   └── public_facts.json
├── outputs/
│   ├── epoch_bake/
│   │   └── epoch_sweep.csv
│   ├── pde_relaxation/
│   │   └── twist_pde_relaxation.png
│   ├── reproduction/
│   │   ├── reproduction_results.csv
│   │   └── stability_islands.png
│   ├── two_gyro_lattice/
│   │   └── two_gyro_full_split_demo_FINAL.mp4
│   └── plots/
│       ├── top10_stability_table.png
│       ├── braiding_phase_histogram.png
│       ├── param_vs_stability_scatter.png
│       └── stability_islands_heatmap.png
├── papers/
│   ├── Aaron's_TOE_Complete.pdf
│   ├── GW_Burste_Threshold.pdf
│   ├── GW_Echo.pdf
│   ├── GW_Echo_Derivation.pdf
│   ├── Lagrangian_Derivation.pdf
│   ├── Observer_Synchronization.pdf
│   └── Relativistic_Completion.pdf
├── pyproject.toml
├── requirements.txt
├── README.md
├── CITATION.cff
└── CONTRIBUTING.md
```

## Latest Reproduction Results (April 17, 2026)
```
   Conduit created successfully
   Using device: cpu
   Loaded RubikConeConduit v10.8
   Trial 10000 complete | braiding_phase=0.81404

============================================================
 REPRODUCTION RESULTS
============================================================
W_g lock          : 111.4080 ± 0.0000  → LOCKED
Braiding phase    : 0.8140 ± 0.0008  (expected ~0.8141)
Mean active_cubes : 12.01  (stability islands observed)
============================================================

 All outputs saved to: outputs/reproduction/
   • reproduction_results_*.csv
   • stability_islands_*.png

 Reproduction complete! The invariants lock as expected.
```

![Banner](outputs/plots/param_vs_stability_scatter.png)
![Banner](outputs/plots/braiding_phase_histogram.png)
![Banner](outputs/plots/stability_islands_heatmap.png)
![Banner](outputs/reproduction/stability_islands.png)
![Banner](outputs/pde_relaxation/twist_pde_relaxation.png)
![Banner](outputs/plots/top10_stability_table.png)

## Related: Mystery φ-e-π Residual Probes

The companion [mystery](https://github.com/kinaar8340/mystery) repo quantifies the near-Pythagorean residual R = φ² + e² − π² ≈ +0.137486 and its dynamical analogs (golden-angle packing ≈ 0.1375; exponential survival e⁻² ≈ 0.1353 at λt = 2).

**Dual-role κ** (see [mystery RESULTS.md](https://github.com/kinaar8340/mystery/blob/main/docs/RESULTS.md)):

| Symbol | Value | Role |
|--------|-------|------|
| **κ_doc** | 0.85 | Documentation / theory (θ_crit, B(κ), Hopf framing) |
| **κ_sim** | ≈ 0.89 | Simulation / production (meta-opt, survival @ λt=2) |
| **κ\*** | ≈ 0.8513 | Exact null e/π − R/π² |

Production meta-opt uses κ_sim; documented formulas retain κ_doc.

**TOE integration:**

| File | Purpose |
|------|---------|
| **[flux_hopf_lib](https://github.com/kinaar8340/flux_hopf_lib)** | Canonical λt survival, κ, Hopf/quaternion primitives |
| `src/relaxation_survival.py` | Thin re-export shim of `flux_hopf_lib.simulation` (backward-compatible) |
| `src/conduit.py` → `RubikConeConduit.run_survival_probe()` | Conduit invariants at normalized λt = 2 |
| `scripts/pde_relaxation.py --normalize-to-lambda-t 2` | PDE survival benchmark |
| `scripts/epoch_bake_sweep.py --topology-grid` | Topology κ bake grid; braid_feedback_gain default 0.002 |
| `scripts/magic_island_sweep.py --topology-grid` | Magic-island braid-gain × topology grid |

Prefer new code::

```python
from flux_hopf_lib.simulation import simulate_twist_pde_survival, compare_to_analogs
# or legacy:
from relaxation_survival import simulate_twist_pde_survival
```

Mystery probes use `flux_hopf_lib` directly; conduit-only scripts still load `toe/src/conduit.py`.

---

## Citation
```bibtex
@misc{kinder2026aarontoe,
  author       = {Kinder, Aaron},
  title        = {Aaron’s Theory of Everything: Flux Flywheels, Gauged Hopf Lattice, and Emergent Reality},
  year         = {2026},
  howpublished = {\url{https://github.com/kinaar8340/toe}},
  note         = {arXiv preprint (pending)}
}
```

## License
MIT License — see [LICENSE](LICENSE).

## Contacts
```
Name:   Aaron Kinder
X:      @kinaar8340
Emails: kinaar0@protonmail.com
Date:   April 2026
```

---

Thank you for helping verify the locked invariants!

---
