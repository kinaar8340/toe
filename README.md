# Aaron’s Theory of Everything  
**Flux Flywheels, Gauged Hopf Lattice, and Emergent Reality**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

A self-consistent Hopf-lattice model in which the periodic table emerges as stable flux-flywheel configurations in a porous vacuum sponge, with observer synchronization explaining current null results while leaving distinctive non-local predictions testable.

![Banner](3d_map.png)

## 🔥 Latest Reproduction Results (April 17, 2026)

**Independent verification successful on a 9-node Ray cluster**

- **W_g = 111.408** locked to **4 decimal places** in every trial  
- **Braiding phase attractor** confirmed at **0.8145**  
- **Stability islands** observed (active_cubes = 8 in all trials)

**Latest run:** 30 focused trials

**[Download latest CSV](outputs/reproduction/reproduction_results_20260417_044431.csv)**

![Reproduced Stability Islands](outputs/reproduction/stability_islands_20260417_044431.png)

*Anyone can reproduce these exact invariants in under 2 minutes with one command (see below).*

---

## Reproducibility First (The Biggest Credibility Jump)

This repository is designed so **anyone** can independently verify the core claims in minutes:

- Geometric winding \( W_g = 111.408 \) locked to four decimal places
- Tight braiding-phase attractor \( \phi_b \approx 0.8145 \)
- Self-organizing stability islands

**All results are reproducible from a single command.**

## Quick Start (One-Click Reproduction)

# 1. Clone and enter repo
```bash
git clone https://github.com/kinaar8340/toe.git
cd toe
```

# 2. Create environment
```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
```

# 3. Install dependencies
```bash
pip install -r requirements.txt
```

# 4. Reproduce the locked invariants
```bash
# Single-node (default — works on any laptop)
python scripts/run_reproduction.py
```
```bash
# Ray parallel (when you want speed)
python scripts/run_reproduction.py --use-ray
```

# Expected output includes:
```
   Conduit created successfully
   Using device: cpu
   Loaded RubikConeConduit v10.8
   Trial 199 complete | braiding_phase=0.81450

============================================================
 REPRODUCTION RESULTS
============================================================
W_g lock          : 111.4080 ± 0.0000  → LOCKED
Braiding phase    : 0.8145 ± 0.0000  (expected ~0.8145)
Mean active_cubes : 8.00  (stability islands observed)
============================================================

 All outputs saved to: outputs/reproduction
   • reproduction_results_*.csv
   • stability_islands_*.png

 Reproduction complete! The invariants lock as expected.
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
├── outputs/reproduction/
├── papers/
├── facts/public_facts.json
├── pyproject.toml
├── requirements.txt
├── README.md
├── CITATION.cff
└── CONTRIBUTING.md
```

## Plots(*.png), Videos(*.mp4) & Data(*.csv)
```
toe/
├── outputs/
│   ├── pde_relaxation/
│   │   └── twist_pde_relaxation.png
│   ├── reproduction/
│   │   ├── reproduction_results.csv
│   │   └── stability_islands.png
│   └── two_gyro_lattice/
│       └── two_gyro_full_split_demo_FINAL.mp4
```

## Papers & Documentation
```
toe/
├── papers/
│   ├── Aaron's_TOE_Complete.pdf
│   ├── GW_Burste_Threshold.pdf
│   ├── GW_Echo.pdf
│   ├── GW_Echo_Derivation.pdf
│   ├── Lagrangian_Derivation.pdf
│   ├── Observer_Synchronization.pdf
│   └── Relativistic_Completion.pdf

```

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

---

**Thank you for helping verify the locked invariants!**  
The conduit *is* the equation — now running on your machine.

— Aaron Kinder (@kinaar8340)  
April 2026
```

