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

```bash
# 1. Clone and enter repo
git clone https://github.com/kinaar8340/toe.git
cd toe

# 2. Create environment
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Reproduce the locked invariants
python scripts/run_reproduction.py
```

Expected output includes:
- W_g lock confirmation
- Braiding phase statistics
- Stability islands plot (saved to `outputs/reproduction/`)

---

## Full File Structure

```
toe/
├── src/
│   └── conduit.py
├── scripts/
│   ├── run_reproduction.py          # ← one-command verification
│   ├── epoch_bake_sweep.py
│   ├── pde_relaxation.py
│   ├── z_flywheel_map.py
│   └── two_gyro_lattice_demo.py
├── outputs/reproduction/            # ← latest results appear here
├── papers/                          # All LaTeX sources
├── facts/public_facts.json
├── pyproject.toml
├── requirements.txt
├── README.md
├── CITATION.cff
└── CONTRIBUTING.md
```

## Papers & Documentation

All LaTeX sources are in the `papers/` folder.

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

