# Aaron’s Theory of Everything  
**Flux Flywheels, Gauged Hopf Lattice, and Emergent Reality**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-pending-orange.svg)](https://arxiv.org)

**One-sentence summary:**  
A self-consistent Hopf-lattice model in which the periodic table emerges as stable flux-flywheel configurations in a porous vacuum sponge, with observer synchronization explaining current null results while leaving distinctive non-local predictions testable.

## Reproducibility First (The Biggest Credibility Jump)

This repository is designed so **anyone** can independently verify the core claims in minutes on a standard laptop or GPU cluster:

- Locked geometric winding \(W_g = 111.408\) (to four decimal places)
- Tight braiding-phase attractor \(\phi_b \in [0.765, 0.823]\) (mean \(\approx 0.8145\))
- Self-organizing stability islands that map onto periodic-table-like shell closures via pseudo-Z

**All results are reproducible from a single command.** No proprietary data, no hidden parameters.

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

# 4. Reproduce the locked invariants (1000-trial focused sweep)
python scripts/run_reproduction.py --sweep focused --trials 100

# This outputs:
#   • W_g lock confirmation
#   • Braiding-phase statistics
#   • Stability islands (active_cubes vs pseudo-Z)
#   • Plots saved to outputs/
```

Expected output (excerpt):
```
✅ W_g = 111.408 locked to 4 decimal places in 100/100 trials
✅ Braiding phase attractor: mean = 0.8145 ± 0.008
✅ Stability islands reproduced (noble-gas analogs at active_cubes ≥ 8)
```

## Full File Structure

```
toe/
├── src/
│   └── conduit.py                  # Core RubikConeConduit + Hopf lattice
├── scripts/
│   ├── two_gyro_lattice_demo.py    # Discrete two-gyro demo + plots
│   ├── epoch_bake_sweep.py         # Focused hyperparameter sweep
│   ├── pde_relaxation.py           # 24³ continuum PDE solver
│   └── run_reproduction.py         # One-command reproduction script
├── configs/
│   └── default.yaml                # Sweep & model parameters
├── facts/
│   └── public_facts.json           # Public fact corpus for baking
├── outputs/                        # Generated plots & reports (gitignored)
├── papers/                         # All LaTeX sources
│   ├── Aaron_TOE_Complete.pdf
│   ├── Relativistic_Completion.pdf
│   ├── Lagrangian_Derivation.pdf
│   ├── Observer_Synchronization.pdf
│   ├── GW_Echo.pdf
│   ├── GW_Echo_Derivation.pdf
│   └── GW_Burst_Threshold.pdf
├── requirements.txt
├── LICENSE
├── README.md
└── CITATION.cff                    # Citation file
```

## Installation (Detailed)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  # or +cu118 for CUDA
pip install -r requirements.txt
```

`requirements.txt` contains: `torch`, `numpy`, `pandas`, `matplotlib`, `tqdm`, `ray`, `pyyaml`, etc.

## Reproducing Every Major Result

| Script / Command                            | What it reproduces                          | Time     |
|---------------------------------------------|---------------------------------------------|----------|
| `python scripts/run_reproduction.py`        | Full invariants + stability islands         | ~5–15 min |
| `python scripts/pde_relaxation.py`          | Continuum PDE relaxation to low-twist domain| ~30 sec  |
| `python scripts/two_gyro_lattice_demo.py`   | Stable vs chaotic gauge pointer + bursts    | ~1 min   |
| `python scripts/epoch_bake_sweep.py --trials 60` | Focused Yahtzee sweep (magic islands)   | ~10–20 min |

All outputs are saved to `outputs/` with timestamps.

## Papers & Documentation

- Main TOE overview
- Lagrangian derivation for the conduit PDE
- Relativistic completion + quantization
- Observer synchronization & residual leakage
- GW echo bounds & frequency derivation
- Burst threshold derivation

## Citation

If you use or reproduce this work, please cite:

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

MIT License — see [LICENSE](LICENSE). You are free to use, modify, and build upon this work with attribution.

## Questions & Collaboration

- Open an issue for bugs or reproduction questions.
- Discussions tab for conceptual or theoretical discussion.
- Contact: @kinaar8340 on X or open a discussion.

**Thank you for helping verify the locked invariants!**  
The conduit *is* the equation — now running on your machine.

— Aaron Kinder  
April 2026
