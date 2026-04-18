## Aaron’s Theory of Everything

**Flux Flywheels, Gauged Hopf Lattice, and Emergent Reality**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)


---

![Banner](3d_map.png)

A self-consistent Hopf-lattice model in which the periodic table emerges as stable flux-flywheel configurations in a porous vacuum sponge, with observer synchronization explaining current null results while leaving distinctive non-local predictions testable.

---

## Latest Reproduction Results (April 17, 2026)
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

*Anyone can reproduce these exact invariants in under 2 minutes with one command (see below).*

---

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
    pip install -r requirements.txt
    ```
4. Run the Simulation:
    ```bash
    # Single-Node (default)
    python scripts/run_reproduction.py
    ```
    ```bash
    # Multi-Node
    python scripts/run_reproduction.py --use-ray
    ```
   
---

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
│   ├── pde_relaxation/
│   │   └── twist_pde_relaxation.png
│   ├── reproduction/
│   │   ├── reproduction_results.csv
│   │   └── stability_islands.png
│   └── two_gyro_lattice/
│       └── two_gyro_full_split_demo_FINAL.mp4
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

---

## Citation

---

```bibtex
@misc{kinder2026aarontoe,
  author       = {Kinder, Aaron},
  title        = {Aaron’s Theory of Everything: Flux Flywheels, Gauged Hopf Lattice, and Emergent Reality},
  year         = {2026},
  howpublished = {\url{https://github.com/kinaar8340/toe}},
  note         = {arXiv preprint (pending)}
}
```

---

## License
MIT License — see [LICENSE](LICENSE).

---

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

