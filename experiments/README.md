# experiments/

Satellite **Model** experiments. Not theorems. Not the QGA spine.

```
MODEL, not theorem.
Geometry and arithmetic stay on QGA / flux_hopf_lib.
The alkane / CH2 / carbene language is an analogy for a discrete insertion step.
n=1 is one flywheel at quaternion identity (methane slot).
n=2 is one extra published step: one extra flywheel XOR one extra rotor insertion.
Do not emit “proves”, “element”, “periodic table identity”, or “carbene is a flywheel”.
group ids are insertion words; molecular names are alias families.
```

## Mapping table (hypothesis only)

Unbranched alkane display series differs by one analog insertion step
\(\mathrm{C}_n\mathrm{H}_{2n+2} \xrightarrow{+\mathrm{CH_2}} \mathrm{C}_{n+1}\mathrm{H}_{2n+4}\).
Methane \(\to\) ethane is the first step. Labels are German aliases from the source figure.

| Analog name | \(n\) | Lattice operator | Implemented |
|---|---|---|---|
| methane slot | 1 | one `FluxFlywheel` at quaternion identity \(q=(1,0,0,0)\) | yes |
| ethane step | 2 | identity flywheel **plus one** extra published step | yes |
| propane+ | \(\ge 3\) | repeat the same published step | optional, \(n\le 10\) |

Exactly one extra step at \(n=2\). CLI `--step {flywheel, rotor, published}` selects which. Default: `rotor`.

## How to run

From the repo root, with the project venv activated:

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[test]"
PYTHONPATH=src:experiments python -m homolog_flywheel.run --n-max 4 --step rotor
```

Other modes:

```bash
PYTHONPATH=src:experiments python -m homolog_flywheel.run --n-max 4 --step flywheel
PYTHONPATH=src:experiments python -m homolog_flywheel.run --n-max 4 --step published
```

Mode comparison at a **fixed** `--n-max` (axis rule, not alkane labels):

```bash
PYTHONPATH=src:experiments python -m homolog_flywheel.run --n-max 4 --compare-modes
```

Writes `homolog_mode_compare.csv` and `homolog_mode_summary.csv`. Aliases match at each \(n\); modes differ by how the insertion axis is chosen. `Z` stays frozen.

`--slow` enables optional PDE / conduit invariants. Tests:

```bash
python -m pytest tests/test_homolog_flywheel_experiment.py -q
```

## What “published step” means in this repo

`--step published` wraps **one** already-public function pair; it does not invent a new lattice operator:

1. `scripts/z_flywheel_map.py` :: `map_z_to_flywheel(z)` with **Z frozen at 2** (that script’s first demo value). A local `step_index` increments with \(n\). **`n` is never passed as `Z`.**
2. Because `map_z_to_flywheel` does not mutate a quaternion, the discrete \(q\) update reuses `flux_hopf_lib.conduit.apply_golden_angle_increment` / `GOLDEN_ANGLE_RAD` (the same golden-angle increment already used by conduit bake / `GoldenAngleMixin`), applied as `small_rotor`.

`--step rotor` is the same `small_rotor` increment without the z-map wrap. `--step flywheel` appends one `FluxFlywheel` instead of rotating the slot quaternion.

Axis rules (the thing that is supposed to differ across modes):

| `--step` | analog axis rule | published source |
|---|---|---|
| `rotor` | CLI `--axis`, fixed (default \(z\)) | CLI |
| `flywheel` | bake \(x=(1,0,0)\) | `RubikConeConduit.epoch_synchronous_bake` rotor axis |
| `published` | golden-angle rotate CLI axis about bake-\(x\) | `apply_golden_angle_increment` + bake \(x\) |

Display aliases (`methan` …) are the same function of \(n\) in every mode. Do not read them as a chemistry check.

Rotor overlap is the single-axis formula \(\lvert\cos((n-1)\theta/2)\rvert\). Published overlap is \(\lvert\mathrm{Re}(q)\rvert\) of composing the same \(\theta\)-rotors whose axes walk by a golden-angle rotation about bake-\(x\). Those numbers match an independent left- or right-multiply (default \(z\)). They are not a butane signature. Default \(z\) rotated about bake-\(x\) stays in the \(yz\)-plane; published grows an \(x\)-component only if `--axis` has one.

Walking-plane probe (published only; still not \(Z\)):

```bash
PYTHONPATH=src:experiments python -m homolog_flywheel.run --n-max 4 --probe-axis 1 0 1
```

Insertion-word catalog (laptop path; group ids are words, names are alias families):

```bash
PYTHONPATH=src:experiments python -m homolog_flywheel.run \
  --catalog experiments/homolog_flywheel/groups.yaml \
  --compare-modes --probe-axis 1 0 1
```

Writes `homolog_catalog.csv` with `closure_rad` and `commutator_norm`. `ring4_rotor` reports closure error at golden \(\theta\); do not tune \(\theta\) to force a cycloalkane.

Local shard 0 only (this is `linear_rotor`, not the full catalog):

```bash
PYTHONPATH=src:experiments python -m homolog_flywheel.run \
  --catalog experiments/homolog_flywheel/groups.yaml \
  --shard-index 0 --shard-count 8 \
  --out experiments/outputs/homolog_shard0.json
```

`--emit-fleet` **prints** SSH commands. It does not fan out. From `~/Playground`:

```bash
cd ~/Playground
bin/fleet copy ~/Projects/toe/experiments/homolog_flywheel/groups.yaml \
  /home/kinaar/Playground/data/groups.yaml
bin/fleet run --hosts bud2,bud3,bud4,bud5,bud6,bud7,bud8,bud9 -- \
  'mkdir -p $HOME/Playground/results; host=$(hostname); i=${host#bud};
   PYTHONPATH=$HOME/Projects/toe/src:$HOME/Projects/toe/experiments \
   $HOME/Projects/toe/venv/bin/python -m homolog_flywheel.run \
     --catalog $HOME/Playground/data/groups.yaml \
     --shard-index $((10#$i - 2)) --shard-count 8 --n-max 4 \
     --out $HOME/Playground/results/homolog_$host.json'
```

See `docs/fleet_catalog.md`. Workers: Python 3.13 venv + `flux-hopf-lib==0.2.2`. Do not `grok -p` the walk.

Overnight analog grid (inverse word, axis hemisphere, optional `--slow`). Golden \(\theta\) stays frozen; `closure_rad(θ)` is a curve, not a fit. `--emit-fleet` only prints:

```bash
PYTHONPATH=src:experiments python -m homolog_flywheel.grid_sweep --dry-run
PYTHONPATH=src:experiments python -m homolog_flywheel.grid_sweep --emit-fleet --n-max 10 --n-axes 64
```

Merge shards on the laptop (empty shards stay witnesses; do not retune \(\theta\)):

```bash
PYTHONPATH=src:experiments python -m homolog_flywheel.merge_shards \
  --in experiments/outputs/fleet \
  --out experiments/outputs/homolog_catalog_merged.json
```

`walk_phase_rad` stays \(\theta\). S² `axis_drift_rad` equals \(\theta\) on default \(z\) (CLI \(\perp\) bake-\(x\)); off-\(yz\) it is the constant chord \(\arccos(v\cdot R_x(\theta)v)\). Identity overlap at \(n=4\) changes. Aliases stay `butan`.

## What this does not do

- No periodic-table proof, no “element” identity, no noble-gas claims.
- No carbene insertion chemistry and no carbene PDE.
- No edits to QGA, `papers/`, or `src/conduit.py`.
- No mapping of chain index \(n\) onto atomic number `Z`.
- No second alkene / cycloalkane lattice.
- Do not search \(\theta\) so that a ring word closes and then call that cyclobutane.

## Relation to `scripts/z_flywheel_map.py`

`Z` remains the existing stability-island index (element analog in that script).
`n` is a new chain-length index for this experiment. They are not the same variable.

## Reproduction

- Seed: `--seed 0`
- Default angle: `GOLDEN_ANGLE_RAD` \(\approx 2\pi / \varphi^2\) (published golden-angle increment). Fallback if that helper cannot be imported: \(2\pi / \varphi^2\), recorded in `notes`.
- Default axis: `0 0 1`
- Writes `experiments/outputs/homolog_table.csv` and `experiments/outputs/homolog_run.json`
- Optional plot `experiments/outputs/homolog_overlap_vs_n.png` if matplotlib imports
- CSV columns: `n, alias, step_mode, axis_rule, q_w, q_x, q_y, q_z, axis_x, axis_y, axis_z, identity_overlap, unit_norm_error, step_geodesic_rad, axis_drift_rad, walk_phase_rad, identity_preservation`
- `identity_preservation` is the cheap overlap analog of the z-map vocabulary, **not** the 300-frame map
- Exit 0 if every state is a unit quaternion and \(n\) ran `1..n-max`. Exit 2 if the identity seed is not \(q=(1,0,0,0)\) within `1e-6`.

Hypothesis recorded in JSON (not confirmed): if the \(\mathrm{CH_2}\) analog is a small stable insertion, `step_geodesic_rad` is nearly constant in \(n\), `unit_norm_error` stays ~0, and `identity_overlap` falls smoothly rather than collapsing at \(n=2\). Contrary runs are marked `hypothesis: not supported by this run`, never as a refutation of QGA.
