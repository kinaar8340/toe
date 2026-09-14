# Fleet path for the insertion-word catalog

```
MODEL, not theorem.
Geometry and arithmetic stay on QGA / flux_hopf_lib.
The alkane / CH2 / carbene language is an analogy for a discrete insertion step.
n=1 is one flywheel at quaternion identity (methane slot).
n=2 is one extra published step: one extra flywheel XOR one extra rotor insertion.
Do not emit “proves”, “element”, “periodic table identity”, or “carbene is a flywheel”.
group ids are insertion words; molecular names are alias families.
```

The cluster is optional until each word is more than a few quaternion products. Worker Grok auth does not change where this experiment should run.

## Keep the two buses apart

`grok -p` on eight hosts is a full agent each. That is for map-reduce of text, not for quaternion products. The homolog catalog is CPU arithmetic. Use the SSH fleet, not eight copies of Grok thinking about alkanes.

| Job | Where | How |
|---|---|---|
| Catalog walk, closure, commutator, axis probes | `bud2`–`bud9` | `bin/fleet run` → same venv + `python -m homolog_flywheel.run` |
| Plot, merge CSV, git, banner checks | `bud` | local Python |
| Conduit / lattice bake if you add `--bake-flywheel` | `bud` (4090) or one worker shard | Python, not `grok -p` |
| “Did the aliases stay aliases?” review | `bud` Grok session | read `results/` |

Shard the catalog, not the narrative. One process per host is enough for five words. Pin `flux-hopf-lib==0.2.2` and a 3.13 venv on the workers; do not use `bud`’s 3.14 interpreter there. Do not copy `auth.json`.

## Laptop dry run (required before fan-out)

```bash
PYTHONPATH=src:experiments python -m homolog_flywheel.run \
  --catalog experiments/homolog_flywheel/groups.yaml \
  --shard-index 0 --shard-count 8 \
  --out experiments/outputs/homolog_shard0.json
```

Or:

```bash
PYTHONPATH=src:experiments python -m homolog_flywheel.cluster_sweep --dry-run
```

Empty shards (index 5–7 with five groups) write JSON with `n_rows: 0` and exit 0.

## Fan-out (Python walks, not grok)

`--emit-fleet` only prints. Run the printed `bin/fleet` lines from `~/Playground`.

```bash
cd ~/Playground
bin/fleet copy ~/Projects/toe/experiments/homolog_flywheel/groups.yaml \
  /home/kinaar/Playground/data/groups.yaml

bin/fleet run -- '
  mkdir -p $HOME/Playground/data $HOME/Playground/results
  host=$(hostname)
  i=${host#bud}
  PYTHONPATH=$HOME/Projects/toe/src:$HOME/Projects/toe/experiments \
  $HOME/Projects/toe/venv/bin/python -m homolog_flywheel.run \
    --catalog $HOME/Playground/data/groups.yaml \
    --shard-index $((10#$i - 2)) --shard-count 8 \
    --n-max 8 \
    --out $HOME/Playground/results/homolog_$host.json
'
```

`--n-max 8` applies to groups that do not set their own `n_max`. `ring4_rotor` stays `n_max: 4`.

## Merge on the laptop

```bash
PYTHONPATH=src:experiments python -m homolog_flywheel.merge_shards \
  --in experiments/outputs/fleet \
  --out experiments/outputs/homolog_catalog_merged.json
```

Merges the five non-empty shards. bud7–bud9 stay empty-shard witnesses. Do not retune \(\theta\) on `ring4_rotor`.

## What Grok-on-workers is for, after JSON exists

```bash
bin/fleet grok -p 'Read ~/Playground/results/homolog_$(hostname).json.
Return JSON only: {host, n_rows, modes, max_closure_rad, aliases_identical, z_frozen}.
Do not say proves, element, periodic table, or flywheel identity.'
```

That is a structured reduce of the `reduce` object already in the JSON, not the walk. Parse the last JSON object. Merge on `bud`.

## Do not do this

Do not `bin/fleet grok -p 'expand this TOE to alcohols and aromatics'`. Eight agents will invent chemistry and collapse the analog. Do not mix local subagents with this fleet sweep.
