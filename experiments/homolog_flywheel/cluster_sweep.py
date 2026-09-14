"""Shard the homolog catalog over SSH hosts. CPU arithmetic, not grok -p.

MODEL, not theorem.
Geometry and arithmetic stay on QGA / flux_hopf_lib.
The alkane / CH2 / carbene language is an analogy for a discrete insertion step.
n=1 is one flywheel at quaternion identity (methane slot).
n=2 is one extra published step: one extra flywheel XOR one extra rotor insertion.
Do not emit “proves”, “element”, “periodic table identity”, or “carbene is a flywheel”.
group ids are insertion words; molecular names are alias families.

The walk itself is `python -m homolog_flywheel.run --catalog --shard-index`.
Do not send the walk through eight Grok agents.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from homolog_flywheel.analog import DISCLAIMER, NOTES_PREFIX
from homolog_flywheel.catalog import CATALOG_PATH
from homolog_flywheel.run import main as run_main

DEFAULT_HOSTS = ["bud2", "bud3", "bud4", "bud5", "bud6", "bud7", "bud8", "bud9"]
REPO_ROOT = Path(__file__).resolve().parents[2]


def fleet_remote_command(
    *,
    catalog: str,
    n_max: int,
    shard_count: int,
    out_template: str,
) -> str:
    """Remote body for bin/fleet run. Shard index is hostname budN → N-2."""
    return (
        "mkdir -p $HOME/Playground/data $HOME/Playground/results; "
        "host=$(hostname); i=${host#bud}; "
        "PYTHONPATH=$HOME/Projects/toe/src:$HOME/Projects/toe/experiments "
        "$HOME/Projects/toe/venv/bin/python -m homolog_flywheel.run "
        f"--catalog {catalog} "
        "--shard-index $((10#$i - 2)) "
        f"--shard-count {shard_count} "
        f"--n-max {n_max} "
        f"--out {out_template}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m homolog_flywheel.cluster_sweep",
        description="Shard catalog walks over bud2-bud9. Not a Grok map-reduce.",
        epilog=DISCLAIMER,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--catalog", default=str(CATALOG_PATH))
    parser.add_argument("--hosts", default=",".join(DEFAULT_HOSTS))
    parser.add_argument("--ssh-user", default="kinaar", help="SSH user for emit-fleet notes")
    parser.add_argument("--workdir", default=str(REPO_ROOT), help="repo path for emit-fleet notes")
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "experiments" / "outputs" / "homolog_shard0.json"),
        help="JSON path for --dry-run; fleet uses per-host names",
    )
    parser.add_argument("--n-max", type=int, default=4)
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run shard 0 locally and write one JSON (required before fan-out)",
    )
    parser.add_argument(
        "--emit-fleet",
        action="store_true",
        help="print bin/fleet run command; do not grok -p the walk",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(DISCLAIMER)
    print()
    if args.emit_fleet:
        remote = fleet_remote_command(
            catalog="$HOME/Playground/data/groups.yaml",
            n_max=int(args.n_max),
            shard_count=int(args.shard_count),
            out_template="$HOME/Playground/results/homolog_$host.json",
        )
        print("# analog: SSH fleet runs Python walks, not grok -p")
        print(f"# ssh-user={args.ssh_user} workdir={args.workdir}")
        print("# This flag only PRINTS. It does not SSH and does not walk.")
        print("# From ~/Playground, run the two commands below.")
        print("# Shard 0 is linear_rotor only; other groups are other hosts.")
        print("cd ~/Playground")
        print("bin/fleet copy ~/Projects/toe/experiments/homolog_flywheel/groups.yaml \\")
        print("  /home/kinaar/Playground/data/groups.yaml")
        print(f"bin/fleet run --hosts {args.hosts} -- '{remote}'")
        print()
        print(f"{NOTES_PREFIX} --emit-fleet did not contact bud2-bud9.")
        print(f"{NOTES_PREFIX} cd ~/Playground and run the bin/fleet lines.")
        return 0
    if args.dry_run:
        out = Path(args.out)
        catalog = args.catalog
        print(f"{NOTES_PREFIX} dry-run shard 0/{args.shard_count} locally; not grok -p")
        return run_main(
            [
                "--catalog",
                catalog,
                "--shard-index",
                "0",
                "--shard-count",
                str(args.shard_count),
                "--n-max",
                str(args.n_max),
                "--out",
                str(out),
            ]
        )
    print(
        f"{NOTES_PREFIX} pass --dry-run (local shard 0 JSON) or --emit-fleet. "
        "Do not grok -p the catalog walk.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
