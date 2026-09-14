"""Merge fleet shard JSON on the laptop. analog only.

MODEL, not theorem.
Geometry and arithmetic stay on QGA / flux_hopf_lib.
The alkane / CH2 / carbene language is an analogy for a discrete insertion step.
n=1 is one flywheel at quaternion identity (methane slot).
n=2 is one extra published step: one extra flywheel XOR one extra rotor insertion.
Do not emit “proves”, “element”, “periodic table identity”, or “carbene is a flywheel”.
group ids are insertion words; molecular names are alias families.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from homolog_flywheel.analog import DISCLAIMER, NOTES_PREFIX
from homolog_flywheel.catalog import CATALOG_COLUMNS, discover_shard_json, merge_shard_files
from homolog_flywheel.run import _print_table, _write_csv, _write_json

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = REPO_ROOT / "experiments" / "outputs" / "fleet"
DEFAULT_OUT = REPO_ROOT / "experiments" / "outputs" / "homolog_catalog_merged.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m homolog_flywheel.merge_shards",
        description="Merge catalog shard JSON. Empty shards stay witnesses. Do not retune θ.",
        epilog=DISCLAIMER,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--in", dest="in_dir", default=str(DEFAULT_IN), help="shard JSON directory")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="merged JSON path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.in_dir)
    if not root.is_absolute():
        root = REPO_ROOT / root
    paths = discover_shard_json(root)
    if not paths:
        print(f"{NOTES_PREFIX} no homolog_*.json under {root}", file=sys.stderr)
        return 2
    merged = merge_shard_files(paths)
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    _write_json(out, merged)
    _write_csv(out.with_suffix(".csv"), merged["rows"], CATALOG_COLUMNS)
    print(DISCLAIMER)
    print()
    print(
        f"{NOTES_PREFIX} merged {merged['n_nonempty_shards']} nonempty + {merged['n_empty_shards']} empty shards"
    )
    print()
    _print_table(
        merged["rows"],
        cols=[
            "group_id",
            "n",
            "alias",
            "step_mode",
            "identity_overlap",
            "closure_rad",
            "commutator_norm",
        ],
    )
    print()
    print(
        f"{NOTES_PREFIX} empty witnesses: {[e.get('host') or e.get('path') for e in merged['empty_shards']]}"
    )
    print(f"{NOTES_PREFIX} wrote {out}")
    print(f"{NOTES_PREFIX} wrote {out.with_suffix('.csv')}")
    print(merged["hypothesis_verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
