"""CLI entry for the homolog_flywheel Model analog.

MODEL, not theorem.
Geometry and arithmetic stay on QGA / flux_hopf_lib.
The alkane / CH2 / carbene language is an analogy for a discrete insertion step.
n=1 is one flywheel at quaternion identity (methane slot).
n=2 is one extra published step: one extra flywheel XOR one extra rotor insertion.
Do not emit “proves”, “element”, “periodic table identity”, or “carbene is a flywheel”.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from homolog_flywheel.analog import (
    DEFAULT_STEP_MODE,
    DISCLAIMER,
    N_MAX,
    N_MIN,
    NOTES_PREFIX,
    STEP_MODES,
)
from homolog_flywheel.catalog import CATALOG_COLUMNS, CATALOG_PATH, run_catalog
from homolog_flywheel.compare import (
    COMPARE_COLUMNS,
    SUMMARY_COLUMNS,
    comparison_verdict,
    mode_comparison,
    published_axis_probe,
    run_chain,
)
from homolog_flywheel.insert import DEFAULT_ANGLE_RAD, DEFAULT_AXIS
from homolog_flywheel.measure import CSV_COLUMNS, hypothesis_verdict
from homolog_flywheel.plot import plot_mode_compare, plot_overlap_vs_n
from homolog_flywheel.seed import IDENTITY_Q

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "experiments" / "outputs"

HYPOTHESIS = (
    "hypothesis: If the CH2 analog is a small stable insertion, "
    "step_geodesic_rad is nearly constant in n, unit_norm_error stays ~0, "
    "and identity_overlap falls smoothly rather than collapsing at n=2."
)


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _lib_version() -> str | None:
    try:
        import flux_hopf_lib

        return getattr(flux_hopf_lib, "__version__", None)
    except ImportError:
        return None


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(payload), indent=2, sort_keys=True) + "\n")


def _print_table(rows: list[dict[str, Any]], cols: list[str] | None = None) -> None:
    if cols is None:
        cols = [
            "n",
            "alias",
            "step_mode",
            "identity_overlap",
            "axis_drift_rad",
            "walk_phase_rad",
            "step_geodesic_rad",
        ]
    widths = {c: max(len(c), 8) for c in cols}
    for row in rows:
        for c in cols:
            widths[c] = max(widths[c], len(_fmt(row.get(c))))
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("  ".join("-" * widths[c] for c in cols))
    limit = min(len(rows), 30)
    for row in rows[:limit]:
        print("  ".join(_fmt(row.get(c)).ljust(widths[c]) for c in cols))


def _fmt(value: Any) -> str:
    if value is None:
        return "nan"
    if isinstance(value, float):
        if value != value:  # NaN
            return "nan"
        return f"{value:.6f}"
    return str(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m homolog_flywheel.run",
        description="Discrete n-step flux-flywheel insertion analog (MODEL, not theorem).",
        epilog=DISCLAIMER,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n-max", type=int, default=4, help="chain length 1..10 (default 4)")
    parser.add_argument(
        "--step",
        choices=STEP_MODES,
        default=DEFAULT_STEP_MODE,
        help="exactly one insertion mode (default rotor)",
    )
    parser.add_argument(
        "--angle-rad",
        type=float,
        default=DEFAULT_ANGLE_RAD,
        help="insertion angle; default is the published golden-angle increment",
    )
    parser.add_argument(
        "--axis",
        nargs=3,
        type=float,
        default=list(DEFAULT_AXIS),
        metavar=("X", "Y", "Z"),
        help="insertion axis (default 0 0 1)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="output directory (default experiments/outputs)",
    )
    parser.add_argument("--slow", action="store_true", help="enable PDE / conduit invariants")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--compare-modes",
        action="store_true",
        help="at fixed n-max, compare rotor vs flywheel vs published by axis rule",
    )
    parser.add_argument(
        "--probe-axis",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="second CLI axis for a published walking-plane probe (use an x-component)",
    )
    parser.add_argument(
        "--catalog",
        nargs="?",
        const=str(CATALOG_PATH),
        default=None,
        help="run insertion-word catalog (default groups.yaml if flag has no path)",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="catalog shard index (bud2 → 0, bud3 → 1, …)",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="number of catalog shards (8 for bud2-bud9)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not N_MIN <= args.n_max <= N_MAX:
        print(f"{NOTES_PREFIX} --n-max must be in {N_MIN}..{N_MAX}", file=sys.stderr)
        return 2
    np.random.seed(args.seed)
    axis = np.asarray(args.axis, dtype=float)
    out_arg = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out_arg.suffix == ".json":
        out_dir = out_arg.parent
        catalog_json_path = out_arg
    else:
        out_dir = out_arg
        catalog_json_path = out_dir / "homolog_catalog.json"
    json_path = out_dir / "homolog_run.json"

    if args.catalog is not None:
        code = _run_catalog(args, Path(args.catalog), out_dir, catalog_json_path)
        if code != 0:
            return code
        if not args.compare_modes and args.probe_axis is None:
            return 0
    if args.compare_modes:
        return _run_compare(args, axis, out_dir, json_path)
    if args.probe_axis is not None:
        return _run_probe(args, axis, np.asarray(args.probe_axis, dtype=float), out_dir, json_path)

    states = run_chain(
        n_max=args.n_max,
        step_mode=args.step,
        angle_rad=float(args.angle_rad),
        axis=axis,
        slow=bool(args.slow),
        seed=int(args.seed),
    )
    seed_q = np.asarray(states[0].q, dtype=float)
    if not np.allclose(seed_q, IDENTITY_Q, atol=1e-6):
        print(f"{NOTES_PREFIX} identity seed is not q=(1,0,0,0) within 1e-6", file=sys.stderr)
        return 2

    rows = [s.invariants for s in states]
    expected_n = list(range(1, args.n_max + 1))
    got_n = [int(r["n"]) for r in rows]
    unit_ok = all(float(r["unit_norm_error"]) < 1e-6 for r in rows)
    if got_n != expected_n or not unit_ok:
        print(f"{NOTES_PREFIX} chain failed unit-norm or n-range check", file=sys.stderr)
        return 2

    csv_path = out_dir / "homolog_table.csv"
    png_path = out_dir / "homolog_overlap_vs_n.png"
    _write_csv(csv_path, rows, CSV_COLUMNS)
    verdict = hypothesis_verdict(rows)
    payload = {
        "disclaimer": DISCLAIMER,
        "hypothesis": HYPOTHESIS,
        "hypothesis_verdict": verdict,
        "config": {
            "n_max": args.n_max,
            "step": args.step,
            "compare_modes": False,
            "angle_rad": float(args.angle_rad),
            "axis": [float(x) for x in axis],
            "slow": bool(args.slow),
            "seed": int(args.seed),
            "out": str(out_dir),
        },
        "git_sha": _git_sha(),
        "flux_hopf_lib_version": _lib_version(),
        "n_ran": got_n,
        "rows": rows,
        "notes": [s.notes for s in states],
    }
    _write_json(json_path, payload)
    plot_overlap_vs_n(rows, png_path)

    print(DISCLAIMER)
    print()
    _print_table(rows)
    print()
    print(f"{NOTES_PREFIX} wrote {csv_path}")
    print(f"{NOTES_PREFIX} wrote {json_path}")
    print(verdict)
    return 0


def _run_compare(
    args: argparse.Namespace,
    axis: np.ndarray,
    out_dir: Path,
    json_path: Path,
) -> int:
    long_rows, summaries, chains = mode_comparison(
        n_max=args.n_max,
        angle_rad=float(args.angle_rad),
        axis=axis,
        slow=bool(args.slow),
        seed=int(args.seed),
    )
    for mode, states in chains.items():
        seed_q = np.asarray(states[0].q, dtype=float)
        if not np.allclose(seed_q, IDENTITY_Q, atol=1e-6):
            print(
                f"{NOTES_PREFIX} identity seed is not q=(1,0,0,0) within 1e-6 ({mode})",
                file=sys.stderr,
            )
            return 2
        got_n = [int(s.n) for s in states]
        unit_ok = all(float(s.invariants["unit_norm_error"]) < 1e-6 for s in states)
        if got_n != list(range(1, args.n_max + 1)) or not unit_ok:
            print(
                f"{NOTES_PREFIX} chain failed unit-norm or n-range check ({mode})", file=sys.stderr
            )
            return 2

    compare_csv = out_dir / "homolog_mode_compare.csv"
    summary_csv = out_dir / "homolog_mode_summary.csv"
    png_path = out_dir / "homolog_mode_compare.png"
    _write_csv(compare_csv, long_rows, COMPARE_COLUMNS)
    _write_csv(summary_csv, summaries, SUMMARY_COLUMNS)
    verdict = comparison_verdict(long_rows, summaries)
    payload = {
        "disclaimer": DISCLAIMER,
        "hypothesis": (
            "hypothesis: at fixed n-max, rotor / flywheel / published differ by axis rule "
            "(cli_fixed vs bake_x vs golden_rotate_cli), not by alkane aliases. "
            "n is chain index; Z stays frozen."
        ),
        "hypothesis_verdict": verdict,
        "config": {
            "n_max": args.n_max,
            "compare_modes": True,
            "angle_rad": float(args.angle_rad),
            "axis": [float(x) for x in axis],
            "slow": bool(args.slow),
            "seed": int(args.seed),
            "out": str(out_dir),
        },
        "git_sha": _git_sha(),
        "flux_hopf_lib_version": _lib_version(),
        "summary": summaries,
        "rows": long_rows,
        "notes": [s.notes for states in chains.values() for s in states],
    }
    _write_json(json_path, payload)
    plot_mode_compare(long_rows, png_path)

    print(DISCLAIMER)
    print()
    print(f"{NOTES_PREFIX} mode comparison at n-max={args.n_max}; Z frozen; n is not Z")
    print()
    _print_table(
        summaries,
        cols=[
            "step_mode",
            "alias",
            "axis_rule",
            "axis_drift_mean",
            "walk_phase_mean",
            "step_geodesic_mean",
            "identity_overlap_at_nmax",
        ],
    )
    print()
    _print_table(long_rows)
    print()
    print(f"{NOTES_PREFIX} wrote {compare_csv}")
    print(f"{NOTES_PREFIX} wrote {summary_csv}")
    print(f"{NOTES_PREFIX} wrote {json_path}")
    print(verdict)
    return 0


def _run_probe(
    args: argparse.Namespace,
    axis_a: np.ndarray,
    axis_b: np.ndarray,
    out_dir: Path,
    json_path: Path,
) -> int:
    rows, summaries = published_axis_probe(
        n_max=args.n_max,
        angle_rad=float(args.angle_rad),
        axis_a=axis_a,
        axis_b=axis_b,
        seed=int(args.seed),
    )
    unit_ok = all(float(r["unit_norm_error"]) < 1e-6 for r in rows)
    if not unit_ok:
        print(f"{NOTES_PREFIX} probe failed unit-norm check", file=sys.stderr)
        return 2
    probe_csv = out_dir / "homolog_axis_probe.csv"
    _write_csv(probe_csv, rows, COMPARE_COLUMNS)
    overlaps = [float(s["identity_overlap_at_nmax"]) for s in summaries]
    verdict = (
        f"{NOTES_PREFIX} published walking-plane probe: aliases unchanged; Z frozen; "
        "walk_phase stays the insertion angle; n=4 overlap changes with CLI axis. "
        "n=4 numbers are not a locked invariant."
        if abs(overlaps[0] - overlaps[1]) > 1e-6
        else "hypothesis: not supported by this run"
    )
    payload = {
        "disclaimer": DISCLAIMER,
        "hypothesis": (
            "hypothesis: under --step published, a CLI axis with an x-component leaves yz; "
            "walk_phase stays θ; identity_overlap at n-max changes. Not an alkane result."
        ),
        "hypothesis_verdict": verdict,
        "config": {
            "n_max": args.n_max,
            "step": "published",
            "probe_axis": True,
            "axis": [float(x) for x in axis_a],
            "probe_axis_vec": [float(x) for x in axis_b],
            "angle_rad": float(args.angle_rad),
            "seed": int(args.seed),
            "out": str(out_dir),
        },
        "git_sha": _git_sha(),
        "flux_hopf_lib_version": _lib_version(),
        "summary": summaries,
        "rows": rows,
    }
    _write_json(json_path, payload)
    print(DISCLAIMER)
    print()
    print(f"{NOTES_PREFIX} published walking-plane probe; Z frozen; n is not Z")
    print()
    _print_table(
        summaries,
        cols=[
            "cli_axis",
            "alias",
            "axis_drift_mean",
            "walk_phase_mean",
            "identity_overlap_at_nmax",
        ],
    )
    print()
    print(f"{NOTES_PREFIX} wrote {probe_csv}")
    print(f"{NOTES_PREFIX} wrote {json_path}")
    print(verdict)
    return 0


def _run_catalog(
    args: argparse.Namespace,
    catalog_path: Path,
    out_dir: Path,
    catalog_json: Path,
) -> int:
    src = catalog_path if catalog_path.is_absolute() else REPO_ROOT / catalog_path
    rows, meta = run_catalog(
        src,
        angle_rad=float(args.angle_rad),
        seed=int(args.seed),
        shard_index=int(args.shard_index),
        shard_count=int(args.shard_count),
        n_max_override=int(args.n_max),
    )
    if int(args.shard_count) > 1 and not rows:
        # Empty shard is a valid fleet slice, not a failed walk.
        out_dir.mkdir(parents=True, exist_ok=True)
        catalog_json.parent.mkdir(parents=True, exist_ok=True)
        csv_path = (
            catalog_json.with_suffix(".csv")
            if catalog_json.suffix == ".json"
            else out_dir / "homolog_catalog.csv"
        )
        _write_csv(csv_path, [], CATALOG_COLUMNS)
        payload = {
            "disclaimer": DISCLAIMER,
            "hypothesis_verdict": f"{NOTES_PREFIX} empty catalog shard; Z frozen.",
            "config": {
                "catalog": str(src),
                "n_max": args.n_max,
                "angle_rad": float(args.angle_rad),
                "seed": int(args.seed),
                "out": str(catalog_json),
                **meta,
            },
            "git_sha": _git_sha(),
            "flux_hopf_lib_version": _lib_version(),
            "rows": [],
            "reduce": meta.get("reduce"),
        }
        _write_json(catalog_json, payload)
        print(DISCLAIMER)
        print()
        print(
            f"{NOTES_PREFIX} empty shard {args.shard_index}/{args.shard_count}; wrote {catalog_json}"
        )
        return 0
    for row in rows:
        if int(row["n"]) == 1:
            q = np.array([row["q_w"], row["q_x"], row["q_y"], row["q_z"]], dtype=float)
            if not np.allclose(q, IDENTITY_Q, atol=1e-6):
                print(
                    f"{NOTES_PREFIX} identity seed is not q=(1,0,0,0) within 1e-6", file=sys.stderr
                )
                return 2
        if float(row["unit_norm_error"]) >= 1e-6:
            print(f"{NOTES_PREFIX} catalog failed unit-norm check", file=sys.stderr)
            return 2
        if int(row["Z"]) != int(meta["z_frozen"]):
            print(f"{NOTES_PREFIX} catalog changed frozen Z", file=sys.stderr)
            return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    catalog_json.parent.mkdir(parents=True, exist_ok=True)
    csv_path = catalog_json.with_suffix(".csv")
    _write_csv(csv_path, rows, CATALOG_COLUMNS)
    verdict = (
        f"{NOTES_PREFIX} catalog: group ids are insertion words; "
        "molecular names are alias families; Z frozen; n is not Z. "
        "Do not tune θ so a ring word closes."
    )
    payload = {
        "disclaimer": DISCLAIMER,
        "hypothesis_verdict": verdict,
        "config": {
            "catalog": str(src),
            "n_max": args.n_max,
            "angle_rad": float(args.angle_rad),
            "seed": int(args.seed),
            "out": str(catalog_json),
            **meta,
        },
        "git_sha": _git_sha(),
        "flux_hopf_lib_version": _lib_version(),
        "rows": rows,
        "reduce": meta.get("reduce"),
    }
    _write_json(catalog_json, payload)
    print(DISCLAIMER)
    print()
    print(f"{NOTES_PREFIX} insertion-word catalog; Z frozen; n is not Z")
    print()
    _print_table(
        rows,
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
    print(f"{NOTES_PREFIX} wrote {csv_path}")
    print(f"{NOTES_PREFIX} wrote {catalog_json}")
    print(verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
