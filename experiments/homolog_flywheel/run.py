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
from homolog_flywheel.insert import DEFAULT_ANGLE_RAD, DEFAULT_AXIS, insert
from homolog_flywheel.measure import CSV_COLUMNS, hypothesis_verdict, measure
from homolog_flywheel.plot import plot_overlap_vs_n
from homolog_flywheel.seed import IDENTITY_Q, identity_seed
from homolog_flywheel.state import HomologState

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


def build_chain(
    n_max: int,
    step_mode: str,
    angle_rad: float,
    axis: np.ndarray,
    *,
    slow: bool = False,
    seed: int = 0,
    Z: int = 2,
) -> list[HomologState]:
    states = [
        identity_seed(
            step_mode=step_mode,
            insertion_axis=axis,
            insertion_angle_rad=angle_rad,
            Z=Z,
        )
    ]
    for _ in range(1, n_max):
        states.append(
            insert(
                states[-1],
                step_mode=step_mode,
                angle_rad=angle_rad,
                axis=axis,
            )
        )
    for i, state in enumerate(states):
        prev = states[i - 1] if i else None
        state.invariants = measure(state, prev, slow=slow, seed=seed)
    return states


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
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


def _print_table(rows: list[dict[str, Any]]) -> None:
    cols = ["n", "alias", "step_mode", "identity_overlap", "unit_norm_error", "step_geodesic_rad"]
    widths = {c: max(len(c), 8) for c in cols}
    for row in rows:
        for c in cols:
            widths[c] = max(widths[c], len(_fmt(row.get(c))))
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("  ".join("-" * widths[c] for c in cols))
    for row in rows[:10]:
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not N_MIN <= args.n_max <= N_MAX:
        print(f"{NOTES_PREFIX} --n-max must be in {N_MIN}..{N_MAX}", file=sys.stderr)
        return 2
    np.random.seed(args.seed)
    axis = np.asarray(args.axis, dtype=float)
    states = build_chain(
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

    out_dir = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    csv_path = out_dir / "homolog_table.csv"
    json_path = out_dir / "homolog_run.json"
    png_path = out_dir / "homolog_overlap_vs_n.png"
    _write_csv(csv_path, rows)
    verdict = hypothesis_verdict(rows)
    payload = {
        "disclaimer": DISCLAIMER,
        "hypothesis": HYPOTHESIS,
        "hypothesis_verdict": verdict,
        "config": {
            "n_max": args.n_max,
            "step": args.step,
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


if __name__ == "__main__":
    sys.exit(main())
