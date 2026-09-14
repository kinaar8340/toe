"""Overnight analog grid: inverse words, axis hemisphere, optional --slow.

MODEL, not theorem.
Geometry and arithmetic stay on QGA / flux_hopf_lib.
The alkane / CH2 / carbene language is an analogy for a discrete insertion step.
n=1 is one flywheel at quaternion identity (methane slot).
n=2 is one extra published step: one extra flywheel XOR one extra rotor insertion.
Do not emit “proves”, “element”, “periodic table identity”, or “carbene is a flywheel”.
group ids are insertion words; molecular names are alias families.

Keep groups.yaml and golden θ frozen. Record closure_rad(θ); do not fit ring4.
"""

from __future__ import annotations

import argparse
import math
import socket
import sys
from pathlib import Path
from typing import Any

import numpy as np

from homolog_flywheel.analog import DISCLAIMER, FROZEN_Z, N_MAX, NOTES_PREFIX, alias_for
from homolog_flywheel.catalog import analog_run_verdict, word_hash
from homolog_flywheel.compare import run_chain
from homolog_flywheel.insert import (
    DEFAULT_ANGLE_RAD,
    DEFAULT_AXIS,
    q_mult,
    q_normalize,
    small_rotor,
    unit_axis,
)
from homolog_flywheel.measure import commutator_norm
from homolog_flywheel.run import _print_table, _write_csv, _write_json
from homolog_flywheel.seed import IDENTITY_Q

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN = float(DEFAULT_ANGLE_RAD)

GRID_COLUMNS = [
    "group_id",
    "n",
    "direction",
    "axis_x",
    "axis_y",
    "axis_z",
    "theta_id",
    "theta_rad",
    "identity_overlap",
    "closure_rad",
    "step_geodesic_rad",
    "commutator_norm",
    "roundtrip_overlap",
    "lt_survival",
    "geometric_winding",
    "Z",
]

# analog: shard jobs for bud2-bud9. Fill empty witnesses with grid slices.
SHARD_JOBS: dict[int, dict[str, Any]] = {
    0: {"group_id": "linear_rotor", "step": "rotor", "scan": "axis_inverse"},
    1: {"group_id": "linear_published", "step": "published", "scan": "axis_inverse"},
    2: {"group_id": "linear_published_offyz", "step": "published", "scan": "axis_grid"},
    3: {"group_id": "branch_yz", "scan": "branch_pairs"},
    4: {
        "group_id": "ring4_rotor",
        "step": "rotor",
        "scan": "theta_sample",
        "slow": True,
        "n_max": 4,
    },
    5: {"group_id": "published_walk", "step": "published", "scan": "axis_seeds"},
    6: {"group_id": "linear_rotor", "step": "rotor", "scan": "slow_word", "slow": True},
    7: {"group_id": "linear_published", "step": "published", "scan": "slow_word", "slow": True},
}


def fibonacci_axes(n: int) -> list[np.ndarray]:
    """Unit axes on S² (golden spiral). analog: hemisphere/sphere grid, not chemistry."""
    n = max(int(n), 1)
    out: list[np.ndarray] = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        y = 1.0 - 2.0 * (i + 0.5) / n
        r = math.sqrt(max(0.0, 1.0 - y * y))
        phi = i * golden
        out.append(unit_axis(np.array([r * math.cos(phi), y, r * math.sin(phi)])))
    return out


def theta_sample_grid(n_samples: int) -> list[tuple[str, float]]:
    """Record θ values including frozen golden. analog: a curve, not a fit."""
    n_samples = max(int(n_samples), 1)
    lo, hi = 0.4, math.pi
    grid = [lo + (hi - lo) * i / max(n_samples - 1, 1) for i in range(n_samples)]
    tagged = [(f"sample_{i}", float(t)) for i, t in enumerate(grid)]
    tagged.append(("golden", GOLDEN))
    return tagged


def roundtrip_overlap(states: list[Any]) -> float:
    """Apply -theta on the forward axis sequence, reversed. analog: reversibility."""
    if len(states) < 2:
        return 1.0
    q = np.array(states[-1].q, dtype=float, copy=True)
    for st in reversed(states[1:]):
        angle = float(st.insertion_angle_rad)
        rotor = small_rotor(-angle, st.insertion_axis)
        q = q_normalize(q_mult(rotor, q))
    return float(abs(np.dot(q, IDENTITY_Q)))


def _annotate_rows(
    states: list[Any],
    *,
    group_id: str,
    direction: str,
    theta_id: str,
    theta_rad: float,
    roundtrip: float,
) -> list[dict[str, Any]]:
    rows = []
    for st in states:
        row = dict(st.invariants)
        row["group_id"] = group_id
        row["direction"] = direction
        row["theta_id"] = theta_id
        row["theta_rad"] = float(theta_rad)
        row["roundtrip_overlap"] = float(roundtrip)
        row["Z"] = int(st.Z)
        rows.append(row)
    return rows


def run_signed_chain(
    *,
    step: str,
    n_max: int,
    axis: np.ndarray,
    theta: float,
    direction: str,
    group_id: str,
    family: str,
    slow: bool,
    seed: int,
) -> list[dict[str, Any]]:
    signed = float(theta) if direction == "+theta" else -float(theta)
    states = run_chain(
        n_max,
        step,
        signed,
        unit_axis(axis),
        slow=slow,
        seed=seed,
        Z=FROZEN_Z,
        alias_family=family,
        group_id=group_id,
    )
    rt = roundtrip_overlap(states) if direction == "+theta" else float("nan")
    return _annotate_rows(
        states,
        group_id=group_id,
        direction=direction,
        theta_id="golden" if abs(abs(theta) - GOLDEN) < 1e-9 else "sample",
        theta_rad=signed,
        roundtrip=rt,
    )


def run_shard_job(
    shard_index: int,
    *,
    n_max: int = 10,
    n_axes: int = 8,
    n_theta: int = 8,
    n_seeds: int = 4,
    slow: bool = False,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if shard_index not in SHARD_JOBS:
        raise ValueError(f"analog: shard_index {shard_index} not in 0..7")
    job = SHARD_JOBS[shard_index]
    group_id = str(job["group_id"])
    scan = str(job["scan"])
    step = str(job.get("step", "rotor"))
    job_slow = bool(slow) and bool(job.get("slow", False))
    job_n_max = int(job.get("n_max", n_max))
    job_n_max = min(job_n_max, N_MAX)
    axes = fibonacci_axes(n_axes)
    rows: list[dict[str, Any]] = []

    if scan == "axis_inverse":
        family = "alkane"
        for axis in axes:
            for direction in ("+theta", "-theta"):
                rows.extend(
                    run_signed_chain(
                        step=step,
                        n_max=job_n_max,
                        axis=axis,
                        theta=GOLDEN,
                        direction=direction,
                        group_id=group_id,
                        family=family,
                        slow=job_slow and direction == "+theta",
                        seed=seed,
                    )
                )
    elif scan == "axis_grid":
        for axis in axes:
            rows.extend(
                run_signed_chain(
                    step="published",
                    n_max=job_n_max,
                    axis=axis,
                    theta=GOLDEN,
                    direction="+theta",
                    group_id=group_id,
                    family="alkane",
                    slow=False,
                    seed=seed,
                )
            )
    elif scan == "branch_pairs":
        pair_axes = axes[: max(4, min(len(axes), 8))]
        frozen = [DEFAULT_AXIS, np.array([0.0, 1.0, 0.0])]
        pairs = [frozen] + [
            [pair_axes[i], pair_axes[(i + 1) % len(pair_axes)]] for i in range(len(pair_axes))
        ]
        for a, b in pairs:
            r0 = small_rotor(GOLDEN, unit_axis(a))
            r1 = small_rotor(GOLDEN, unit_axis(b))
            comm = commutator_norm(r0, r1)
            chain = run_signed_chain(
                step="rotor",
                n_max=job_n_max,
                axis=a,
                theta=GOLDEN,
                direction="+theta",
                group_id=group_id,
                family="isoalkane",
                slow=False,
                seed=seed,
            )
            for row in chain:
                row["commutator_norm"] = comm
                row["axis_x"], row["axis_y"], row["axis_z"] = (float(x) for x in unit_axis(a))
            rows.extend(chain)
    elif scan == "theta_sample":
        # analog: record closure_rad(θ). Do not argmin and call it cyclobutane.
        for theta_id, theta in theta_sample_grid(n_theta):
            chain = run_signed_chain(
                step="rotor",
                n_max=4,
                axis=DEFAULT_AXIS,
                theta=theta,
                direction="+theta",
                group_id=group_id,
                family="cyclo",
                slow=job_slow,
                seed=seed,
            )
            for row in chain:
                row["theta_id"] = theta_id
                row["alias"] = alias_for(int(row["n"]), family="cyclo")
            rows.extend(chain)
    elif scan == "axis_seeds":
        for s in range(max(1, n_seeds)):
            rng = np.random.default_rng(seed + s)
            axis = unit_axis(rng.normal(size=3))
            rows.extend(
                run_signed_chain(
                    step="published",
                    n_max=job_n_max,
                    axis=axis,
                    theta=GOLDEN,
                    direction="+theta",
                    group_id=group_id,
                    family="alkane",
                    slow=False,
                    seed=seed + s,
                )
            )
    elif scan == "slow_word":
        rows.extend(
            run_signed_chain(
                step=step,
                n_max=job_n_max,
                axis=DEFAULT_AXIS,
                theta=GOLDEN,
                direction="+theta",
                group_id=group_id,
                family="alkane",
                slow=True,
                seed=seed,
            )
        )
        rows.extend(
            run_signed_chain(
                step=step,
                n_max=job_n_max,
                axis=DEFAULT_AXIS,
                theta=GOLDEN,
                direction="-theta",
                group_id=group_id,
                family="alkane",
                slow=False,
                seed=seed,
            )
        )
    else:
        raise ValueError(f"analog: unknown scan {scan!r}")

    meta = {
        "disclaimer": DISCLAIMER,
        "host": socket.gethostname(),
        "shard_index": shard_index,
        "shard_count": 8,
        "job": job,
        "n_axes": n_axes,
        "n_max": job_n_max,
        "golden_theta": GOLDEN,
        "z_frozen": FROZEN_Z,
        "notes": (
            f"{NOTES_PREFIX} grid sweep. Frozen golden θ is the reference. "
            "closure_rad(θ) is a curve, not a fit. n is not Z."
        ),
        "hypothesis_verdict": analog_run_verdict(rows)
        if rows
        else f"{NOTES_PREFIX} empty grid shard",
        "word_hash": word_hash({"job": job, "n_max": job_n_max, "n_axes": n_axes}),
    }
    return rows, meta


def fleet_night1_command(*, n_max: int, n_axes: int) -> str:
    return (
        "mkdir -p $HOME/Playground/results/night1; "
        "host=$(hostname); i=${host#bud}; "
        "PYTHONPATH=$HOME/Projects/toe/src:$HOME/Projects/toe/experiments "
        "$HOME/Projects/toe/venv/bin/python -m homolog_flywheel.grid_sweep "
        "--shard-index $((10#$i - 2)) --shard-count 8 "
        f"--n-max {n_max} --n-axes {n_axes} --slow "
        "--out $HOME/Playground/results/night1/homolog_$host.json"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m homolog_flywheel.grid_sweep",
        description="Overnight analog grid: inverse, axes, optional --slow. Do not fit θ.",
        epilog=DISCLAIMER,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--n-max", type=int, default=N_MAX)
    parser.add_argument("--n-axes", type=int, default=8)
    parser.add_argument("--n-theta", type=int, default=8)
    parser.add_argument("--n-seeds", type=int, default=4)
    parser.add_argument("--slow", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "experiments" / "outputs" / "homolog_grid_shard0.json"),
    )
    parser.add_argument(
        "--emit-fleet",
        action="store_true",
        help="print bin/fleet night1 command; does not SSH",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="tiny local shard 0, no --slow",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(DISCLAIMER)
    print()
    if args.emit_fleet:
        print("# analog: SSH fleet runs Python grid walks, not grok -p")
        print("# This flag only PRINTS. It does not SSH.")
        print("# Frozen golden θ is the reference. Do not retune ring4.")
        print("cd ~/Playground")
        print(
            f"bin/fleet run --hosts bud2,bud3,bud4,bud5,bud6,bud7,bud8,bud9 -- '{fleet_night1_command(n_max=int(args.n_max), n_axes=int(args.n_axes))}'"
        )
        print()
        print(f"{NOTES_PREFIX} --emit-fleet did not contact bud2-bud9.")
        return 0

    shard = 0 if args.dry_run else int(args.shard_index)
    n_axes = 4 if args.dry_run else int(args.n_axes)
    n_max = 4 if args.dry_run else min(int(args.n_max), N_MAX)
    slow = False if args.dry_run else bool(args.slow)
    if not 0 <= shard < 8:
        print(f"{NOTES_PREFIX} --shard-index must be 0..7", file=sys.stderr)
        return 2
    rows, meta = run_shard_job(
        shard,
        n_max=n_max,
        n_axes=n_axes,
        n_theta=int(args.n_theta) if not args.dry_run else 3,
        n_seeds=int(args.n_seeds) if not args.dry_run else 2,
        slow=slow,
        seed=int(args.seed),
    )
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    payload = {**meta, "rows": rows, "n_rows": len(rows)}
    _write_json(out, payload)
    _write_csv(out.with_suffix(".csv"), rows, GRID_COLUMNS)
    print(f"{NOTES_PREFIX} grid shard {shard}/8 job={meta['job']['scan']} n_rows={len(rows)}")
    _print_table(
        rows[:24],
        cols=[
            "group_id",
            "n",
            "direction",
            "theta_id",
            "identity_overlap",
            "closure_rad",
            "roundtrip_overlap",
        ],
    )
    print()
    print(f"{NOTES_PREFIX} wrote {out}")
    print(meta["hypothesis_verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
