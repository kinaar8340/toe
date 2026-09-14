"""Catalog of insertion words for the homolog_flywheel Model analog.

MODEL, not theorem.
Geometry and arithmetic stay on QGA / flux_hopf_lib.
The alkane / CH2 / carbene language is an analogy for a discrete insertion step.
n=1 is one flywheel at quaternion identity (methane slot).
n=2 is one extra published step: one extra flywheel XOR one extra rotor insertion.
Do not emit “proves”, “element”, “periodic table identity”, or “carbene is a flywheel”.
group ids are insertion words; molecular names are alias families.
"""

from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path
from typing import Any

from homolog_flywheel.analog import DEFAULT_ALIAS_FAMILY, FROZEN_Z
from homolog_flywheel.compare import run_chain
from homolog_flywheel.insert import DEFAULT_ANGLE_RAD, small_rotor, unit_axis
from homolog_flywheel.measure import commutator_norm

CATALOG_PATH = Path(__file__).resolve().parent / "groups.yaml"

CATALOG_COLUMNS = [
    "group_id",
    "alias_family",
    "n",
    "alias",
    "step_mode",
    "axis_rule",
    "identity_overlap",
    "axis_drift_rad",
    "walk_phase_rad",
    "step_geodesic_rad",
    "closure_rad",
    "commutator_norm",
    "word_hash",
    "Z",
]

DEFAULT_CATALOG: dict[str, Any] = {
    "defaults": {
        "z_frozen": FROZEN_Z,
        "theta": "golden",
        "bake": [1.0, 0.0, 0.0],
        "n_max": 4,
    },
    "groups": [
        {
            "id": "linear_rotor",
            "alias_family": "alkane",
            "step": "rotor",
            "axis": [0.0, 0.0, 1.0],
        },
        {
            "id": "linear_published",
            "alias_family": "alkane",
            "step": "published",
            "axis": [0.0, 0.0, 1.0],
        },
        {
            "id": "linear_published_offyz",
            "alias_family": "alkane",
            "step": "published",
            "axis": [1.0, 0.0, 1.0],
        },
        {
            "id": "branch_yz",
            "alias_family": "isoalkane",
            "word": ["rotor", "branch", "rotor"],
            "axes": [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        },
        {
            "id": "ring4_rotor",
            "alias_family": "cyclo",
            "step": "rotor",
            "n_max": 4,
            "metric": "closure",
        },
    ],
}


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        if path.resolve() == CATALOG_PATH.resolve():
            return DEFAULT_CATALOG
        raise
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("analog: catalog yaml must be a mapping")
    return data


def load_catalog(path: Path | None = None) -> dict[str, Any]:
    """Load groups.yaml. Names are alias families, not chemical identities."""
    src = path or CATALOG_PATH
    return _load_yaml(Path(src))


def resolve_theta(token: Any, *, angle_rad: float | None = None) -> float:
    if angle_rad is not None:
        return float(angle_rad)
    if token is None or token == "golden":
        return float(DEFAULT_ANGLE_RAD)
    return float(token)


def word_hash(spec: dict[str, Any]) -> str:
    payload = json.dumps(spec, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _annotate(
    rows: list[dict[str, Any]],
    *,
    group_id: str,
    family: str,
    digest: str,
    comm: float,
) -> list[dict[str, Any]]:
    for row in rows:
        row["group_id"] = group_id
        row["alias_family"] = family
        row["word_hash"] = digest
        row["commutator_norm"] = comm
    return rows


def shard_specs(
    groups: list[dict[str, Any]],
    shard_index: int,
    shard_count: int,
) -> list[dict[str, Any]]:
    """Give group i to shard i % shard_count. analog: shard the catalog, not the narrative."""
    if shard_count < 1:
        raise ValueError("analog: --shard-count must be >= 1")
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"analog: --shard-index must be in 0..{shard_count - 1}")
    return [g for i, g in enumerate(groups) if i % shard_count == shard_index]


def run_group(
    spec: dict[str, Any],
    defaults: dict[str, Any],
    *,
    angle_rad: float,
    seed: int = 0,
    n_max_override: int | None = None,
) -> list[dict[str, Any]]:
    """One catalog word. analog: insertion word, not a molecule."""
    group_id = str(spec["id"])
    family = str(spec.get("alias_family", DEFAULT_ALIAS_FAMILY))
    z = int(spec.get("z_frozen", defaults.get("z_frozen", FROZEN_Z)))
    if "n_max" in spec:
        n_max = int(spec["n_max"])
    elif n_max_override is not None:
        n_max = int(n_max_override)
    else:
        n_max = int(defaults.get("n_max", 4))
    digest = word_hash({"id": group_id, **spec, "theta": angle_rad, "z": z})
    comm = float("nan")

    if spec.get("word"):
        axes = [unit_axis(a) for a in spec.get("axes", [[0.0, 0.0, 1.0]])]
        primary = axes[0]
        if len(axes) >= 2:
            r0 = small_rotor(angle_rad, axes[0])
            r1 = small_rotor(angle_rad, axes[1])
            comm = commutator_norm(r0, r1)
        states = run_chain(
            n_max,
            "rotor",
            angle_rad,
            primary,
            seed=seed,
            Z=z,
            alias_family=family,
            group_id=group_id,
        )
        return _annotate(
            [s.invariants for s in states],
            group_id=group_id,
            family=family,
            digest=digest,
            comm=comm,
        )

    step = str(spec.get("step", "rotor"))
    axis = unit_axis(spec.get("axis", [0.0, 0.0, 1.0]))
    states = run_chain(
        n_max,
        step,
        angle_rad,
        axis,
        seed=seed,
        Z=z,
        alias_family=family,
        group_id=group_id,
    )
    return _annotate(
        [s.invariants for s in states],
        group_id=group_id,
        family=family,
        digest=digest,
        comm=comm,
    )


def catalog_reduce(rows: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    """Structured reduce fields for a later text pass. Not a chemistry claim."""
    aliases_ok = True
    seen: dict[tuple[str, int], str] = {}
    for row in rows:
        key = (str(row["alias_family"]), int(row["n"]))
        alias = str(row["alias"])
        if key in seen and seen[key] != alias:
            aliases_ok = False
        seen[key] = alias
    closures = [float(r["closure_rad"]) for r in rows] if rows else []
    return {
        "host": socket.gethostname(),
        "n_rows": len(rows),
        "modes": sorted({str(r["step_mode"]) for r in rows}),
        "group_ids": sorted({str(r["group_id"]) for r in rows}),
        "max_closure_rad": max(closures) if closures else None,
        "aliases_identical": aliases_ok,
        "z_frozen": int(meta.get("z_frozen", FROZEN_Z)),
        "shard_index": int(meta.get("shard_index", 0)),
        "shard_count": int(meta.get("shard_count", 1)),
    }


def run_catalog(
    path: Path | None = None,
    *,
    angle_rad: float | None = None,
    seed: int = 0,
    shard_index: int = 0,
    shard_count: int = 1,
    n_max_override: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = load_catalog(path)
    defaults = dict(data.get("defaults") or {})
    theta = resolve_theta(defaults.get("theta"), angle_rad=angle_rad)
    groups = list(data.get("groups") or [])
    sliced = shard_specs(groups, shard_index, shard_count)
    rows: list[dict[str, Any]] = []
    for spec in sliced:
        rows.extend(
            run_group(
                spec,
                defaults,
                angle_rad=theta,
                seed=seed,
                n_max_override=n_max_override,
            )
        )
    meta = {
        "defaults": defaults,
        "theta": theta,
        "n_groups": len(groups),
        "n_groups_this_shard": len(sliced),
        "group_ids": [str(g["id"]) for g in sliced],
        "z_frozen": int(defaults.get("z_frozen", FROZEN_Z)),
        "shard_index": shard_index,
        "shard_count": shard_count,
    }
    meta["reduce"] = catalog_reduce(rows, meta)
    return rows, meta
