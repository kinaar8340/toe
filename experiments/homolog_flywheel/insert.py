"""n -> n+1 insertion operator for the homolog_flywheel Model analog.

MODEL, not theorem.
Geometry and arithmetic stay on QGA / flux_hopf_lib.
The alkane / CH2 / carbene language is an analogy for a discrete insertion step.
n=1 is one flywheel at quaternion identity (methane slot).
n=2 is one extra published step: one extra flywheel XOR one extra rotor insertion.
Do not emit “proves”, “element”, “periodic table identity”, or “carbene is a flywheel”.

Exactly one extra published step per call. Three modes (CLI --step):
  rotor     — left-multiply by small_rotor (default).
  flywheel  — append one FluxFlywheel; slot q stays at identity.
  published — wrap map_z_to_flywheel at frozen Z plus apply_golden_angle_increment.
"""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from homolog_flywheel.analog import N_MAX, NOTES_PREFIX, STEP_MODES, alias_for
from homolog_flywheel.state import HomologState

Array = NDArray[np.floating]

QUAT_SOURCE = "flux_hopf_lib.quaternion"
ANGLE_SOURCE = "flux_hopf_lib.constants.GOLDEN_ANGLE_RAD"
FLYWHEEL_SOURCE = "flux_hopf_lib.flux.FluxFlywheel"
ZMAP_SOURCE = "scripts.z_flywheel_map.map_z_to_flywheel"


def _phi() -> float:
    return (1.0 + math.sqrt(5.0)) / 2.0


try:
    from flux_hopf_lib.constants import GOLDEN_ANGLE_RAD as _GOLDEN_ANGLE_RAD
    from flux_hopf_lib.quaternion import q_conj, q_mult, q_normalize, small_rotor

    DEFAULT_ANGLE_RAD = float(_GOLDEN_ANGLE_RAD)
except ImportError:  # pragma: no cover - pin 0.2.2 provides these
    QUAT_SOURCE = "conduit (fallback)"
    ANGLE_SOURCE = "fallback 2π/φ²"
    DEFAULT_ANGLE_RAD = 2.0 * math.pi / _phi() ** 2
    try:
        from conduit import q_conj as _c_q_conj
        from conduit import q_mult as _c_q_mult
        from conduit import q_normalize as _c_q_normalize
        from conduit import small_rotor as _c_small_rotor

        def _as_np(x: Any) -> np.ndarray:
            if hasattr(x, "detach"):
                return np.asarray(x.detach().cpu().numpy(), dtype=float)
            return np.asarray(x, dtype=float)

        def q_mult(q1: Array, q2: Array) -> np.ndarray:  # type: ignore[misc]
            return _as_np(_c_q_mult(q1, q2))

        def q_conj(q: Array) -> np.ndarray:  # type: ignore[misc]
            return _as_np(_c_q_conj(q))

        def q_normalize(q: Array, eps: float = 1e-12) -> np.ndarray:  # type: ignore[misc]
            return _as_np(_c_q_normalize(q))

        def small_rotor(angle_rad: float, axis: Array | None = None) -> np.ndarray:  # type: ignore[misc]
            if axis is None:
                axis = np.array([0.0, 0.0, 1.0])
            return _as_np(_c_small_rotor(angle_rad, axis))
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "analog: need flux_hopf_lib.quaternion or conduit quaternion helpers"
        ) from exc

try:
    from flux_hopf_lib.conduit import apply_golden_angle_increment as _apply_golden
except ImportError:  # pragma: no cover
    _apply_golden = None
    if ANGLE_SOURCE.startswith("flux_hopf_lib"):
        ANGLE_SOURCE = "fallback 2π/φ² (apply_golden_angle_increment missing)"
        DEFAULT_ANGLE_RAD = 2.0 * math.pi / _phi() ** 2

try:
    from flux_hopf_lib.flux import FluxFlywheel as _FluxFlywheel
except ImportError:  # pragma: no cover - shim if pin lacks the class

    @dataclass
    class _FluxFlywheel:  # type: ignore[no-redef]
        quaternion: np.ndarray = field(
            default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        )

    FLYWHEEL_SOURCE = "local shim (FluxFlywheel import failed)"


DEFAULT_AXIS = np.array([0.0, 0.0, 1.0], dtype=float)


def unit_axis(axis: Array | None) -> np.ndarray:
    ax = DEFAULT_AXIS if axis is None else np.asarray(axis, dtype=float).reshape(3)
    n = float(np.linalg.norm(ax))
    if n < 1e-12:
        return DEFAULT_AXIS.copy()
    return ax / n


def make_flywheel(quaternion: Array | None = None) -> Any:
    """One FluxFlywheel (or shim) with the given unit quaternion."""
    q = (
        np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        if quaternion is None
        else q_normalize(np.asarray(quaternion, dtype=float))
    )
    try:
        return _FluxFlywheel(quaternion=q)
    except TypeError:  # pragma: no cover
        wheel = _FluxFlywheel()
        wheel.quaternion = q
        return wheel


def _load_map_z_to_flywheel() -> Callable[..., dict[str, Any]] | None:
    """Load scripts/z_flywheel_map.py without treating experiments as a second package."""
    repo = Path(__file__).resolve().parents[2]
    path = repo / "scripts" / "z_flywheel_map.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("z_flywheel_map_homolog", path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "map_z_to_flywheel", None)
    return fn if callable(fn) else None


_map_z_to_flywheel = _load_map_z_to_flywheel()


def published_phase_rad(step_index: int, angle_rad: float) -> float:
    """Published golden-angle phase for local step_index. Does not use n as Z."""
    if _apply_golden is not None:
        _, phase = _apply_golden(0.0, step_index=int(step_index), mode="golden")
        return float(phase)
    return float((step_index * angle_rad) % (2.0 * math.pi))


def insert(
    state: HomologState,
    *,
    step_mode: str | None = None,
    angle_rad: float | None = None,
    axis: Array | None = None,
) -> HomologState:
    """One n -> n+1 insertion. Exactly one mode per call."""
    mode = step_mode or state.step_mode
    if mode not in STEP_MODES:
        raise ValueError(f"analog: unknown --step {mode!r}; expected one of {STEP_MODES}")
    n_next = state.n + 1
    if n_next > N_MAX:
        raise ValueError(f"analog: n={n_next} exceeds N_MAX={N_MAX}")

    angle = float(state.insertion_angle_rad if angle_rad is None else angle_rad)
    ax = unit_axis(state.insertion_axis if axis is None else axis)
    frozen_z = int(state.Z)
    q = np.array(state.q, dtype=float, copy=True)
    flywheels = list(state.flywheels)
    extra = f"quat={QUAT_SOURCE}; angle={ANGLE_SOURCE}"

    if mode == "rotor":
        rotor = small_rotor(angle, ax)
        q = q_normalize(q_mult(rotor, q))
        notes = (
            f"{NOTES_PREFIX} rotor insertion n={state.n}->{n_next}; "
            f"len(flywheels) stays {len(flywheels)}. {extra}"
        )
    elif mode == "flywheel":
        rotor = small_rotor(angle, ax)
        flywheels.append(make_flywheel(quaternion=rotor))
        notes = (
            f"{NOTES_PREFIX} flywheel append n={state.n}->{n_next}; "
            f"slot q stays at identity; len(flywheels)={len(flywheels)}; "
            f"wheel={FLYWHEEL_SOURCE}. {extra}"
        )
    else:
        # published: wrap map_z_to_flywheel at frozen Z; increment local step_index.
        step_index = int(state.n)
        z_note = f"map_z_to_flywheel unavailable; Z stays {frozen_z}"
        if _map_z_to_flywheel is not None:
            stats = _map_z_to_flywheel(frozen_z)
            z_returned = int(stats.get("Z", frozen_z))
            z_note = (
                f"wrapped {ZMAP_SOURCE}(Z={frozen_z}) returned Z={z_returned}; "
                f"step_index={step_index} is not Z"
            )
        phase = published_phase_rad(step_index, angle)
        rotor = small_rotor(angle, ax)
        q = q_normalize(q_mult(rotor, q))
        notes = (
            f"{NOTES_PREFIX} published insertion n={state.n}->{n_next}; {z_note}; "
            f"apply_golden_angle_increment step_index={step_index} "
            f"golden_phase={phase:.6f}. {extra}"
        )

    return HomologState(
        n=n_next,
        alias=alias_for(n_next),
        q=q,
        flywheels=flywheels,
        step_mode=mode,
        insertion_axis=ax,
        insertion_angle_rad=angle,
        invariants={},
        notes=notes,
        Z=frozen_z,
    )
