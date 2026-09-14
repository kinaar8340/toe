"""Optional overlap-vs-n plot for the homolog_flywheel Model analog.

MODEL, not theorem.
Geometry and arithmetic stay on QGA / flux_hopf_lib.
The alkane / CH2 / carbene language is an analogy for a discrete insertion step.
n=1 is one flywheel at quaternion identity (methane slot).
n=2 is one extra published step: one extra flywheel XOR one extra rotor insertion.
Do not emit “proves”, “element”, “periodic table identity”, or “carbene is a flywheel”.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_overlap_vs_n(rows: list[dict[str, Any]], out_path: Path) -> Path | None:
    """Write analog: identity_overlap vs n. Skip if matplotlib is missing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    ns = [int(r["n"]) for r in rows]
    ys = [float(r["identity_overlap"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ns, ys, marker="o")
    ax.set_xlabel("n (chain index)")
    ax.set_ylabel("identity_overlap")
    ax.set_title("analog: identity_overlap vs n (MODEL, not theorem)")
    ax.set_xticks(ns)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
