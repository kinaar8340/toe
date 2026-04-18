"""End-to-end smoke tests for the training CLI scripts (matches real CLI)."""

import glob
import subprocess
import sys
from pathlib import Path

import pytest

# Temporary src-layout fix
project_root = Path(__file__).parent.parent
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

SCRIPTS_DIR = project_root / "scripts"
OUTPUTS_DIR = project_root / "outputs"


@pytest.mark.slow
def test_epoch_bake_sweep_smoke():
    """Run epoch_bake_sweep.py with tiny trial count and verify output CSV."""
    # Clean any previous test files (optional)
    for f in glob.glob(str(OUTPUTS_DIR / "epoch_bake" / "epoch_sweep_*.csv")):
        Path(f).unlink(missing_ok=True)

    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "epoch_bake_sweep.py"),
        "--trials",
        "2",  # tiny for smoke test
        "--dense",  # optional but fast grid
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, f"Script failed:\n{result.stderr}"

    # Check that a timestamped CSV was created
    csv_files = list((OUTPUTS_DIR / "epoch_bake").glob("epoch_sweep_*.csv"))
    assert len(csv_files) >= 1, "No epoch_sweep CSV was generated"
    print(f"✅ epoch_bake_sweep produced {csv_files[-1]}")


@pytest.mark.slow
def test_run_reproduction_smoke():
    """Run run_reproduction.py with minimal trials."""
    for f in glob.glob(str(OUTPUTS_DIR / "reproduction" / "reproduction_results_*.csv")):
        Path(f).unlink(missing_ok=True)

    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "run_reproduction.py"),
        "--trials",
        "2",
        "--dense",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, f"Script failed:\n{result.stderr}"

    csv_files = list((OUTPUTS_DIR / "reproduction").glob("reproduction_results_*.csv"))
    assert len(csv_files) >= 1, "No reproduction_results CSV was generated"
    print(f"✅ run_reproduction produced {csv_files[-1]}")


def test_scripts_accept_use_ray_flag():
    """Make sure --use-ray is accepted (even if Ray not installed)."""
    for script in ["epoch_bake_sweep.py", "run_reproduction.py"]:
        cmd = [sys.executable, str(SCRIPTS_DIR / script), "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        assert "--use-ray" in result.stdout


# ====================== FULL BAKE → REPRODUCTION CYCLE ======================
@pytest.mark.slow
def test_full_bake_train_recall_cycle():
    """Run bake → reproduction pipeline end-to-end (tiny trials)."""
    # Bake
    bake_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "epoch_bake_sweep.py"),
        "--trials",
        "2",
        "--dense",
    ]
    result = subprocess.run(bake_cmd, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, f"Bake failed:\n{result.stderr}"

    # Reproduction (imports from bake script)
    repro_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "run_reproduction.py"),
        "--trials",
        "2",
        "--dense",
    ]
    result = subprocess.run(repro_cmd, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, f"Reproduction failed:\n{result.stderr}"

    # Verify both CSVs exist
    assert list((OUTPUTS_DIR / "epoch_bake").glob("epoch_sweep_*.csv")), "Bake CSV missing"
    assert list((OUTPUTS_DIR / "reproduction").glob("reproduction_results_*.csv")), (
        "Repro CSV missing"
    )


if __name__ == "__main__":
    print("✅ End-to-end script smoke tests updated and ready!")
