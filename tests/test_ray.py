"""Ray parallel integration tests for the CLI scripts."""

import importlib.util
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


def ray_is_available():
    """Check if Ray is installed in the current environment."""
    return importlib.util.find_spec("ray") is not None


# ====================== RAY FLAG & FALLBACK ======================
def test_scripts_accept_use_ray_flag():
    """Both scripts accept --use-ray flag (CLI parsing)."""
    for script in ["epoch_bake_sweep.py", "run_reproduction.py"]:
        cmd = [sys.executable, str(SCRIPTS_DIR / script), "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        assert result.returncode == 0
        assert "--use-ray" in result.stdout.lower()


@pytest.mark.slow
def test_ray_fallback_when_not_installed():
    """If Ray is missing, --use-ray should gracefully fall back to sequential."""
    if ray_is_available():
        pytest.skip("Ray is installed — skipping fallback test")

    for script in ["epoch_bake_sweep.py", "run_reproduction.py"]:
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / script),
            "--trials",
            "2",
            "--dense",
            "--use-ray",  # force Ray path even if not installed
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, f"{script} failed with Ray fallback:\n{result.stderr}"
        # Should contain fallback message (adjust if your exact log differs)
        assert any(
            phrase in result.stderr.lower()
            for phrase in ["fallback", "ray not found", "sequential", "no ray"]
        )


# ====================== RAY PARALLEL EXECUTION (if available) ======================
@pytest.mark.slow
@pytest.mark.skipif(not ray_is_available(), reason="Ray not installed")
def test_ray_parallel_execution():
    """When Ray is installed, --use-ray should run in parallel and succeed."""
    for script in ["epoch_bake_sweep.py", "run_reproduction.py"]:
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / script),
            "--trials",
            "3",  # enough to see parallelism
            "--dense",
            "--use-ray",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, f"Ray parallel run failed for {script}:\n{result.stderr}"
        assert "ray" in result.stderr.lower() or "parallel" in result.stderr.lower()


# ====================== FULL PIPELINE WITH RAY ======================
@pytest.mark.slow
@pytest.mark.skipif(not ray_is_available(), reason="Ray not installed")
def test_full_bake_train_recall_with_ray():
    """End-to-end bake → reproduction using Ray parallelism."""
    # Bake with Ray
    bake_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "epoch_bake_sweep.py"),
        "--trials",
        "3",
        "--dense",
        "--use-ray",
    ]
    result = subprocess.run(bake_cmd, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"Ray bake failed:\n{result.stderr}"

    # Reproduction with Ray
    repro_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "run_reproduction.py"),
        "--trials",
        "3",
        "--dense",
        "--use-ray",
    ]
    result = subprocess.run(repro_cmd, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"Ray reproduction failed:\n{result.stderr}"

    # Verify outputs were created
    assert list((OUTPUTS_DIR / "epoch_bake").glob("epoch_sweep_*.csv")), "Ray bake CSV missing"
    assert list((OUTPUTS_DIR / "reproduction").glob("reproduction_results_*.csv")), (
        "Ray repro CSV missing"
    )


if __name__ == "__main__":
    print("✅ Ray parallel integration tests ready!")
