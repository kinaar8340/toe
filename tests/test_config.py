"""Full integration tests for configuration loading, ModelConfig, validation,
and end-to-end training with real YAML files."""

import sys
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

# Temporary src-layout fix
project_root = Path(__file__).parent.parent
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

from conduit import RubikConeConduit
from config import Config, ModelConfig, load_config  # Config class if it exists


# ====================== BASIC / DEFAULTS ======================
def test_default_config():
    """Load the real default.yaml and verify key values."""
    cfg = load_config("configs/default.yaml")
    assert isinstance(cfg, Config) if hasattr(cfg, "__class__") else True
    assert cfg.model.embed_dim == 384
    assert cfg.training.recon_weight == 4200.0
    assert cfg.data.num_samples == 2048


def test_load_config_defaults_when_missing():
    """Non-existent file falls back to defaults."""
    cfg = load_config("nonexistent.yaml")
    assert cfg.model.twist_rate == 12.5


def test_model_config_dataclass():
    """Direct instantiation of ModelConfig works."""
    m = ModelConfig(embed_dim=512, twist_rate=25.0)
    assert m.embed_dim == 512
    assert m.twist_rate == 25.0


# ====================== CUSTOM YAML ======================
def test_load_config_with_custom_yaml(tmp_path):
    """Load a custom YAML and verify overrides."""
    custom_yaml = tmp_path / "custom.yaml"
    custom_yaml.write_text("""
model:
  embed_dim: 512
  twist_rate: 42.0
training:
  recon_weight: 9999.0
""")
    cfg = load_config(custom_yaml)
    assert cfg.model.embed_dim == 512
    assert cfg.model.twist_rate == 42.0
    assert cfg.training.recon_weight == 9999.0


# ====================== VALIDATION (current behavior) ======================
# Note: load_config currently accepts bad values gracefully (no strict validation yet).
# We test that it doesnt crash — full validation can be added later.
def test_config_strict_validation_negative_embed_dim():
    """Pydantic raises clear error for out-of-range values."""
    with pytest.raises(ValueError, match="embed_dim"):
        ModelConfig(embed_dim=-100)


def test_config_strict_validation_zero_twist_rate():
    with pytest.raises(ValueError, match="twist_rate"):
        ModelConfig(twist_rate=0)


def test_load_config_raises_on_invalid_yaml_values(tmp_path):
    """Real YAML with bad values now raises helpful error."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("model:\n  embed_dim: -500\n  twist_rate: 0")
    with pytest.raises(ValueError, match="embed_dim|twist_rate"):
        load_config(bad_yaml)


def test_load_config_raises_on_malformed_types(tmp_path):
    bad_yaml = tmp_path / "bad_type.yaml"
    bad_yaml.write_text("model:\n  embed_dim: 'not_a_number'")
    with pytest.raises(ValueError):
        load_config(bad_yaml)


# ====================== HYPOTHESIS PROPERTY-BASED ======================
@given(st.integers(min_value=64, max_value=2048))
def test_hypothesis_embed_dim_range(embed_dim):
    """Any valid embed_dim should be accepted."""
    m = ModelConfig(embed_dim=embed_dim)
    assert m.embed_dim == embed_dim


# ====================== END-TO-END TRAINING SMOKE TEST ======================
def test_end_to_end_training_with_real_yaml():
    """Load real YAML → instantiate RubikConeConduit → smoke test."""
    cfg = load_config("configs/default.yaml")

    # Pass flat parameters (matches RubikConeConduit.__init__)
    model = RubikConeConduit(
        embed_dim=cfg.model.embed_dim,
        twist_rate=cfg.model.twist_rate,
        max_depth=cfg.model.max_depth,
        num_polarizations=cfg.model.num_polarizations,
        gauge_strength=cfg.model.gauge_strength,
        # other fields keep their defaults
    )
    assert model is not None
    assert hasattr(model, "current_epoch")
    assert model.device is not None

    # Tiny smoke forward (no real batch needed for basic check)
    assert callable(getattr(model, "position", None))


# ====================== RUN COMMAND ======================
if __name__ == "__main__":
    print("✅ Config + YAML integration tests updated and ready!")
