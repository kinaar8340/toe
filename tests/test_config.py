"""Tests for configuration loading and dataclasses."""

import sys
from pathlib import Path

# Temporary src-layout fix for PyCharm + pytest
project_root = Path(__file__).parent.parent
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

from config import ModelConfig, load_config


def test_default_config():
    cfg = load_config("configs/default.yaml")
    assert cfg.model.embed_dim == 384
    assert cfg.training.recon_weight == 4200.0
    assert cfg.data.num_samples == 2048


def test_load_config_defaults_when_missing():
    cfg = load_config("nonexistent.yaml")
    assert cfg.model.twist_rate == 12.5


def test_model_config_dataclass():
    m = ModelConfig(embed_dim=512, twist_rate=25.0)
    assert m.embed_dim == 512
    assert m.twist_rate == 25.0
