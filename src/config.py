"""Strictly typed configuration with runtime validation for the TOE project."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator


class ModelConfig(BaseModel):
    embed_dim: int = Field(default=384, ge=64, le=4096)
    twist_rate: float = Field(default=12.5, gt=0)
    max_depth: float = Field(default=56.0, gt=0)
    num_polarizations: int = Field(default=3, ge=1)
    gauge_strength: float = Field(default=1.0, gt=0)

    @field_validator("embed_dim", "num_polarizations")
    @classmethod
    def must_be_positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be positive")
        return v


class TrainingConfig(BaseModel):
    recon_weight: float = Field(default=4200.0, gt=0)
    learning_rate: float = Field(default=1e-4, gt=0)
    batch_size: int = Field(default=64, ge=1)
    num_epochs: int = Field(default=5, ge=1)
    max_steps: int = Field(default=1000, ge=1)


class DataConfig(BaseModel):
    num_samples: int = Field(default=2048, ge=1)
    seq_len: int = Field(default=128, ge=1)


# IdentityConfig is not yet used in the main Config — made consistent with Pydantic
class IdentityConfig(BaseModel):
    canonical_order: dict[str, list[str]] = Field(default_factory=dict)


class Config(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    # identity: IdentityConfig = Field(default_factory=IdentityConfig)  # uncomment when needed


def load_config(config_path: str | Path | None = None) -> Config:
    """Load config from YAML (or defaults) with full validation."""
    if not config_path or not Path(config_path).exists():
        return Config()

    try:
        with open(config_path, encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return Config.model_validate(raw)
    except (yaml.YAMLError, ValidationError) as e:
        raise ValueError(f"Invalid configuration in {config_path}: {e}") from e
