# src/config.py — v1.0.0 (April 17, 2026)
# Fully config-driven with strong emphasis on security.
# Global topological features (winding, linking, braiding phases + ShellCube differential) remain primary.

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    embed_dim: int = 384
    twist_rate: float = 12.5
    max_depth: float = 56.0
    num_polarizations: int = 18  # ← best from sweep
    quat_logical_dim: int = 96

    # ── New epoch-sync hyperparameters ──
    num_layers: int = 3  # ← best from sweep
    max_facts: int = 30  # ← best from sweep (30-42 range)
    gauge_strength: float = 0.86  # ← best from sweep
    omega_R: float = 0.0225  # ← best from sweep (0.0215-0.0235)


@dataclass
class TrainingConfig:
    # Strong local fidelity (these must dominate)
    recon_weight: float = 4200.0
    mag_weight: float = 80.0
    align_weight: float = 1800.0

    # Very gentle global topological guidance
    winding_weight: float = 0.048
    locality_weight: float = 12.0
    braiding_weight: float = 0.018

    # Depth pull
    depth_pull_weight: float = 5.2

    # Orthogonality & preservation
    ortho_weight: float = 5.0
    cos_margin: float = 0.25
    cos_pres_active_delta_s: float = 1.5

    # Read / write parameters
    locality_strength: float = 1.50
    kernel_sigma: float = 0.32
    read_bandwidth_factor: float = 2.4

    # Optimization
    grad_clip_max_norm: float = 1.15
    learning_rate: float = 2.8e-4
    weight_decay: float = 1.3e-4

    # Logging / eval
    eval_every: int = 20
    vis_every: int = 100
    save_best_recall: bool = True

    # Paths
    checkpoint_dir: str = "checkpoints"


@dataclass
class DataConfig:
    num_samples: int = 2048
    batch_size: int = 16
    n_clusters: int = 8
    depth_span_per_cluster: float = 9.0
    intra_cluster_noise: float = 0.20
    drift_strength: float = 0.018
    noise_strength: float = 0.052
    norm_target: float = 0.95


@dataclass
class IdentityConfig:
    canonical_order: dict[str, list] = field(default_factory=dict)


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)

    device: str = "auto"
    seed: int = 42
    epochs: int = 200


def load_config(config_path: str = "configs/default.yaml") -> Config:
    """Robust YAML loader with full section merging.
    Environment variables take priority for sensitive fields."""
    path = Path(config_path)
    if not path.exists():
        print(f"Config {config_path} not found → using defaults")
        return Config()

    with path.open("r") as f:
        raw = yaml.safe_load(f) or {}

    cfg = Config()

    # Merge each top-level section (DRY, extensible)
    sections = [
        ("model", cfg.model),
        ("data", cfg.data),
        ("training", cfg.training),
        ("identity", cfg.identity),
    ]

    for section_name, target in sections:
        if section_name in raw:
            for k, v in raw[section_name].items():
                if hasattr(target, k):
                    setattr(target, k, v)
                else:
                    target.__dict__[k] = v

    return cfg
