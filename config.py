"""MMG-MPP configuration."""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    # GIN backbone
    gin_hidden_dim: int = 300
    gin_num_layers: int = 4
    gin_dropout: float = 0.3

    # Transformer
    transformer_hidden_dim: int = 300
    transformer_num_heads: int = 8
    transformer_num_layers: int = 4
    transformer_dropout: float = 0.1

    # GMAE (Fragment branch)
    gmae_mask_ratio: float = 0.3
    gmae_scale_factor: float = 1.2

    # MPH (3D topology branch)
    mph_grid_size: int = 64
    mph_distance_range: tuple = (0.5, 8.0)
    mph_charge_range: tuple = (-0.5, 0.5)
    mph_cnn_kernel: int = 3
    mph_cnn_channels: int = 64
    mph_cnn_layers: int = 3

    # EFF Fusion
    eff_hidden_dim: int = 300

    # General
    embedding_dim: int = 256
    use_3d_branch: bool = True


@dataclass
class PretrainConfig:
    batch_size: int = 256
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    info_nce_tau: float = 0.1
    gmae_loss_weight: float = 0.25


@dataclass
class FinetuneConfig:
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    dropout: float = 0.3
    early_stop_patience: int = 20
    embedding_dim: int = 256


@dataclass
class DataConfig:
    pretrain_data_path: str = "data/chemberta_10m.csv"
    downstream_data_dir: str = "data/moleculenet"
    num_workers: int = 4
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    scaffold_split: bool = True
