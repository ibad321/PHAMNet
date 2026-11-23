"""Configuration: hyper-parameters, model variants and run presets.

This module has no TensorFlow dependency so that the launcher and the tests can
import it without initialising a GPU.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

CLASS_KEYS: Dict[str, str] = {
    "cataract": "Cataract",
    "diabetic": "DR",
    "glaucoma": "Glaucoma",
    "normal": "Normal",
}
IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class TrainConfig:
    """Training and evaluation settings shared by every model variant."""

    data_dir: Optional[str] = None
    out_dir: str = "phamnet_results"
    image_size: int = 224
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 3e-4
    early_stopping_patience: int = 30
    lr_patience: int = 5
    lr_factor: float = 0.5
    min_lr: float = 1e-6
    dropout: float = 0.45
    dense_units: int = 256
    l2: float = 0.016
    activity_l1: float = 0.006
    split_seed: int = 42
    duplicate_threshold: float = 0.95
    duplicate_aware_split: bool = False
    bootstrap_resamples: int = 5000
    smoke_test: bool = False
    runs: List[str] = field(default_factory=lambda: ["b4_tl", "phamnet"])
    seeds: List[int] = field(default_factory=lambda: [42])

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def from_json(path: str) -> "TrainConfig":
        with open(path) as f:
            return TrainConfig(**json.load(f))


@dataclass(frozen=True)
class Variant:
    """Architecture of one model variant."""

    stem: bool = False            # multiscale feature fusion stem in front of the backbone
    attention: Optional[str] = None  # None, sab, gcab, ham, ham_fixed, cbam, bam
    sab_kernel: int = 7
    linear_projection: bool = False  # MFFS 1x1 projection without BatchNorm/ReLU
    backbone: str = "b4"          # b4 or vgg16
    learning_rate: Optional[float] = None  # overrides TrainConfig.learning_rate


VARIANTS: Dict[str, Variant] = {
    "b4_tl": Variant(),
    "b4_sab": Variant(attention="sab"),
    "b4_gcab": Variant(attention="gcab"),
    "b4_ham": Variant(attention="ham"),
    "mffs_b4": Variant(stem=True),
    "phamnet": Variant(stem=True, attention="ham"),
    "b4_ham_fixed": Variant(stem=True, attention="ham_fixed"),
    "phamnet_k3": Variant(stem=True, attention="ham", sab_kernel=3),
    "phamnet_linproj": Variant(stem=True, attention="ham", linear_projection=True),
    "b4_cbam": Variant(attention="cbam"),
    "b4_bam": Variant(attention="bam"),
    "vgg16_tl": Variant(backbone="vgg16", learning_rate=1e-4),
}

# name -> (models, seeds)
PRESETS: Dict[str, Tuple[List[str], List[int]]] = {
    "main": (["b4_tl", "b4_sab", "phamnet"], [42, 7, 2024]),
    "main_a": (["b4_tl", "b4_sab"], [42, 7, 2024]),
    "main_b": (["phamnet"], [42, 7, 2024]),
    "ablation": (
        ["b4_gcab", "b4_ham", "mffs_b4", "b4_cbam", "b4_bam", "phamnet_k3", "phamnet_linproj", "vgg16_tl"],
        [42],
    ),
}
