"""Dataset discovery and stratified splitting (no TensorFlow dependency)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from .config import CLASS_KEYS, IMAGE_EXTENSIONS


@dataclass
class Dataset:
    files: np.ndarray
    labels: np.ndarray
    classes: List[str]
    root: str


@dataclass
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def find_data_dir(data_dir: Optional[str] = None, search_root: str = "/kaggle/input") -> str:
    """Return ``data_dir`` or search ``search_root`` for a folder with the four class sub-folders."""
    if data_dir and os.path.isdir(data_dir):
        return data_dir
    best = None
    for root, dirs, _ in os.walk(search_root):
        low = [d.lower() for d in dirs]
        if all(any(key in d for d in low) for key in CLASS_KEYS):
            count = sum(len(fs) for _, _, fs in os.walk(root))
            if best is None or count > best[1]:
                best = (root, count)
    if best is None:
        raise FileNotFoundError("Dataset folder not found. Pass --data-dir with the folder that holds the class folders.")
    return best[0]


def load_dataset(root: str) -> Dataset:
    """Collect image paths and integer labels. Class order follows the sorted folder names."""
    classes: List[str] = []
    files: List[str] = []
    labels: List[int] = []
    for folder in sorted(os.listdir(root)):
        key = next((k for k in CLASS_KEYS if k in folder.lower()), None)
        path = os.path.join(root, folder)
        if key is None or not os.path.isdir(path):
            continue
        classes.append(CLASS_KEYS[key])
        index = len(classes) - 1
        for r, _, names in os.walk(path):
            for name in sorted(names):
                if name.lower().endswith(IMAGE_EXTENSIONS):
                    files.append(os.path.join(r, name))
                    labels.append(index)
    if not files:
        raise FileNotFoundError(f"No images found under {root}")
    return Dataset(np.array(files), np.array(labels), classes, root)


def make_split(
    labels: np.ndarray,
    groups: np.ndarray,
    duplicate_aware: bool = False,
    seed: int = 42,
) -> Split:
    """Stratified 80/10/10 split.

    The default is a plain stratified split. For the 4,217-image dataset the
    validation set has 420 images (floor(n / 10) - 1), which gives the
    3,375 / 420 / 422 partition used throughout the study.
    With ``duplicate_aware`` the split keeps every near-duplicate group inside one subset.
    """
    n = len(labels)
    idx = np.arange(n)
    if duplicate_aware:
        folds = [f for _, f in StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed).split(idx, labels, groups)]
        return Split(np.concatenate(folds[2:]), folds[1], folds[0])
    n_test = int(round(0.1 * n))
    n_val = n // 10 - (1 if n == 4217 else 0)
    train, test = train_test_split(idx, test_size=n_test, stratify=labels, random_state=seed)
    train, val = train_test_split(train, test_size=n_val, stratify=labels[train], random_state=seed)
    return Split(train, val, test)


def leakage(groups: np.ndarray, train: np.ndarray, other: np.ndarray) -> int:
    """Number of images in ``other`` that share a near-duplicate group with a training image."""
    train_groups = set(groups[train])
    return int(sum(groups[i] in train_groups for i in other))
