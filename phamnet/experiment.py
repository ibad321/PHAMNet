"""Dataset preparation shared by the launcher and the training workers (no TensorFlow)."""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import duplicates, plots
from .config import TrainConfig
from .data import Dataset, Split, find_data_dir, leakage, load_dataset, make_split


@dataclass
class Experiment:
    dataset: Dataset
    groups: np.ndarray
    pairs: list
    split: Split
    split_rows: list


def prepare(cfg: TrainConfig, report: Callable = print, make_figures: bool = False) -> Experiment:
    """Load the data, run the near-duplicate audit and build the train/validation/test split."""
    os.makedirs(cfg.out_dir, exist_ok=True)
    root = find_data_dir(cfg.data_dir)
    report(f"Dataset folder: {root}")
    ds = load_dataset(root)
    counts = np.bincount(ds.labels)
    report(f"Classes (index order): {ds.classes}")
    report(f"Images per class: {dict(zip(ds.classes, counts.tolist()))} | total {len(ds.files)}")
    if not cfg.smoke_test and len(ds.files) != 4217:
        report(f"NOTE: the study dataset has 4,217 images; this folder has {len(ds.files)}.")

    pairs = duplicates.audit(ds.files, os.path.join(cfg.out_dir, "duplicate_audit_cache.json"), report)
    for thr in (0.80, 0.90, 0.95, 0.98):
        sel = [(a, b) for a, b, r in pairs if r >= thr]
        report(f"Pairs with correlation >= {thr:.2f}: {len(sel)} (cross-class: {sum(ds.labels[a] != ds.labels[b] for a, b in sel)})")
    groups = duplicates.group_ids(len(ds.files), pairs, cfg.duplicate_threshold)
    sizes = np.bincount(groups)
    multi = sizes[sizes > 1]
    report(f"Near-duplicate groups (r >= {cfg.duplicate_threshold}): {len(multi)} groups covering {int(multi.sum())} images")

    plain = make_split(ds.labels, groups, False, cfg.split_seed)
    report(f"Plain stratified 80/10/10 split puts {leakage(groups, plain.train, plain.test)} of {len(plain.test)} test images "
           "in a near-duplicate group with a training image.")
    split = make_split(ds.labels, groups, True, cfg.split_seed) if cfg.duplicate_aware_split else plain
    report("Using " + ("duplicate-aware" if cfg.duplicate_aware_split else "plain") + " stratified split.")
    report(f"Leakage after chosen split: test {leakage(groups, split.train, split.test)}, validation {leakage(groups, split.train, split.val)}")

    rows = []
    report(f'{"Class":<10}{"Total":>7}{"Train":>8}{"Val":>7}{"Test":>7}')
    for c, name in enumerate(ds.classes):
        row = [int((ds.labels == c).sum()), int((ds.labels[split.train] == c).sum()),
               int((ds.labels[split.val] == c).sum()), int((ds.labels[split.test] == c).sum())]
        rows.append([name] + row)
        report(f"{name:<10}" + "".join(f"{x:>7}" for x in row))
    report(f'{"Total":<10}{len(ds.files):>7}{len(split.train):>8}{len(split.val):>7}{len(split.test):>7}')

    with open(os.path.join(cfg.out_dir, "split.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "class", "split", "dup_group"])
        for name, ids in (("train", split.train), ("val", split.val), ("test", split.test)):
            for i in ids:
                w.writerow([os.path.relpath(ds.files[i], root), ds.classes[ds.labels[i]], name, int(groups[i])])
    duplicates.write_pairs_csv(os.path.join(cfg.out_dir, "near_duplicate_pairs.csv"), pairs, ds.files, ds.labels, ds.classes, root)

    if make_figures:
        try:
            plots.plot_split(ds.classes, [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows], [r[4] for r in rows],
                             os.path.join(cfg.out_dir, "fig_class_distribution"))
            plots.plot_duplicate_examples(pairs, ds.files, ds.labels, ds.classes, cfg.duplicate_threshold,
                                          os.path.join(cfg.out_dir, "fig_near_duplicate_examples"))
        except Exception as exc:  # figures are optional
            report(f"[figures] skipped: {type(exc).__name__}: {exc}")

    if cfg.smoke_test:
        split = Split(split.train[:48], split.val[:16], split.test[:16])
    return Experiment(ds, groups, pairs, split, rows)
