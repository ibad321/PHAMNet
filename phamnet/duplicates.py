"""Near-duplicate audit (no TensorFlow dependency).

Near-duplicates are found by correlating the high-pass-filtered centre of 96x96
grey thumbnails, also against the mirrored image. Low-frequency content (black
circular mask, vignetting, colour) is removed first because it is shared by
almost every fundus photograph and makes simple perceptual hashes report false
matches.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import io
import json
import os
import time
from collections import Counter
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

Pair = Tuple[int, int, float]


def thumbnail(path: str) -> Tuple[np.ndarray, str]:
    raw = open(path, "rb").read()
    grey = np.asarray(Image.open(io.BytesIO(raw)).convert("L").resize((96, 96), Image.BILINEAR), dtype=np.float32)
    grey = grey - gaussian_filter(grey, 2.5)
    grey = grey[20:76, 20:76]  # the black border of the fundus circle is common to all images
    return (grey - grey.mean()) / (grey.std() + 1e-6), hashlib.md5(raw).hexdigest()


def audit(files: Sequence[str], cache_path: str, report=print, min_correlation: float = 0.80) -> List[Pair]:
    """Return all image pairs with correlation >= ``min_correlation`` (cached on disk)."""
    n = len(files)
    if os.path.exists(cache_path):
        cached = json.load(open(cache_path))
        if cached.get("n") == n:
            report(f"Duplicate audit loaded from cache ({n} images).")
            report(cached["pct_line"])
            report(cached["md5_line"])
            return [tuple(p) for p in cached["pairs"]]
    start = time.time()
    thumbs, md5 = zip(*[thumbnail(p) for p in files])
    a = np.stack([t.ravel() for t in thumbs]).astype(np.float32)
    flipped = np.stack([t[:, ::-1].ravel() for t in thumbs]).astype(np.float32)
    del thumbs
    corr = np.maximum(a @ a.T, a @ flipped.T) / a.shape[1]
    np.fill_diagonal(corr, -1)
    del a, flipped
    report(f"Compared {n} images ({n * (n - 1) // 2:,} pairs) in {time.time() - start:.0f}s")
    values = corr[np.triu_indices(n, 1)]
    pct_line = "Correlation of all image pairs, percentiles 50/99/99.9/99.99: " + " / ".join(
        f"{np.percentile(values, q):.3f}" for q in (50, 99, 99.9, 99.99)
    )
    md5_line = f"Exact byte-identical duplicates: {sum(c - 1 for c in Counter(md5).values() if c > 1)}"
    report(pct_line)
    report(md5_line)
    pairs = [(int(i), int(j), float(corr[i, j])) for i, j in zip(*np.where(np.triu(corr, 1) >= min_correlation))]
    del corr, values
    gc.collect()
    json.dump({"n": n, "pairs": pairs, "pct_line": pct_line, "md5_line": md5_line}, open(cache_path, "w"))
    return pairs


def group_ids(n: int, pairs: Sequence[Pair], threshold: float) -> np.ndarray:
    """Union-find over pairs with correlation >= ``threshold``; returns one group id per image."""
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j, r in pairs:
        if r >= threshold:
            parent[find(i)] = find(j)
    return np.array([find(i) for i in range(n)])


def write_pairs_csv(path: str, pairs: Sequence[Pair], files, labels, classes, root: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_a", "class_a", "file_b", "class_b", "correlation"])
        for i, j, r in sorted(pairs, key=lambda x: -x[2]):
            w.writerow([os.path.relpath(files[i], root), classes[labels[i]], os.path.relpath(files[j], root), classes[labels[j]], r])
