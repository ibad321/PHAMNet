"""Evaluation metrics (no TensorFlow dependency)."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import binomtest
from sklearn.metrics import (cohen_kappa_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    """Exact binomial confidence interval for k successes out of n."""
    lo = 0.0 if k == 0 else beta_dist.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta_dist.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    pred = probs.argmax(1)
    conf = probs.max(1)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            ece += sel.mean() * abs((pred[sel] == y[sel]).mean() - conf[sel].mean())
    return float(ece)


def evaluate(y: np.ndarray, probs: np.ndarray, classes: Sequence[str], resamples: int = 5000, seed: int = 0) -> Dict:
    """Test-set metrics with bootstrap confidence intervals."""
    num = len(classes)
    pred = probs.argmax(1)
    n = len(y)
    cm = confusion_matrix(y, pred, labels=range(num))
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    tn = n - tp - fp - fn

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(resamples):
        i = rng.integers(0, n, n)
        boot.append(((pred[i] == y[i]).mean(), f1_score(y[i], pred[i], average="macro"), matthews_corrcoef(y[i], pred[i])))
    boot = np.array(boot) if boot else np.zeros((1, 3))

    def interval(col: int):
        return [float(np.percentile(boot[:, col], 2.5)), float(np.percentile(boot[:, col], 97.5))]

    auc = []
    for k in range(num):
        positive = (y == k).astype(int)
        auc.append(float(roc_auc_score(positive, probs[:, k])) if 0 < positive.sum() < n else float("nan"))

    def safe_div(a, b):
        return float(a / max(b, 1))

    return {
        "n": int(n),
        "correct": int((pred == y).sum()),
        "accuracy": float((pred == y).mean()),
        "accuracy_ci": interval(0),
        "accuracy_ci_exact": list(clopper_pearson(int((pred == y).sum()), int(n))),
        "macro_precision": float(precision_score(y, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "macro_f1_ci": interval(1),
        "mcc": float(matthews_corrcoef(y, pred)),
        "mcc_ci": interval(2),
        "kappa": float(cohen_kappa_score(y, pred)),
        "per_class": {
            classes[k]: {
                "sensitivity": safe_div(tp[k], tp[k] + fn[k]),
                "specificity": safe_div(tn[k], tn[k] + fp[k]),
                "ppv": safe_div(tp[k], tp[k] + fp[k]),
                "npv": safe_div(tn[k], tn[k] + fn[k]),
                "f1": safe_div(2 * tp[k], 2 * tp[k] + fp[k] + fn[k]),
                "auroc": auc[k],
            }
            for k in range(num)
        },
        "macro_auroc": float(np.nanmean(auc)),
        "ece_15bins": expected_calibration_error(probs, y),
        "brier": float(np.mean(np.sum((probs - np.eye(num)[y]) ** 2, 1))),
        "confusion_matrix": cm.tolist(),
    }


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray):
    """Exact McNemar test on paired correctness vectors. Returns (only_a, only_b, p_value)."""
    only_a = int((correct_a & ~correct_b).sum())
    only_b = int((~correct_a & correct_b).sum())
    d = only_a + only_b
    p = binomtest(min(only_a, only_b), d, 0.5).pvalue if d else 1.0
    return only_a, only_b, float(p)
