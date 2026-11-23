"""Figures (matplotlib only)."""
from __future__ import annotations

import os
from typing import Dict, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from PIL import Image  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 13, "axes.grid": False, "figure.facecolor": "white",
    "axes.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False,
})


def _save(fig, path_no_ext: str, png: bool = True) -> None:
    fig.savefig(path_no_ext + ".pdf")
    if png:
        fig.savefig(path_no_ext + ".png", dpi=300)
    plt.close(fig)


def plot_history(history: Dict[str, list], path: str) -> None:
    """Accuracy, loss and learning-rate curves; the epoch with the lowest validation loss is marked."""
    epochs = np.arange(1, len(history["loss"]) + 1)
    best = int(np.argmin(history["val_loss"])) + 1
    lr = history.get("learning_rate", history.get("lr"))
    panels = 3 if lr is not None else 2
    fig, axes = plt.subplots(1, panels, figsize=(5.2 * panels, 4))
    for ax, key, label in ((axes[0], "accuracy", "Accuracy"), (axes[1], "loss", "Loss")):
        ax.plot(epochs, history[key], color="#1f4e9c", lw=1.8, label="Training")
        ax.plot(epochs, history["val_" + key], color="#d1495b", lw=1.8, label="Validation")
        ax.plot(best, history["val_" + key][best - 1], "o", color="black", ms=6, label=f"Lowest val. loss (epoch {best})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.legend(frameon=False, fontsize=11)
    axes[0].set_title("(a) Accuracy")
    axes[1].set_title("(b) Loss")
    if lr is not None:
        axes[2].step(epochs, lr, where="post", color="#2e7d32", lw=1.8)
        axes[2].set_yscale("log")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Learning rate")
        axes[2].set_title("(c) Learning rate")
    for ax in np.ravel(axes):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    _save(fig, path)


def plot_confusion_pair(matrices: Sequence[np.ndarray], titles: Sequence[str], classes: Sequence[str], path: str) -> None:
    fig, axes = plt.subplots(1, len(matrices), figsize=(5.75 * len(matrices), 4.8), gridspec_kw={"wspace": 0.35}, squeeze=False)
    axes = axes[0]
    image = None
    for ax, cm, title in zip(axes, matrices, titles):
        cm = np.asarray(cm)
        pct = 100 * cm / cm.sum(1, keepdims=True)
        image = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for a in range(len(classes)):
            for b in range(len(classes)):
                ax.text(b, a, f"{cm[a, b]}\n({pct[a, b]:.1f}%)", ha="center", va="center", fontsize=11,
                        color="white" if pct[a, b] > 55 else "black")
        ax.set_xticks(range(len(classes)), classes)
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label" if ax is axes[0] else "")
        ax.set_title(title, fontsize=13)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.colorbar(image, ax=list(axes), fraction=0.025).set_label("Row-normalised (%)")
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_split(classes: Sequence[str], totals, train, val, test, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(classes))
    train, val, test = map(np.asarray, (train, val, test))
    ax.bar(x, train, color="#4C72B0", label=f"Train (n = {train.sum():,})")
    ax.bar(x, val, bottom=train, color="#DD8452", label=f"Validation (n = {val.sum():,})")
    ax.bar(x, test, bottom=train + val, color="#55A868", label=f"Test (n = {test.sum():,})")
    for i, total in enumerate(totals):
        ax.text(i, total * 1.01 + 5, f"{total:,}", ha="center")
    ax.set_xticks(x, classes)
    ax.set_ylabel("Number of images")
    ax.set_ylim(0, max(totals) * 1.18)
    ax.legend(frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.13), fontsize=11)
    fig.tight_layout()
    _save(fig, path)


def plot_duplicate_examples(pairs, files, labels, classes, threshold: float, path: str) -> None:
    ordered = sorted(pairs, key=lambda p: -p[2])
    examples = ordered[:6] + [p for p in ordered if p[2] < threshold][:2]
    if not examples:
        return
    fig, axes = plt.subplots(2, len(examples), figsize=(2.2 * len(examples), 4.8), squeeze=False)
    for j, (a, b, r) in enumerate(examples):
        for row, i in enumerate((a, b)):
            axes[row, j].imshow(Image.open(files[i]).convert("RGB").resize((180, 180)))
            axes[row, j].axis("off")
            axes[row, j].set_title(classes[labels[i]] + (f"  (r = {r:.3f})" if row == 0 else ""), fontsize=10)
    fig.tight_layout()
    fig.savefig(path + ".png", dpi=200)
    plt.close(fig)
