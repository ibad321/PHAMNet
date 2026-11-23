"""Train and evaluate one model variant with one seed (runs inside a worker process)."""
from __future__ import annotations

import gc
import json
import math
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras as K

from . import metrics, plots
from .config import TrainConfig
from .experiment import Experiment
from .model import build_model
from .pipeline import make_dataset


class LearningRateLogger(K.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            lr = self.model.optimizer.learning_rate
            logs["learning_rate"] = float(lr.numpy() if hasattr(lr, "numpy") else K.backend.get_value(lr))


def measure_latency(model: K.Model, size: int) -> float:
    """Median single-image forward-pass time in milliseconds (eager timing of a compiled function)."""
    x = tf.random.normal([1, size, size, 3]) * 60 + 120
    fn = tf.function(lambda z: model(z, training=False))
    for _ in range(10):
        fn(x)
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        fn(x).numpy()
        times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1000)


def run_one(run: str, seed: int, cfg: TrainConfig, exp: Experiment, report=print) -> dict:
    """Train ``run`` with ``seed``. Existing results are loaded instead of retrained."""
    tag = f"{run}_seed{seed}"
    out = os.path.join(cfg.out_dir, tag)
    os.makedirs(out, exist_ok=True)
    done = os.path.join(out, "metrics.json")
    if os.path.exists(done) and os.path.exists(os.path.join(out, "test_probs.npy")):
        report(f"{tag}: already finished, loading saved results")
        return json.load(open(done))

    ds, split = exp.dataset, exp.split
    classes = ds.classes
    report(f"{tag}: training")
    batch = cfg.batch_size
    while True:
        try:
            K.backend.clear_session()
            gc.collect()
            K.utils.set_random_seed(seed)
            model = build_model(run, cfg, len(classes), pretrained=not cfg.smoke_test)
            train_ds = make_dataset(ds.files[split.train], ds.labels[split.train], cfg.image_size, len(classes), batch, True, seed)
            val_ds = make_dataset(ds.files[split.val], ds.labels[split.val], cfg.image_size, len(classes), batch, False, seed)
            callbacks = [
                K.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.early_stopping_patience, restore_best_weights=True),
                K.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=cfg.min_lr),
                LearningRateLogger(),
                K.callbacks.CSVLogger(os.path.join(out, "history.csv")),
            ]
            t0 = time.time()
            hist = model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs, callbacks=callbacks, verbose=2)
            minutes = (time.time() - t0) / 60
            break
        except tf.errors.ResourceExhaustedError:
            if batch <= 8:
                raise
            batch //= 2
            report(f"   GPU out of memory -> retrying with batch size {batch}")

    h = {k: [float(v) for v in vs] for k, vs in hist.history.items()}
    model.save_weights(os.path.join(out, "model.weights.h5"))
    test_files, test_labels = ds.files[split.test], ds.labels[split.test]
    probs = model.predict(make_dataset(test_files, test_labels, cfg.image_size, len(classes), 32, False, seed), verbose=0)
    np.save(os.path.join(out, "test_probs.npy"), probs)

    result = metrics.evaluate(test_labels, probs, classes, cfg.bootstrap_resamples, seed)
    best = int(np.argmin(h["val_loss"])) + 1
    lr = h.get("learning_rate", [])
    result.update({
        "run": run, "seed": seed, "batch_size": batch, "epochs_run": len(h["loss"]), "best_val_loss_epoch": best,
        "val_acc_at_best_epoch": h["val_accuracy"][best - 1], "best_val_acc": max(h["val_accuracy"]),
        "lr_reductions": [(i + 1, lr[i]) for i in range(1, len(lr)) if not math.isclose(lr[i], lr[i - 1])],
        "train_minutes": minutes, "params": int(model.count_params()),
        "trainable_params": int(sum(np.prod(w.shape) for w in model.trainable_weights)),
    })
    try:
        attention = model.get_layer("attention")
        if attention.mode in ("ham", "ham_fixed"):
            result["alpha"] = float(np.array(attention.alpha))
            result["beta"] = float(np.array(attention.beta))
    except ValueError:
        pass
    try:
        result["latency_ms_batch1"] = measure_latency(model, cfg.image_size)
    except Exception as exc:
        report(f"[latency] skipped: {exc}")
    with open(done, "w") as f:
        json.dump(result, f, indent=1)

    try:
        plots.plot_history(h, os.path.join(out, f"fig_curves_{tag}"))
    except Exception as exc:
        report(f"[figures] skipped: {type(exc).__name__}: {exc}")

    lo, hi = result["accuracy_ci"]
    report(f"   {tag}: test acc {result['accuracy'] * 100:.2f}% [{lo * 100:.2f}, {hi * 100:.2f}], macro-F1 {result['macro_f1']:.4f}, "
           f"MCC {result['mcc']:.4f}, epochs {result['epochs_run']}, best epoch {best}, {minutes:.0f} min")
    del model, train_ds, val_ds, hist
    K.backend.clear_session()
    gc.collect()
    return result
