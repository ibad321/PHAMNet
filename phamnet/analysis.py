"""Summary of finished runs: tables, paired tests, confusion matrices and model complexity."""
from __future__ import annotations

import glob
import json
import os
from typing import Callable, Dict

import numpy as np

from . import metrics, plots
from .config import TrainConfig
from .experiment import Experiment


def load_results(out_dir: str) -> Dict[str, dict]:
    results = {}
    for path in sorted(glob.glob(os.path.join(out_dir, "*_seed*", "metrics.json"))):
        results[os.path.basename(os.path.dirname(path))] = json.load(open(path))
    return results


def _mean_sd(values, scale=1.0) -> str:
    v = np.asarray(values) * scale
    return f"{v.mean():.2f} +- {v.std(ddof=1):.2f}" if len(v) > 1 else f"{v.mean():.2f}"


def summarize(cfg: TrainConfig, exp: Experiment, report: Callable = print, with_complexity: bool = True) -> Dict[str, dict]:
    results = load_results(cfg.out_dir)
    classes = exp.dataset.classes
    if not results:
        report("No finished runs found.")
        return results
    tags = list(results)
    report(f'{"run":<22}{"acc %":>8}{"95% CI":>18}{"macroP":>8}{"macroR":>8}{"macroF1":>9}{"MCC":>8}{"AUROC":>8}{"ECE":>7}{"ep":>5}{"best":>6}')
    for t in tags:
        r = results[t]
        report(f'{t:<22}{r["accuracy"] * 100:>8.2f}   [{r["accuracy_ci"][0] * 100:5.2f},{r["accuracy_ci"][1] * 100:6.2f}]'
               f'{r["macro_precision"]:>8.4f}{r["macro_recall"]:>8.4f}{r["macro_f1"]:>9.4f}{r["mcc"]:>8.4f}'
               f'{r["macro_auroc"]:>8.4f}{r["ece_15bins"]:>7.3f}{r["epochs_run"]:>5}{r["best_val_loss_epoch"]:>6}')
    report("\nPer-class (sensitivity / specificity / PPV / NPV / F1 / AUROC):")
    for t in tags:
        for c, v in results[t]["per_class"].items():
            report(f"  {t:<22}{c:<10}" + " ".join(f"{v[k]:.4f}" for k in ("sensitivity", "specificity", "ppv", "npv", "f1", "auroc")))
    report(f"\nConfusion matrices (rows = true, columns = predicted, order {classes}):")
    for t in tags:
        report(f"  {t}: {results[t]['confusion_matrix']}")

    y_test = exp.dataset.labels[exp.split.test]
    probs = {t: np.load(os.path.join(cfg.out_dir, t, "test_probs.npy")) for t in tags}
    report("\nExact McNemar test, PHAMNet against every other run with the same seed (paired, same test images):")
    for t in tags:
        r = results[t]
        ref = f"phamnet_seed{r['seed']}"
        if r["run"] == "phamnet" or ref not in probs:
            continue
        only_a, only_b, p = metrics.mcnemar_exact(probs[ref].argmax(1) == y_test, probs[t].argmax(1) == y_test)
        report(f"  phamnet vs {r['run']:<16} seed {r['seed']:<5}: only PHAMNet correct {only_a:>3}, only {r['run']} correct {only_b:>3}, p = {p:.4f}")

    report("\nComparison table (mean +- SD over seeds where n > 1; macro-F1, MCC and class F1 shown x100):")
    report(f'{"run":<18}{"n":>3}{"accuracy %":>18}{"macro-F1":>18}{"MCC":>18}')
    for run in dict.fromkeys(results[t]["run"] for t in tags):
        v = [results[t] for t in tags if results[t]["run"] == run]
        report(f'{run:<18}{len(v):>3}{_mean_sd([x["accuracy"] for x in v], 100):>18}'
               f'{_mean_sd([x["macro_f1"] for x in v], 100):>18}{_mean_sd([x["mcc"] for x in v], 100):>18}')

    pair = [t for t in tags if t.startswith("b4_tl_")][:1] + [t for t in tags if t.startswith("phamnet_")][:1]
    if len(pair) == 2:
        plots.plot_confusion_pair([np.array(results[t]["confusion_matrix"]) for t in pair],
                                  ["(a) EfficientNet-B4 + TL (baseline)", "(b) PHAMNet (proposed)"], classes,
                                  os.path.join(cfg.out_dir, "fig_confusion_matrices"))
    if with_complexity:
        try:
            report("\nComplexity at %dx%d (FLOPs count multiply and add separately):" % (cfg.image_size, cfg.image_size))
            for run, params, gflops in complexity(sorted({results[t]["run"] for t in tags}), cfg, len(classes)):
                report(f"  {run:<16} params {params:>12,}  FLOPs {gflops:.3f} G")
        except Exception as exc:
            report(f"[complexity] skipped: {type(exc).__name__}: {exc}")
    with open(os.path.join(cfg.out_dir, "all_results.json"), "w") as f:
        json.dump({"classes": classes, "split": exp.split_rows, "results": results}, f, indent=1)
    return results


def complexity(runs, cfg: TrainConfig, num_classes: int):
    """Parameters and forward-pass FLOPs (batch size 1) of each variant."""
    import tensorflow as tf
    from tensorflow import keras as K
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    from .model import build_model

    rows = []
    for run in runs:
        K.backend.clear_session()
        model = build_model(run, cfg, num_classes, pretrained=False)
        fn = tf.function(lambda x: model(x)).get_concrete_function(tf.TensorSpec([1, cfg.image_size, cfg.image_size, 3]))
        graph = convert_variables_to_constants_v2(fn).graph
        options = tf.compat.v1.profiler.ProfileOptionBuilder(tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()).with_empty_output().build()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=tf.compat.v1.RunMetadata(), cmd="op", options=options).total_float_ops
        rows.append((run, model.count_params(), flops / 1e9))
        del model
    K.backend.clear_session()
    return rows
