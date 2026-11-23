"""Launcher: every model is trained in its own worker process.

The launcher never imports TensorFlow, so a crashed worker (out of memory, driver
error) cannot take the GPU or the notebook kernel down with it. Finished runs are
skipped on re-launch, and a worker that dies is restarted with a smaller batch.
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Callable

from .config import TrainConfig, VARIANTS


def _spawn(cfg_path: str, run: str, seed: int, batch: int, log_path: str, report: Callable) -> int:
    cmd = [sys.executable, "-m", "phamnet", "worker", "--config", cfg_path, "--run", run, "--seed", str(seed), "--batch-size", str(batch)]
    env = dict(os.environ, PYTHONUNBUFFERED="1", TF_CPP_MIN_LOG_LEVEL="2", TF_FORCE_GPU_ALLOW_GROWTH="true")
    with open(log_path, "w") as raw:
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        for line in proc.stdout:
            raw.write(line)
            raw.flush()
            if line.strip() and "cuda_timer" not in line and "absl" not in line:
                report(line.rstrip())
        proc.wait()
    return proc.returncode


def launch(cfg: TrainConfig, report: Callable = print) -> None:
    os.makedirs(os.path.join(cfg.out_dir, "raw_logs"), exist_ok=True)
    cfg_path = os.path.join(cfg.out_dir, "config.json")
    cfg.to_json(cfg_path)
    for run in cfg.runs:
        if run not in VARIANTS:
            report(f"Unknown run name: {run}")
            continue
        for seed in cfg.seeds:
            tag = f"{run}_seed{seed}"
            run_dir = os.path.join(cfg.out_dir, tag)
            if os.path.exists(os.path.join(run_dir, "metrics.json")) and os.path.exists(os.path.join(run_dir, "test_probs.npy")):
                report(f"{tag}: already finished, skipped")
                continue
            report(f"{tag}: training in a separate process")
            for batch in dict.fromkeys([cfg.batch_size, max(cfg.batch_size // 2, 4), max(cfg.batch_size // 4, 2)]):
                code = _spawn(cfg_path, run, seed, batch, os.path.join(cfg.out_dir, "raw_logs", f"{tag}_b{batch}.log"), report)
                if os.path.exists(os.path.join(run_dir, "metrics.json")):
                    break
                oom = " (killed by the system: out of memory)" if code in (-9, 137) else ""
                report(f"{tag}: worker stopped with exit code {code}{oom} at batch size {batch}")
