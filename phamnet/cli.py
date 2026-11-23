"""Command-line interface: ``python -m phamnet {train,worker,summarize,audit}``."""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from .config import PRESETS, VARIANTS, TrainConfig
from .utils import Reporter


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", help="folder with the four class sub-folders (auto-detected under /kaggle/input if omitted)")
    p.add_argument("--out-dir", default="phamnet_results")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float, help="learning rate")
    p.add_argument("--duplicate-aware-split", action="store_true", help="keep near-duplicate groups inside one subset")
    p.add_argument("--smoke", action="store_true", help="tiny CPU-friendly run without pretrained weights to check the installation")


def _config(args, runs=None, seeds=None) -> TrainConfig:
    cfg = TrainConfig(data_dir=args.data_dir, out_dir=args.out_dir, duplicate_aware_split=args.duplicate_aware_split, smoke_test=args.smoke)
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.learning_rate = args.lr
    if args.smoke:
        cfg.epochs, cfg.batch_size, cfg.lr_patience, cfg.bootstrap_resamples = 2, 8, 1, 200
        if args.epochs:
            cfg.epochs = args.epochs
        if args.batch_size:
            cfg.batch_size = args.batch_size
    if runs is not None:
        cfg.runs = runs
    if seeds is not None:
        cfg.seeds = seeds
    return cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phamnet", description="PHAMNet: EfficientNet-B4 + MFFS + HAM for fundus disease recognition")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train and evaluate models (one worker process per model and seed)")
    _common(train)
    train.add_argument("--preset", choices=sorted(PRESETS), help="named set of models and seeds")
    train.add_argument("--runs", nargs="+", choices=sorted(VARIANTS), help="model variants to train")
    train.add_argument("--seeds", nargs="+", type=int)

    worker = sub.add_parser("worker", help="train one model variant with one seed in this process")
    worker.add_argument("--config", required=True)
    worker.add_argument("--run", required=True, choices=sorted(VARIANTS))
    worker.add_argument("--seed", required=True, type=int)
    worker.add_argument("--batch-size", type=int)

    summ = sub.add_parser("summarize", help="tables, paired tests and figures for finished runs")
    _common(summ)
    summ.add_argument("--no-complexity", action="store_true", help="skip the parameter / FLOP count")

    audit = sub.add_parser("audit", help="near-duplicate audit and split statistics only (no TensorFlow)")
    _common(audit)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "worker":
        cfg = TrainConfig.from_json(args.config)
        if args.batch_size:
            cfg.batch_size = args.batch_size
        from .experiment import prepare
        from .trainer import run_one

        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        exp = prepare(cfg, report=lambda *_: None)
        run_one(args.run, args.seed, cfg, exp)
        return 0

    reporter = Reporter()
    if args.command == "train":
        runs, seeds = None, args.seeds
        if args.preset:
            runs = PRESETS[args.preset][0]
            seeds = seeds or PRESETS[args.preset][1]
        runs = args.runs or runs
        if not runs:
            print("Choose --preset or --runs.", file=sys.stderr)
            return 2
        cfg = _config(args, runs, seeds)
        from .experiment import prepare
        from .runner import launch

        reporter.header("Dataset and split")
        prepare(cfg, report=reporter, make_figures=True)
        reporter.header("Training")
        launch(cfg, report=reporter)
        reporter.header("Summary")
        _summarize(cfg, reporter, True)
    elif args.command == "summarize":
        cfg = _config(args)
        _summarize(cfg, reporter, not args.no_complexity)
    elif args.command == "audit":
        cfg = _config(args)
        from .experiment import prepare

        prepare(cfg, report=reporter, make_figures=True)
    os.makedirs(args.out_dir, exist_ok=True)
    reporter.write(os.path.join(args.out_dir, f"REPORT_{args.command}.txt"))
    return 0


def _summarize(cfg: TrainConfig, reporter, complexity: bool) -> None:
    from .analysis import summarize
    from .experiment import prepare

    exp = prepare(cfg, report=lambda *_: None)
    summarize(cfg, exp, reporter, with_complexity=complexity)


if __name__ == "__main__":
    sys.exit(main())
