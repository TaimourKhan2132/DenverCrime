"""Command-line entry point: `python -m denvercrime <command>`."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from denvercrime.config import DEFAULT_CONFIG, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="denvercrime", description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="TOML config (default: %(default)s)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="download the latest snapshot from Denver's FeatureServer")
    p = sub.add_parser("prepare", help="raw snapshot -> incidents -> panel -> features")
    p.add_argument("--raw", type=Path, help="raw .parquet or Hub .csv (default: newest in data/raw)")
    sub.add_parser("backtest", help="train on train+val, score model and baselines on the test period")
    p = sub.add_parser("maps", help="maps and plots for a backtest run")
    p.add_argument("--run", type=Path, help="run directory (default: latest)")
    p.add_argument("--week", help="Monday of the test week to map, YYYY-MM-DD (default: last)")
    p = sub.add_parser("all", help="prepare + backtest + maps (does not download)")
    p.add_argument("--raw", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = load_config(args.config)

    from denvercrime import pipeline  # imported late so `--help` stays fast

    if args.command == "download":
        from denvercrime.data.download import download

        download(cfg)
    elif args.command == "prepare":
        pipeline.prepare(cfg, args.raw)
    elif args.command == "backtest":
        pipeline.backtest(cfg, args.config)
    elif args.command == "maps":
        pipeline.make_maps(cfg, args.run, args.week)
    elif args.command == "all":
        pipeline.prepare(cfg, args.raw)
        run_dir = pipeline.backtest(cfg, args.config)
        pipeline.make_maps(cfg, run_dir)
    return 0
