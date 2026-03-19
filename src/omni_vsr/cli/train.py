"""CLI for model training."""

from __future__ import annotations

import argparse

from omni_vsr.config import load_config
from omni_vsr.training.trainer import train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config, overrides=args.overrides)
    best_checkpoint = train_model(config)
    if best_checkpoint is not None:
        print(f"Best checkpoint: {best_checkpoint}")


if __name__ == "__main__":
    main()
