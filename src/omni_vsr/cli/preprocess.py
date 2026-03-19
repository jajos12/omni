"""CLI for ROI preprocessing."""

from __future__ import annotations

import argparse

from omni_vsr.config import load_config
from omni_vsr.preprocessing import preprocess_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--no-skip", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config, overrides=args.overrides)
    splits = args.splits or [config.data.train_split, config.data.test_split]
    for split in splits:
        stats = preprocess_split(
            data_root=config.resolve(config.project.data_dir),
            output_root=config.resolve(config.project.roi_dir),
            split=split,
            target_size=config.data.target_size,
            skip_existing=not args.no_skip,
        )
        print(f"{split}: success={stats.success} failed={stats.failed} skipped={stats.skipped}")


if __name__ == "__main__":
    main()
