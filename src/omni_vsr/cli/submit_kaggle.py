"""CLI for Kaggle competition submissions."""

from __future__ import annotations

import argparse

from omni_vsr.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--competition", default=None)
    parser.add_argument("--file", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config, overrides=args.overrides)
    competition = args.competition or config.kaggle.competition
    if not competition:
        raise ValueError("Competition slug is required. Set kaggle.competition in config or pass --competition.")

    from kaggle.api.kaggle_api_extended import KaggleApi  # Imported lazily.

    api = KaggleApi()
    api.authenticate()
    response = api.competition_submit(file_name=args.file, message=args.message, competition=competition)
    print(response)


if __name__ == "__main__":
    main()
