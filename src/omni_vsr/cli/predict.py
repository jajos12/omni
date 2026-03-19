"""CLI for batch inference."""

from __future__ import annotations

import argparse

from omni_vsr.config import load_config
from omni_vsr.inference.predictor import run_inference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-csv", default=None)
    parser.add_argument("--output-csv", default="submission.csv")
    parser.add_argument("--device", default=None)
    parser.add_argument("--beam-width", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config, overrides=args.overrides)
    run_inference(
        config=config,
        checkpoint_path=args.checkpoint,
        output_csv=args.output_csv,
        sample_csv=args.sample_csv,
        device=args.device,
        beam_width=args.beam_width,
        alpha=args.alpha,
        beta=args.beta,
        batch_size=args.batch_size,
        use_tta=args.tta,
    )
    print(f"Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
