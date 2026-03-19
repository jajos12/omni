# Omni VSR

Omni VSR is a modular visual speech recognition codebase for the OmniSub2026 competition. It turns the guide in [OmniSub2026_VSR_Guide-1.md](C:\Users\LOQ\source\repos\projects\omni\OmniSub2026_VSR_Guide-1.md) into a real project with:

- a conda-first environment
- a package-based `src/omni_vsr` layout
- CLI entry points for preprocessing, training, inference, and Kaggle submission
- cleaner separation between data, models, training, and decoding

## Quick Start

```powershell
conda env create -f environment.yml
conda activate omni-vsr
pip install -e .
```

Run the main commands as:

```powershell
omni-preprocess --config configs/default.yaml
omni-train --config configs/default.yaml
omni-predict --config configs/default.yaml --checkpoint checkpoints/finetuned/best.pt
```

## Layout

```text
configs/
  default.yaml
scripts/
src/
  omni_vsr/
    cli/
    data/
    inference/
    models/
    preprocessing/
    training/
    utils/
```

## Notes

- The guide assumes external assets such as the pretrained checkpoint, competition data, sample submission, and KenLM binaries.
- `omni-submit-kaggle` is included so we can wire up Kaggle submissions as soon as you share the token and the competition slug.
