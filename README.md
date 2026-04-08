# SOC_DATA

This repository is reorganized around a single, direct pipeline built on aligned S11 CSV inputs.

## Data source

The training and evaluation pipeline now reads directly from:

`D:\SOC_DATA\data_complete\data_processing\S11_ALIGN_SOC`

Each CSV is treated as one aligned source file. Magnitude files are required. Phase files are paired automatically when `--include-phase` is enabled.

## Structure

- `src/data/`: data discovery and dataset building from aligned S11 CSV files
- `src/models/`: our Transformer regressor and baseline model factory
- `src/utils/`: shared seed, metrics, and IO helpers
- `src/train/`: training pipeline
- `src/validation/`: checkpoint evaluation pipeline
- `src/visualization/`: plots used by train/eval/baseline runs
- `output/`: new run outputs
- `legacy/`: archived old scripts and framework remnants

## Main entrypoints

- Train our model: `python run.py --include-phase`
- Evaluate checkpoint: `python eval_infer.py --checkpoint output/train/<run>/best.pt --include-phase`
- Train baseline: `python baseline_tabular.py --model xgb --include-phase`

## Notes

- Group-aware splitting is done at the source-file level.
- `DC` filtering is supported through `--dc-mode all|D|C`.
- Frequency windowing is supported through `--freq-min` and `--freq-max`.
- Old root-level experiment scripts are moved under `legacy/old_scripts`.
