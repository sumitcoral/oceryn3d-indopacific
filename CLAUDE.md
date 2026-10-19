# CLAUDE.md

Guidance for Claude Code sessions in this repository.

## What this is

**Oceryn3D Indo-Pacific**: a regional, daily-step AI emulator of the 3D ocean
on GLORYS12's 1/12° grid, 30°E–180°E × 30°S–30°N, 24 levels to 1,062 m,
forced by ERA5 wind stress, fine-tuned on 5-day rollouts. The sibling of
Oceryn3D Global (`sumitcoral/ocean-nowcast` = Model 1, `sumitcoral/ocean-emulator`
= Model 2). Start with `docs/STATUS.md`; the reasons behind the design are in
`docs/DESIGN.md`; `README.md` is the step-by-step guide.

## Ground rules

- **Never modify `sumitcoral/ocean-emulator` or `sumitcoral/ocean-nowcast`.**
  Read them (clone next to this repo) to check conventions; code shared with
  them was copied, not imported.
- **Never run downloads, training, inference or any pipeline stage on NSCC,
  and never `qsub`.** The user submits every PBS job and runs the downloads.
  When a change needs a run, name the command and stop.
- Workflow: edit → push → the user pulls on NSCC and runs it there.
- All data lives under `/data/projects/11003571/sumit/work/indopacific/` on NSCC.

## Layout and commands

- `config/default.yaml` is the single source of settings. Other configs start
  with `base: default.yaml` (merged recursively by `src/config.py`; lists
  replace). Keep `paths:` absolute (PBS jobs start in `$HOME`).
- `src/` holds flat scripts run as `python src/<name>.py`; siblings import each
  other directly. Pure-numpy helpers (`grid.py`, `config.py`) stay free of
  xarray/netCDF/torch so tests use them without data.
- `scripts/download_*.py` (internet, login node), `scripts/0N_*.pbs` (run
  order), `scripts/env_nscc.sh`.
- Tests: `python -m pytest -q` (synthetic data, CPU, ~15 s).

## Contracts and conventions (don't "fix" these)

- Channel contract C1–C4 (`config/default.yaml` header, asserted by
  `config.Cfg.validate`): the first `out_ch` (97) input channels are the
  targets at day t in target order, because the model is residual.
  `forcing_next` (taux, tauy at t+1) is appended last: in_ch 133.
- GLORYS files are matched to the config BY COORDINATE AND DEPTH VALUE, never
  by position (`glorys_io.GlorysMonth`, `grid.match_axis/match_depths`); a
  mismatch raises.
- Land, invalid values and levels below the sea floor are 0 after
  z-scoring in every channel, and forecasts are re-masked the same way after
  each rollout step.
- Ocean mask per level: k < `n_valid_levels`, the intersection of temp, u, v.
- The domain is regional: no longitude wrap anywhere; replicate padding in
  the U-Net; `dataset.edge_margin_px` cells at the open boundary are left out
  of the loss and of every score.
- Evaluation runs full-domain rollouts tile by tile (training-size tiles,
  blended), because GroupNorm statistics must match training.
- Splits by year: train 1993–2016, val 2017–2019, test 2020–2023. Stats use
  train years only. A sample is horizon+1 consecutive days of one split.
- Stage scripts are idempotent and write atomically (tmp name, then rename).
  Fail loudly with the cause and the fix; never add silent defaults that
  could change a channel count.
- Checkpoints embed the merged config (`ck["cfg"]`).
