# Oceryn3D Indo-Pacific

A regional AI ocean emulator for the Indo-Pacific: currents, temperature and
salinity on 24 levels from the surface to 1,062 m, plus sea level, on
GLORYS12's 1/12° grid over **30°E–180°E × 30°S–30°N** (720 × 1800 cells).
It steps the ocean **one day at a time** and is fine-tuned on 5-day rollouts,
so it forecasts each of the next five days. It is trained on the GLORYS12V1
daily reanalysis (Mercator, 1993–) and forced by ERA5 daily-mean wind stress.

It is the regional sibling of **Oceryn3D Global**
([ocean-nowcast](https://github.com/sumitcoral/ocean-nowcast) +
[ocean-emulator](https://github.com/sumitcoral/ocean-emulator)): the same
residual U-Net, channel contract, normalization and fail-loudly conventions,
on a different dataset, grid and time step. Design choices and their reasons
are in [docs/DESIGN.md](docs/DESIGN.md); what has run and what to run next is
in [docs/STATUS.md](docs/STATUS.md). Roadmap context: *Ocean Weather Service
Roadmap*, Phase 1 steps 1–2 (tides and WaveWatch III are later modules).

## Pipeline

All settings live in `config/default.yaml`. `config/proto_quarter.yaml` (the
1/4° prototype, 240 × 600) and `config/rollout5.yaml` (5-day fine-tuning)
start from it with `base: default.yaml` and override only what differs.
Every stage takes `--config`.

| Stage | Command | Runs where | Output (under `paths.root`) |
|---|---|---|---|
| 0a download GLORYS12 | `python scripts/download_glorys.py --config config/default.yaml --static` | login node (internet), `nohup`, resumable | `GLORYS12/YYYY/glorys12_YYYY_MM.nc` (int16), `glorys12_static.nc` |
| 0b download ERA5 | `python scripts/download_era5.py --config config/default.yaml` | login node (internet), resumable | `ERA5/YYYY/era5_stress_YYYY_MM.nc` |
| 1 forcing | `qsub scripts/01_build_forcing.pbs` | PBS CPU | `forcing/YYYY/forcing_YYYY_MM.npy` (+ `.json`) |
| 2 day index | `python src/build_index.py --config …` | login node, seconds | `meta/days.csv` |
| 3 static fields | `python src/build_static.py --config …` | login node, a minute | `meta/static_fields.npz` |
| 4 stats | `qsub scripts/04_compute_stats.pbs` | PBS CPU | `meta/stats.json` |
| 5 cache | `qsub scripts/05_build_cache_array.pbs` | PBS array, one task per year | `cache/YYYY/day_YYYY_MM_DD.npy` (fp16) |
| 6a train, 1 day | `qsub -l walltime=02:00:00 -v SMOKE=1 scripts/06_train.pbs`, then `qsub scripts/06_train.pbs` | PBS GPU | `runs/ip_daily_v1/{last,best}.pt`, `metrics.csv` |
| 6b rollout to day 5 | `qsub -v CONFIG=config/rollout5.yaml scripts/06_train.pbs` | PBS GPU | `runs/ip_rollout5_v1/` |
| 7 evaluate | `qsub -v CKPT=<run>/best.pt scripts/07_evaluate.pbs` (`NEXT=persist` for no wind forecast) | PBS GPU | `eval_<split>_<ckpt>.json`, `_levels.csv` |

Run from the repo root on NSCC, after `source scripts/env_nscc.sh` (it
activates the conda env `ocean_recon`, the same as Oceryn3D Global's). The
downloads also need `pip install copernicusmarine cdsapi` in that env, a
one-time `copernicusmarine login`, and a CDS key in `~/.cdsapirc`. They can
also run on the Ubuntu workstation, writing to the sshfs mount with
`--remap /data/projects/11003571/sumit/work=/home/sumit/work/model/test_model`.

**Do the 1/4° prototype first**: the same stages with
`--config config/proto_quarter.yaml` (it reads the same downloads; its own
forcing, meta, cache and runs are 9× smaller) to debug masks, normalization
and wind inputs cheaply before the 1/12° runs.

### Sizes and times (estimates)

| Item | 1/12° | 1/4° |
|---|---|---|
| GLORYS12 raw, 1993–2023 (int16, compressed) | ~1–2 TB | (shared) |
| ERA5 daily stress | ~15 GB | (shared) |
| fp16 cache, 99 × h × w per day | 257 MB/day, ~2.9 TB | 29 MB/day, ~0.3 TB |
| patches per day (256 px, stride 224) | 32 before dropping mostly-land ones | 32 (128 px) |

## What the model sees

Input, 133 channels (contract C2): u, v, temp, salt × 24 levels and ssh at
day t (the state, 97); taux, tauy at day t; 24 depth-encoding channels; land
mask, bathymetry, sin/cos lat, sin/cos lon, sin/cos day of year; taux, tauy
at day t+1. Output, 97 channels: the state at day t+1. The model is
residual, so it predicts the one-day change. For day 2–5 its own forecast is
fed back (rollout); the forcing of those days comes from a wind forecast in
operation, which `evaluate.py --next-forcing persist|blend:A` brackets.

## Tests

Synthetic GLORYS12/ERA5 files on a 64 × 48 piece of the real grid run every
stage, training (1-day and rollout) and evaluation on CPU in ~15 s:

```bash
pip install pytest   # once; needs numpy, xarray, netCDF4, pyyaml, torch
python -m pytest -q
python src/dataset.py --config config/default.yaml   # channel counts, origins, RAM (no data)
python src/model.py                                  # U-Net shape check
```

A zero-change model scores exactly persistence (skill 1.000) in the
evaluation test, and tiled full-domain inference equals direct inference for
a pointwise model, so the scoring and tiling arithmetic are checked.
