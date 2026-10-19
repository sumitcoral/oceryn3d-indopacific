# Design decisions

Each choice below has its reason, so later changes can be weighed against it.
The roadmap context is the *Ocean Weather Service Roadmap* (Phase 1: build
the ocean engine; Gate 1: better currents than Copernicus Marine in the
Singapore and Malacca straits at 1–5 days, a forecast in minutes).

## Data

- **GLORYS12V1 daily means** (`cmems_mod_glo_phy_my_0.083deg_P1D-m`, then the
  interim continuation `..._myint_...` from July 2021). Daily, 1/12°, 50
  levels, altimetry-constrained, and free for commercial use with
  attribution, unlike SODA4.15.2, whose commercial terms are unknown. The
  Oceryn3D Global team decided against GLORYS12 for the global model on
  11 Oct 2026 (5–6× SODA4's data, no wind stress in the files, a daily
  step); a regional domain makes the volume manageable, and the daily step
  is what this product needs.
- **ERA5 daily-mean turbulent surface stress** (taux, tauy) as forcing, from
  the CDS daily-statistics dataset (small downloads, ~15 GB in all). ERA5 is
  the atmosphere GLORYS12 was forced with. The CDS daily mean of an hourly
  mean-rate field covers 23:00 the day before to 23:00, a 1 h offset from
  GLORYS12's day, negligible for a daily mean.
- **24 levels, 0.49–1062 m**, picked by depth: every other level in the top
  40 m, all levels from 40 to 541 m, then 763 and 1062 m. 24 matches
  Oceryn3D Global's channel count and keeps the cache at ~257 MB/day.
- Raw files are stored as **int16** with fixed scale factors (1 mm/s,
  0.001 °C, 0.001 psu, 1 mm): half of float32 and far below model error.

- **No nowcast model**: the starting state is Mercator's analysis (GLORYS12
  in training, GLO12 in operation), which already assimilates satellite and
  in-situ observations. Oceryn3D Global needs its own nowcast because SODA4
  is not produced in real time. The cost is dependence on Mercator, and a
  shift between GLORYS12 (training) and GLO12 (operation) that must be
  measured before the daily chain is trusted. A regional nowcast (satellite
  surface fields to 3D, like ocean-nowcast) is an option for later.
- **Data QC before training** (`src/scan_days.py`): repeated, interpolated
  and jump days, the weekly assimilation rhythm in the daily changes, and
  the shift at the reanalysis -> interim switch in July 2021, which falls in
  the test years.

## Grid and domain

- **No horizontal regridding of the ocean fields**: the model grid is
  GLORYS12's own, cut to the domain; 30°E and 30°S are exact grid points.
  The domain stops at 179.92°E so nothing crosses the date line.
- **Open boundaries**: the edges have no information about what flows in, so
  errors grow inward with lead time. The loss and the scores ignore a 2°
  margin (`edge_margin_px`). Feeding boundary values from an outer forecast
  (Copernicus GLO12, or Oceryn3D Global) is the planned fix; the input
  layout can take extra channels without changing the rest.
- **1/4° prototype** (`proto_quarter.yaml`, roadmap step 1) by 3×3
  block-averaging at read time: a coarse cell is ocean when at least half its
  fine cells are, so coastlines do not flip on one cell.

## Model and training

- **The Oceryn3D Global U-Net unchanged** (residual, GroupNorm, replicate
  padding, base 64, 13.4 M parameters), the same channel contract plus the
  forcing at t+1. Reusing it isolates what the new data and step change.
- **Daily step, then rollout fine-tuning to day 5** (`rollout5.yaml`): the
  one-day model is trained first (cheap, stable), then fine-tuned on 5-step
  autoregressive rollouts with the loss on every day, so it learns to cope
  with its own errors. Each forecast is re-masked before it is fed back.
- **Forcing at t+1 is an input** (unlike Oceryn3D Global's default): a day's
  currents respond to that day's wind. In operation it is a wind forecast.
  Oceryn3D Global's J3 test (Oct 2026) showed that a model trained on perfect
  future stress can lose to a no-forecast model when fed a real forecast,
  partly through sub-degree stress structure a forecast cannot give.
  Therefore: every evaluation also runs with `--next-forcing persist` (no
  forecast) and `blend:A`; `forcing_next_smooth_deg` box-smooths the t+1
  forcing in training and evaluation; and the real-forecast test (archived
  ECMWF forecasts) should come early, not at the pilot.
- **Loss** as Oceryn3D Global's data term: per-variable weights (currents 1,
  temp/salt 0.25, ssh 1), surface-heavy level weights, per-level ocean mask,
  cos(lat) area weight, edge margin.

## Evaluation

- **Full-domain rollouts, tiled**: each step runs on overlapping
  training-size tiles, blended, so GroupNorm sees what it saw in training and
  information crosses tile borders between steps. Validation during training
  is patch-based (cheaper) and only tracks progress.
- **Baselines**: persistence now. The Gate 1 comparison needs Copernicus
  GLO12 forecasts and observations (tide gauges, ADCP, drifters) in the
  straits; the region `malacca_singapore` is already scored separately.

## Not in this build (later modules)

- Tide-plus-residual layer (harmonic tides from Delft3D-FM; GLORYS12 daily
  means contain no tides).
- Nested WaveWatch III coupled to the emulator currents.
- The daily operational pipeline (GLO12 analysis + ECMWF forecast winds),
  API and viewer (Phase 2).
