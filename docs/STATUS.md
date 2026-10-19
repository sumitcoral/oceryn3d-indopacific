# Status

_Last updated: 19 Oct 2026._

## Built

The full Phase 1 core (roadmap steps 1–2): download scripts, Stages 1–7, the
1/4° prototype and 5-day rollout configs, PBS jobs and tests. Nothing has run
on real data yet. The test suite (`python -m pytest -q`, 26 tests) runs every
stage, training and evaluation on synthetic GLORYS12/ERA5 files and passes.

## Next steps (all on NSCC, submitted by the user)

1. **Environment**: in `ocean_recon`, `pip install copernicusmarine cdsapi`;
   `copernicusmarine login`; CDS key in `~/.cdsapirc`.
2. **Test downloads, one month each**, and check the levels and grid match:
   ```bash
   python scripts/download_glorys.py --config config/default.yaml --start 1993-01 --end 1993-01 --static
   python scripts/download_era5.py   --config config/default.yaml --start 1993-01 --end 1993-01
   ```
   `download_glorys.py` refuses a file whose depths or grid do not match the
   config (by value, within 1%), so a wrong `grid.depth_m` shows up here.
   If the CDS rejects the stress variable names, the error names the field;
   tell Claude what the CDS form calls it.
3. **Full downloads** with `nohup` (resumable; rerun to fill failures).
   Check disk first: ~1–2 TB of GLORYS12 + ~2.9 TB of 1/12° cache.
4. **1/4° prototype end to end**: Stages 1–7 with
   `--config config/proto_quarter.yaml` (`-v CONFIG=...` for PBS jobs).
5. **1/12° runs**: Stages 1–5 with `default.yaml`, then 6a, 6b and 7, with
   `NEXT=persist` as well as the default.

## Open questions

- Dataset IDs and end dates of GLORYS12 (`download.glorys_datasets`):
  check against the Copernicus Marine catalogue at the first download.
- Disk budget for raw GLORYS12 + cache (delete raw years after their cache is
  verified, if space is short; raw is only needed to rebuild after a stats change).
- Boundary forcing from an outer model, and the real-forecast (ECMWF) test of
  the t+1 wind input: see DESIGN.md.
