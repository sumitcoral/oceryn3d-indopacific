#!/bin/bash
# ============================================================================
# NSCC environment setup - source before any CDO/python stage.
#   source scripts/env_nscc.sh
#
# CHANGED FROM v1: the conda block was guarded by `if ! command -v python`.
# On any node that already has a system python -- which is most of them -- that
# guard skipped conda activation entirely and the job then ran against the
# system interpreter with no torch and no xarray. The guard is gone.
# Still NO `module purge`: it would unload cdo loaded just above.
# ============================================================================
module load cdo    2>/dev/null || true
module load nco    2>/dev/null || true
module load netcdf 2>/dev/null || true

if ! command -v conda >/dev/null 2>&1; then
    module load miniforge3/25.3.1 2>/dev/null || true
fi
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate ocean_recon \
        || echo "[env_nscc] WARNING: conda env 'ocean_recon' not found."
else
    echo "[env_nscc] WARNING: conda not on PATH; python stages will fail."
fi

command -v cdo    >/dev/null 2>&1 || echo "[env_nscc] WARNING: cdo missing."
command -v ncdump >/dev/null 2>&1 || echo "[env_nscc] WARNING: ncdump missing."

echo "[env_nscc] cdo:    $(command -v cdo    || echo MISSING)"
echo "[env_nscc] ncdump: $(command -v ncdump || echo MISSING)"
echo "[env_nscc] python: $(command -v python || echo MISSING)"
echo "[env_nscc] conda env: ${CONDA_DEFAULT_ENV:-none}"
