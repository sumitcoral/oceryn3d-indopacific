#!/usr/bin/env python
"""
Data QC of the raw GLORYS12 days, before the cache is built.

    python src/scan_days.py --config config/default.yaml [--levels 0,13] [--start 1993-01-01 --end 2023-12-31]

Reads a few levels (default: the surface and ~131 m) of u, v, temp, salt and
ssh for every day of the downloaded monthly files and looks for four things
that would make the model learn something other than ocean physics:

  1. repeated days      a day identical to the one before (change ~ 0)
  2. interpolated days  a day exactly halfway between its neighbours (second
                        difference ~ 0). Oceryn3D Global found ten of these in
                        SODA4 (scan_pentads.py, 14 Oct 2026).
  3. jumps              a day-to-day change far above the usual (robust z > 6
                        against the surrounding +-30 days)
  4. a 7-day rhythm     GLORYS12 assimilates observations in weekly cycles
                        (spread over the week, but not invisible). The day-to-
                        day change series is detrended and its power at a
                        7-day period compared with the 5-10 day band; the mean
                        change by day of the 7-day cycle is reported as well.
  5. dataset switches   at each download.glorys_datasets start date after the
                        first (the reanalysis -> interim switch, July 2021):
                        the change on that day (robust z) and the shift of the
                        30-day mean across it, compared with the same calendar
                        window in the other years.

Outputs next to days.csv: scan_days.csv (one row per day: change, second
difference and domain mean per field) and scan_days.json (summary and
flagged dates). Exit code 0 even when something is flagged: read the summary.
Flagged dates that are real errors go into data.skip_dates.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import date, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Cfg, parse_date                   # noqa: E402
from glorys_io import GlorysMonth, list_month_files  # noqa: E402

DUP_REL = 1e-4          # change below this x the median change -> repeated day
INTERP_REL = 0.02       # second difference below this x the change -> interpolated
JUMP_Z = 6.0            # robust z of the change against +-30 days -> jump
WEEK_RATIO = 8.0        # 7-day power over the 5-10 day band median -> rhythm


# ---------------------------------------------------------------------------
# Pure analysis (numpy; tested on synthetic series)
# ---------------------------------------------------------------------------
def rms(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    return float(np.sqrt(np.mean(a * a))) if a.size else float("nan")


def robust_z(x: np.ndarray, half: int = 30) -> np.ndarray:
    """z of each value against the median/MAD of its +-half neighbours."""
    x = np.asarray(x, float)
    z = np.full_like(x, np.nan)
    for i in range(len(x)):
        w = x[max(0, i - half):i + half + 1]
        w = w[np.isfinite(w)]
        if w.size < 5 or not np.isfinite(x[i]):
            continue
        med = np.median(w)
        mad = 1.4826 * np.median(np.abs(w - med))
        if mad > 0:
            z[i] = (x[i] - med) / mad
    return z


def running_median(x: np.ndarray, half: int) -> np.ndarray:
    out = np.empty_like(x)
    for i in range(len(x)):
        w = x[max(0, i - half):i + half + 1]
        out[i] = np.median(w[np.isfinite(w)]) if np.isfinite(w).any() else np.nan
    return out


def weekly_rhythm(change: np.ndarray, start_index: int = 0) -> dict:
    """Power of a 7-day period in a daily change series, relative to the
    median power over 5-10 day periods (excluding the 7-day bin and its
    neighbours), and the mean detrended change by day of the 7-day cycle."""
    x = np.asarray(change, float)
    ok = np.isfinite(x)
    if ok.sum() < 70:
        return {"power_ratio": None, "cycle_amplitude": None, "cycle_means": None}
    xi = np.interp(np.arange(len(x)), np.flatnonzero(ok), x[ok])
    d = xi - running_median(xi, 15)
    d = d - d.mean()
    p = np.abs(np.fft.rfft(d)) ** 2
    f = np.fft.rfftfreq(len(d), 1.0)
    k7 = int(np.argmin(np.abs(f - 1 / 7)))
    band = (f >= 1 / 10) & (f <= 1 / 5)
    band[max(0, k7 - 1):k7 + 2] = False
    ratio = float(p[k7] / np.median(p[band])) if band.any() else None
    phase = (np.arange(len(d)) + start_index) % 7
    means = [float(d[phase == k].mean()) for k in range(7)]
    sd = float(d.std())
    return {"power_ratio": ratio, "cycle_amplitude": (max(means) - min(means)) / sd if sd > 0 else None,
            "cycle_means": means}


def window_shift(dates, means: np.ndarray, switch: date, n: int = 30):
    """Shift of the n-day mean across `switch` that year, and its z against
    the same calendar window in the other years present."""
    idx = {d: i for i, d in enumerate(dates)}

    def shift(sw):
        a = [means[idx[sw - timedelta(days=k)]] for k in range(1, n + 1)
             if (sw - timedelta(days=k)) in idx]
        b = [means[idx[sw + timedelta(days=k)]] for k in range(0, n)
             if (sw + timedelta(days=k)) in idx]
        a, b = np.asarray(a, float), np.asarray(b, float)
        if a.size < n // 2 or b.size < n // 2:
            return None
        return float(np.nanmean(b) - np.nanmean(a))

    s0 = shift(switch)
    years = sorted({d.year for d in dates} - {switch.year})
    others = []
    for y in years:
        try:
            v = shift(switch.replace(year=y))
        except ValueError:                       # 29 Feb
            v = None
        if v is not None:
            others.append(v)
    if s0 is None:
        return {"shift": None, "z": None, "n_other_years": len(others)}
    if len(others) < 3:
        return {"shift": s0, "z": None, "n_other_years": len(others)}
    o = np.asarray(others)
    sd = o.std()
    return {"shift": s0, "z": float((s0 - o.mean()) / sd) if sd > 0 else None,
            "n_other_years": len(others)}


def flag_days(dates, change: np.ndarray, second: np.ndarray):
    """{kind: [dates]} for repeated, interpolated and jump days of one series."""
    med = np.nanmedian(change)
    out = {"repeated": [], "interpolated": [], "jump": []}
    z = robust_z(change)
    for i, d in enumerate(dates):
        c = change[i]
        if not np.isfinite(c):
            continue
        if c < DUP_REL * med:
            out["repeated"].append(d)
        if np.isfinite(second[i]) and c > DUP_REL * med and second[i] < INTERP_REL * c:
            out["interpolated"].append(d)
        if np.isfinite(z[i]) and z[i] > JUMP_Z:
            out["jump"].append(d)
    return out


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def series_names(cfg: Cfg, levels):
    names = []
    for v in cfg.vars3d:
        names += [f"{v}_{cfg.depth_m[k]:.0f}m" for k in levels]
    names += [v for v in cfg.targets_2d]
    return names


def read_fields(gm: GlorysMonth, cfg: Cfg, day: date, levels) -> dict:
    f = gm.read_day(day, levels=levels)
    out = {}
    for v in cfg.vars3d:
        for j, k in enumerate(levels):
            out[f"{v}_{cfg.depth_m[k]:.0f}m"] = f[v][j]
    for v in cfg.targets_2d:
        out[v] = f[v]
    return out


def scan(cfg: Cfg, levels, start=None, end=None):
    files = list_month_files(cfg.paths["glorys_dir"])
    if not files:
        raise FileNotFoundError(f"no GLORYS files under {cfg.paths['glorys_dir']}")
    names = series_names(cfg, levels)
    dates, ch, mean = [], {n: [] for n in names}, {n: [] for n in names}
    sec = {n: [] for n in names}
    prev2 = prev = None
    prev_day = None
    t0 = time.time()
    for y, m, p in files:
        with GlorysMonth(p, cfg) as gm:
            for d in gm.dates:
                if (start and d < start) or (end and d > end):
                    continue
                f = read_fields(gm, cfg, d, levels)
                consecutive = prev_day is not None and (d - prev_day).days == 1
                dates.append(d)
                for n in names:
                    a = f[n]
                    mean[n].append(float(np.nanmean(a)))
                    ch[n].append(rms(a - prev[n]) if consecutive else np.nan)
                    sec[n].append(np.nan)
                    # second difference centred on the PREVIOUS day
                    if consecutive and prev2 is not None and len(dates) >= 3 \
                            and (prev_day - dates[-3]).days == 1:
                        sec[n][-2] = rms(a - 2 * prev[n] + prev2[n])
                prev2, prev, prev_day = prev, f, d
        print(f"  {y}-{m:02d}: {len(dates)} days, {time.time() - t0:.0f} s", flush=True)
    return dates, names, {n: np.array(v) for n, v in ch.items()}, \
        {n: np.array(v) for n, v in sec.items()}, {n: np.array(v) for n, v in mean.items()}


def summarize(cfg: Cfg, dates, names, ch, sec, mean) -> dict:
    out = {"n_days": len(dates), "first": dates[0].isoformat(), "last": dates[-1].isoformat(),
           "thresholds": {"repeated_rel": DUP_REL, "interpolated_rel": INTERP_REL,
                          "jump_z": JUMP_Z, "week_power_ratio": WEEK_RATIO},
           "fields": {}, "flagged_dates": {}, "switches": {}}
    all_flags = {}
    for n in names:
        fl = flag_days(dates, ch[n], sec[n])
        wk = weekly_rhythm(ch[n])
        out["fields"][n] = {"median_change": float(np.nanmedian(ch[n])),
                            "weekly": wk,
                            "weekly_flag": bool(wk["power_ratio"] and wk["power_ratio"] > WEEK_RATIO),
                            **{k: len(v) for k, v in fl.items()}}
        for k, v in fl.items():
            for d in v:
                all_flags.setdefault(d.isoformat(), set()).add(f"{k}:{n}")
    out["flagged_dates"] = {d: sorted(s) for d, s in sorted(all_flags.items())}
    for e in cfg.download["glorys_datasets"][1:]:
        sw = parse_date(e["start"])
        if sw not in dates:
            continue
        i = dates.index(sw)
        rep = {}
        for n in names:
            z = robust_z(ch[n])
            rep[n] = {"change_z": None if not np.isfinite(z[i]) else float(z[i]),
                      **window_shift(dates, mean[n], sw)}
        out["switches"][e["id"]] = {"date": sw.isoformat(), "fields": rep}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--levels", default="", help="indices into grid.depth_m (default: 0 and ~130 m)")
    ap.add_argument("--start")
    ap.add_argument("--end")
    args = ap.parse_args(argv)
    cfg = Cfg.load(args.config)
    if args.levels:
        levels = [int(x) for x in args.levels.split(",")]
    else:
        levels = sorted({0, int(np.argmin(np.abs(np.asarray(cfg.depth_m) - 130.0)))})
    dates, names, ch, sec, mean = scan(cfg, levels, parse_date(args.start) if args.start else None,
                                       parse_date(args.end) if args.end else None)
    res = summarize(cfg, dates, names, ch, sec, mean)
    meta = cfg.paths["meta_dir"]
    os.makedirs(meta, exist_ok=True)
    with open(os.path.join(meta, "scan_days.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date"] + [f"{n}_{k}" for n in names for k in ("change", "second", "mean")])
        for i, d in enumerate(dates):
            w.writerow([d.isoformat()] + [f"{a[n][i]:.6g}" for n in names for a in (ch, sec, mean)])
    with open(os.path.join(meta, "scan_days.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"[scan] {res['n_days']} days {res['first']}..{res['last']}")
    for n, s in res["fields"].items():
        wk = s["weekly"]["power_ratio"]
        print(f"  {n:14s} repeated {s['repeated']:3d}  interpolated {s['interpolated']:3d}  "
              f"jumps {s['jump']:3d}  7-day power x{wk:.1f}" if wk else
              f"  {n:14s} repeated {s['repeated']:3d}  interpolated {s['interpolated']:3d}  "
              f"jumps {s['jump']:3d}  7-day power n/a")
    for did, s in res["switches"].items():
        worst = max((abs(v["z"]) for v in s["fields"].values() if v["z"] is not None), default=None)
        print(f"  switch to {did} on {s['date']}: largest 30-day-mean shift z "
              f"{worst if worst is None else round(worst, 1)} vs other years")
    print(f"  {len(res['flagged_dates'])} flagged dates -> {meta}/scan_days.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
