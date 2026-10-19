"""scan_days.py: each kind of problem planted in a synthetic series is found,
and the scan runs end to end on the synthetic raw files."""
import json
import os
from datetime import date, timedelta

import numpy as np

import scan_days as sd


def _days(n, d0=date(2000, 1, 1)):
    return [d0 + timedelta(days=k) for k in range(n)]


def test_flags_repeated_interpolated_and_jump():
    rng = np.random.default_rng(1)
    n = 200
    x = np.cumsum(rng.normal(size=(n, 50)), axis=0)          # a random walk field
    x[50] = x[49]                                            # repeated day
    x[100] = 0.5 * (x[99] + x[101])                          # interpolated day
    x[150:] += 40.0                                          # jump into day 150
    ch = np.array([np.nan] + [sd.rms(x[i] - x[i - 1]) for i in range(1, n)])
    sec = np.array([np.nan] + [sd.rms(x[i + 1] - 2 * x[i] + x[i - 1]) for i in range(1, n - 1)] + [np.nan])
    fl = sd.flag_days(_days(n), ch, sec)
    assert fl["repeated"] == [_days(n)[50]]
    assert _days(n)[100] in fl["interpolated"] and len(fl["interpolated"]) == 1
    assert fl["jump"] == [_days(n)[150]]


def test_weekly_rhythm_detected_only_when_present():
    rng = np.random.default_rng(2)
    n = 730
    base = 1.0 + 0.1 * rng.normal(size=n)
    with_cycle = base + 0.3 * (np.arange(n) % 7 == 3)
    assert sd.weekly_rhythm(with_cycle)["power_ratio"] > sd.WEEK_RATIO
    assert sd.weekly_rhythm(base)["power_ratio"] < sd.WEEK_RATIO
    m = sd.weekly_rhythm(with_cycle)["cycle_means"]
    assert int(np.argmax(m)) == 3


def test_switch_shift_against_other_years():
    dates = [d for d in _days(365 * 6, date(2016, 1, 1))]
    t = np.arange(len(dates))
    means = np.sin(2 * np.pi * t / 365.25) + 0.01 * np.random.default_rng(3).normal(size=len(dates))
    sw = date(2021, 7, 1)
    means_shift = means + np.array([0.5 if d >= sw else 0.0 for d in dates])
    r = sd.window_shift(dates, means_shift, sw)
    assert r["z"] > 10 and r["n_other_years"] >= 4
    assert abs(sd.window_shift(dates, means, sw)["z"]) < 4


def test_scan_end_to_end(world):
    assert sd.main(["--config", world["cfg"]]) == 0
    res = json.load(open(os.path.join(world["root"], "meta", "scan_days.json")))
    assert res["n_days"] == 32 and "u_0m" in res["fields"] and "ssh" in res["fields"]
    # The synthetic fields change smoothly every day: nothing repeated or interpolated.
    assert all(f["repeated"] == 0 and f["interpolated"] == 0 for f in res["fields"].values())
