#!/usr/bin/env python3
"""Pre-registered analysis of the p2 suite: does the agent widen against informed flow?

The pre-registration is ``notes/p2_preregistration.md``. This script computes
exactly what it names and nothing else: the three gates, the paired v400-v10
contrast per OFI arm, the difference-in-differences that *is* the claim, the
four-level dose-response, and the adverse-selection-per-fill vs spread
relationship that the stated falsifier turns on.

Everything here is **per seed**, because every contrast in the pre-registration
is paired by seed. The existing tooling (``scripts/quote_stats.py``,
``scripts/offset_distribution.py``) aggregates per cell and so cannot form a
paired difference; this is the per-seed version of the same measurements,
computed through the same shared quantisation path.

The offset mapping is never re-derived here. Every trajectory is replayed under
the ``QuoteMode`` read from **its own** ``run_meta.json`` and pushed through
``quote_analysis.effective_quotes`` / ``participation_stats``, which call the
same ``actions.resolve_quote`` the live agent calls. ``scripts/fill_skew.py``
once hand-rolled the offset arithmetic and was wrong; there is no second copy
of it here.

Per seed, the three eval trajectories (eval seeds 42/43/44) are **concatenated**
before any statistic is taken, so a seed contributes one number per DV. Fill
ratios are pooled the same way - total dollars over total fills across the three
files - not averaged over per-file ratios, which would weight a 40-fill file the
same as a 400-fill one.

    python scripts/p2_analysis.py --runs-root ~/marl-lob/runs --out p2_analysis.json
    python scripts/p2_analysis.py --json p2_analysis.json --report   # tables only
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from marl_lob.actions import DEFAULT_MIN_OFFSET_TICKS
from marl_lob.quote_analysis import QuoteMode, effective_quotes, participation_stats

DOSES = (10, 50, 150, 400)
ARMS = ("ofi_off", "ofi_on")
CELLS = tuple(f"v{v}_{a}" for a in ARMS for v in DOSES)

# Observation layout (observation_extractor.py): 4K+4 base features with K=10,
# then the two optional OFI channels, then the identity one-hot. So with
# include_ofi the channels are at 44 (last-step OFI) and 45 (EWMA OFI).
OFI_COLS = (44, 45)
OFI_NAMES = ("ofi_last", "ofi_ewma")


# ---------------------------------------------------------------- per-seed ---

def markout(d) -> dict:
    """Spread capture and adverse selection for one trajectory.

    Same decomposition as ``experiments/collect.py`` and ``scripts/markout.py``:
    the reference price for a fill is the mid at the wake-up strictly *before*
    it, because the mid at or after the fill already contains the move that
    caused it. With inventory starting flat, d_equity = capture + adverse.
    """
    if "fill_timestamps" not in d:
        return {"n_fills": 0, "shares": 0, "capture": 0.0, "adverse": 0.0,
                "d_equity": 0.0}
    ts = d["timestamps"]
    mid = d["mid_price"].astype(np.int64)
    ft = d["fill_timestamps"]
    fs = d["fill_side"].astype(np.int64)
    fp = d["fill_price"].astype(np.int64)
    fq = d["fill_quantity"].astype(np.int64)
    qi = np.clip(np.searchsorted(ts, ft, side="left") - 1, 0, len(mid) - 1)
    mid_q = mid[qi]
    eq = d["cash"].astype(np.int64) + d["inventory"].astype(np.int64) * mid
    return {
        "n_fills": int(len(ft)),
        "shares": int(fq.sum()),
        "capture": float((fs * fq * (mid_q - fp)).sum() / 100.0),
        "adverse": float((fs * fq * (mid[-1] - mid_q)).sum() / 100.0),
        "d_equity": float((eq[-1] - eq[0]) / 100.0),
    }


def tick_summary(ticks: np.ndarray, floor_tick: int, ceil_tick: int) -> dict:
    """Dispersion of the resolved offsets, over quoting steps only.

    A zero tick means the side posted nothing; including those would report the
    withdrawal rate as offset dispersion.
    """
    t = ticks[ticks > 0]
    if t.size == 0:
        return {"n_quote_steps": 0, "distinct_ticks": 0, "tick_sd": 0.0,
                "tick_mean": float("nan"), "floor_share_pct": float("nan"),
                "ceil_share_pct": float("nan")}
    vals, counts = np.unique(t, return_counts=True)
    return {
        "n_quote_steps": int(t.size),
        "distinct_ticks": int(vals.size),
        "tick_mean": float(t.mean()),
        "tick_sd": float(t.std()),
        "tick_min": int(vals.min()),
        "tick_max": int(vals.max()),
        "floor_share_pct": 100.0 * float((t == floor_tick).sum()) / t.size,
        "ceil_share_pct": 100.0 * float((t == ceil_tick).sum()) / t.size,
        "histogram": {int(v): int(c) for v, c in zip(vals, counts, strict=False)},
    }


def seed_row(run_dir: Path, max_size: int = 100) -> dict | None:
    """Every per-seed quantity, from that seed's three eval trajectories."""
    meta_path = run_dir / "run_meta.json"
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text())
    margs = meta.get("args", {})
    mode = QuoteMode.from_run_meta(meta)

    npzs = sorted((run_dir / "eval").glob("*.npz"))
    if not npzs:
        return None

    acts, obs, fills = [], [], []
    for npz in npzs:
        d = np.load(npz)
        a = np.asarray(d["actions"], dtype=np.float32)
        acts.append(a)
        o = np.asarray(d["observations"], dtype=np.float32)
        obs.append(o[:len(a)])
        fills.append(markout(d))
    A = np.concatenate(acts)
    observations = np.concatenate(obs)

    stats = participation_stats(A, max_size, **mode.kwargs())
    quotes = effective_quotes(A, max_size, **mode.kwargs())

    floor_tick = int(margs.get("min_offset_ticks", DEFAULT_MIN_OFFSET_TICKS))
    span = margs.get("offset_span")
    ceil_tick = (floor_tick + int(span) if span
                 else int(margs.get("max_offset_cents", 50)))

    both = np.concatenate([quotes[:, 0], quotes[:, 1]])

    n_fills = sum(f["n_fills"] for f in fills)
    shares = sum(f["shares"] for f in fills)
    capture = sum(f["capture"] for f in fills)
    adverse = sum(f["adverse"] for f in fills)

    row = {
        "seed": int(margs.get("seed", -1)),
        "run_dir": str(run_dir),
        "mode": mode.label(),
        "n_eval_files": len(npzs),
        "n_steps": int(len(A)),
        "spread_cents": stats["spread_cents"],
        "withdrawn_pct": stats["withdrawn_pct"],
        "two_sided_pct": stats["two_sided_pct"],
        "any_quote_pct": stats["any_quote_pct"],
        "crossed_pct": stats["crossed_pct"],
        "quote_size": stats["quote_size"],
        "floor_tick": floor_tick,
        "ceil_tick": ceil_tick,
        "ticks_both": tick_summary(both, floor_tick, ceil_tick),
        "ticks_bid": tick_summary(quotes[:, 0], floor_tick, ceil_tick),
        "ticks_ask": tick_summary(quotes[:, 1], floor_tick, ceil_tick),
        "n_fills": n_fills,
        "shares": shares,
        "capture": capture,
        "adverse": adverse,
        "d_equity": sum(f["d_equity"] for f in fills),
        "capture_per_fill": capture / n_fills if n_fills else float("nan"),
        "adverse_per_fill": adverse / n_fills if n_fills else float("nan"),
        "capture_per_share_c": 100.0 * capture / shares if shares else float("nan"),
        "adverse_per_share_c": 100.0 * adverse / shares if shares else float("nan"),
        # Per *eval trajectory*, so these are directly comparable with the
        # rescale suite's per-trajectory table (gate 1, control reproduction).
        "d_equity_per_eval": sum(f["d_equity"] for f in fills) / len(npzs),
        "fills_per_eval": n_fills / len(npzs),
        "tick_mean": tick_summary(both, floor_tick, ceil_tick)["tick_mean"],
        "per_file": fills,
    }

    if bool(margs.get("ofi", False)):
        if observations.shape[1] <= max(OFI_COLS):
            row["ofi_error"] = f"observation width {observations.shape[1]} has no OFI columns"
        else:
            for name, col in zip(OFI_NAMES, OFI_COLS, strict=False):
                row[f"{name}_std"] = float(observations[:, col].std())
                row[f"{name}_mean"] = float(observations[:, col].mean())
                row[f"{name}_absmean"] = float(np.abs(observations[:, col]).mean())
    return row


# ------------------------------------------------------------------ stats ---

# Two-sided t critical values, df = n-1. Written out so the script has no scipy
# dependency; the suite is fixed at 20 seeds so df=19 is the one that is used.
_T95 = {9: 2.262, 14: 2.145, 15: 2.131, 18: 2.101, 19: 2.093, 20: 2.086,
        29: 2.045, 39: 2.023, 59: 2.001, 119: 1.980}


def t_crit(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in _T95:
        return _T95[df]
    return 1.96 + 2.4 / df  # adequate for df well past the range above


def _t_sf(t: float, df: int) -> float:
    """Two-sided p-value for Student-t, by the incomplete beta function."""
    if not math.isfinite(t) or df <= 0:
        return float("nan")
    x = df / (df + t * t)
    return _betainc(df / 2.0, 0.5, x)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b), continued-fraction form."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta) / a
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x)
    lbeta2 = lbeta
    front2 = math.exp(math.log(1 - x) * b + math.log(x) * a - lbeta2) / b
    return 1.0 - front2 * _betacf(b, a, 1.0 - x)


def _betacf(a: float, b: float, x: float, itmax: int = 300,
            eps: float = 1e-12) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def paired(diffs: list[float]) -> dict:
    """Mean, SEM, 95% CI, sign counts and paired-t p for one set of differences.

    The sign counts are reported because with n=20 and a seed dispersion this
    large, a mean can be carried by two seeds; 11-9 and 19-1 are different
    findings behind the same interval.
    """
    d = np.asarray([x for x in diffs if math.isfinite(x)], dtype=float)
    n = int(d.size)
    if n < 2:
        return {"n": n, "mean": float("nan"), "sem": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"),
                "sd": float("nan"), "t": float("nan"), "p": float("nan"),
                "n_pos": 0, "n_neg": 0, "n_zero": 0}
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    sem = sd / math.sqrt(n)
    tc = t_crit(n - 1)
    t = mean / sem if sem > 0 else float("nan")
    return {
        "n": n, "mean": mean, "sd": sd, "sem": sem,
        "ci_lo": mean - tc * sem, "ci_hi": mean + tc * sem,
        "t": t, "p": _t_sf(t, n - 1),
        "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()),
        "n_zero": int((d == 0).sum()),
    }


def describe(vals: list[float]) -> dict:
    v = np.asarray([x for x in vals if math.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"),
                "min": float("nan"), "max": float("nan")}
    return {"n": int(v.size), "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "sem": (float(v.std(ddof=1) / math.sqrt(v.size))
                    if v.size > 1 else float("nan")),
            "min": float(v.min()), "max": float(v.max()),
            "median": float(np.median(v))}


def ols(x: list[float], y: list[float]) -> dict:
    """Simple regression with the slope's standard error and p-value.

    Reported with the slope in the DV's own units per cent of spread, so the
    effect size is readable next to the seed dispersion rather than only as a
    correlation.
    """
    pairs = [(a, b) for a, b in zip(x, y, strict=False)
             if math.isfinite(a) and math.isfinite(b)]
    n = len(pairs)
    if n < 3:
        return {"n": n, "slope": float("nan"), "se": float("nan"),
                "p": float("nan"), "r": float("nan"), "intercept": float("nan")}
    xa = np.array([p[0] for p in pairs])
    ya = np.array([p[1] for p in pairs])
    xm, ym = xa.mean(), ya.mean()
    sxx = float(((xa - xm) ** 2).sum())
    if sxx == 0:
        return {"n": n, "slope": float("nan"), "se": float("nan"),
                "p": float("nan"), "r": float("nan"), "intercept": float("nan"),
                "note": "no variation in x"}
    slope = float(((xa - xm) * (ya - ym)).sum() / sxx)
    intercept = float(ym - slope * xm)
    resid = ya - (intercept + slope * xa)
    dof = n - 2
    s2 = float((resid ** 2).sum() / dof)
    se = math.sqrt(s2 / sxx)
    t = slope / se if se > 0 else float("nan")
    syy = float(((ya - ym) ** 2).sum())
    r = (float(((xa - xm) * (ya - ym)).sum() / math.sqrt(sxx * syy))
         if syy > 0 else float("nan"))
    return {"n": n, "slope": slope, "se": se, "t": t, "p": _t_sf(t, dof),
            "r": r, "intercept": intercept, "x_mean": float(xm),
            "x_sd": float(xa.std(ddof=1)), "y_mean": float(ym),
            "y_sd": float(ya.std(ddof=1))}


def demeaned_ols(groups: dict[str, tuple[list[float], list[float]]]) -> dict:
    """Within-group regression: each group's x and y centred on its own mean.

    Equivalent to a cell-fixed-effects slope. This is the version that answers
    "does spread move adverse selection *holding informed flow fixed*", which
    the between-cell version cannot, since the cells differ in informed flow by
    construction and that is itself a driver of adverse selection.
    """
    xs, ys = [], []
    for gx, gy in groups.values():
        pairs = [(a, b) for a, b in zip(gx, gy, strict=False)
                 if math.isfinite(a) and math.isfinite(b)]
        if len(pairs) < 2:
            continue
        xa = np.array([p[0] for p in pairs])
        ya = np.array([p[1] for p in pairs])
        xs.extend(xa - xa.mean())
        ys.extend(ya - ya.mean())
    out = ols(xs, ys)
    # One df is spent per group mean, not one overall intercept.
    out["note"] = ("within-cell (cell-demeaned); p uses n-2 df and is "
                   "therefore slightly anticonservative")
    return out


# ----------------------------------------------------------------- collect ---

def collect(root: Path, cells=CELLS, max_size: int = 100) -> dict:
    out = {"runs_root": str(root), "cells": {}}
    for cell in cells:
        cell_dir = root / cell
        if not cell_dir.is_dir():
            out["cells"][cell] = {"error": "missing"}
            continue
        run_dirs = sorted((p for p in cell_dir.iterdir() if p.is_dir()),
                          key=lambda p: (len(p.name), p.name))
        rows = []
        for rd in run_dirs:
            r = seed_row(rd, max_size)
            if r is not None:
                rows.append(r)
        rows.sort(key=lambda r: r["seed"])
        out["cells"][cell] = {"n_seeds": len(rows), "seeds": rows}
    return out


def by_seed(cell: dict, key: str) -> dict[int, float]:
    return {r["seed"]: r.get(key, float("nan")) for r in cell["seeds"]}


def pair_diff(data: dict, cell_hi: str, cell_lo: str, key: str) -> list[float]:
    hi, lo = by_seed(data["cells"][cell_hi], key), by_seed(data["cells"][cell_lo], key)
    return [hi[s] - lo[s] for s in sorted(set(hi) & set(lo))]


def analyse(data: dict) -> dict:
    dvs = ("spread_cents", "withdrawn_pct", "two_sided_pct")
    res: dict = {"dose_response": {}, "contrast": {}, "did": {},
                 "dispersion": {}, "gates": {}, "fills": {}}

    # dose-response and cross-seed dispersion, per cell per DV
    for cell, c in data["cells"].items():
        if "seeds" not in c:
            continue
        res["dose_response"][cell] = {
            k: describe([r[k] for r in c["seeds"]])
            for k in dvs + ("capture_per_fill", "adverse_per_fill",
                            "capture_per_share_c", "adverse_per_share_c",
                            "d_equity_per_eval", "fills_per_eval", "quote_size",
                            "tick_mean")
        }

    # v400 - v10, paired by seed, per arm; and the difference-in-differences
    for k in dvs:
        res["contrast"][k] = {
            arm: paired(pair_diff(data, f"v400_{arm}", f"v10_{arm}", k))
            for arm in ARMS
        }
        on = dict(zip(sorted(by_seed(data["cells"]["v400_ofi_on"], k)),
                      pair_diff(data, "v400_ofi_on", "v10_ofi_on", k), strict=False))
        off = dict(zip(sorted(by_seed(data["cells"]["v400_ofi_off"], k)),
                       pair_diff(data, "v400_ofi_off", "v10_ofi_off", k), strict=False))
        shared = sorted(set(on) & set(off))
        res["did"][k] = paired([on[s] - off[s] for s in shared])
        # every adjacent dose step too, so a non-monotone shape is visible
        res["contrast"][k]["steps"] = {
            f"v{hi}-v{lo}|{arm}": paired(
                pair_diff(data, f"v{hi}_{arm}", f"v{lo}_{arm}", k))
            for arm in ARMS
            for hi, lo in zip(DOSES[1:], DOSES[:-1], strict=False)
        }

    # gate 2 - dispersion
    for cell, c in data["cells"].items():
        if "seeds" not in c:
            continue
        per_seed_pass = [
            (r["ticks_both"]["distinct_ticks"] >= 12
             and r["ticks_both"]["tick_sd"] > 2.0)
            for r in c["seeds"]
        ]
        pooled_hist: dict[int, int] = {}
        for r in c["seeds"]:
            for t, n in r["ticks_both"]["histogram"].items():
                pooled_hist[int(t)] = pooled_hist.get(int(t), 0) + int(n)
        tot = sum(pooled_hist.values())
        mean = sum(t * n for t, n in pooled_hist.items()) / tot if tot else float("nan")
        sd = (math.sqrt(sum(n * (t - mean) ** 2 for t, n in pooled_hist.items()) / tot)
              if tot else float("nan"))
        floor_t = c["seeds"][0]["floor_tick"]
        ceil_t = c["seeds"][0]["ceil_tick"]
        res["gates"].setdefault("dispersion", {})[cell] = {
            "pooled_distinct_ticks": len(pooled_hist),
            "pooled_tick_sd": sd,
            "pooled_tick_mean": mean,
            "pooled_floor_share_pct": (100.0 * pooled_hist.get(floor_t, 0) / tot
                                       if tot else float("nan")),
            "pooled_ceil_share_pct": (100.0 * pooled_hist.get(ceil_t, 0) / tot
                                      if tot else float("nan")),
            "seeds_passing": int(sum(per_seed_pass)),
            "n_seeds": len(per_seed_pass),
            "min_seed_distinct": min(r["ticks_both"]["distinct_ticks"] for r in c["seeds"]),
            "min_seed_tick_sd": min(r["ticks_both"]["tick_sd"] for r in c["seeds"]),
            "seed_distinct": describe([r["ticks_both"]["distinct_ticks"] for r in c["seeds"]]),
            "seed_tick_sd": describe([r["ticks_both"]["tick_sd"] for r in c["seeds"]]),
            "seed_floor_share": describe([r["ticks_both"]["floor_share_pct"] for r in c["seeds"]]),
            "seed_ceil_share": describe([r["ticks_both"]["ceil_share_pct"] for r in c["seeds"]]),
            "failing_seeds": [r["seed"] for r, ok in zip(c["seeds"], per_seed_pass, strict=False)
                              if not ok],
        }

    # gate 3 - feature health
    for cell, c in data["cells"].items():
        if "seeds" not in c or not cell.endswith("ofi_on"):
            continue
        res["gates"].setdefault("feature_health", {})[cell] = {
            name: describe([r.get(f"{name}_std", float("nan")) for r in c["seeds"]])
            for name in OFI_NAMES
        }

    # the falsifier: adverse selection per fill against quoted spread
    groups = {cell: ([r["spread_cents"] for r in c["seeds"]],
                     [r["adverse_per_fill"] for r in c["seeds"]])
              for cell, c in data["cells"].items() if "seeds" in c}
    res["fills"]["within_cell"] = {
        cell: ols(x, y) for cell, (x, y) in groups.items()}
    res["fills"]["within_cell_pooled"] = demeaned_ols(groups)
    res["fills"]["within_cell_v400"] = demeaned_ols(
        {k: v for k, v in groups.items() if k.startswith("v400_")})
    res["fills"]["between_cell"] = ols(
        [np.nanmean(x) for x, _ in groups.values()],
        [np.nanmean(y) for _, y in groups.values()])
    res["fills"]["all_runs"] = ols(
        [v for x, _ in groups.values() for v in x],
        [v for _, y in groups.values() for v in y])
    # same, for capture per fill, as the contrast the pre-registration draws
    cap = {cell: ([r["spread_cents"] for r in c["seeds"]],
                  [r["capture_per_fill"] for r in c["seeds"]])
           for cell, c in data["cells"].items() if "seeds" in c}
    res["fills"]["capture_within_cell_pooled"] = demeaned_ols(cap)
    res["fills"]["capture_between_cell"] = ols(
        [np.nanmean(x) for x, _ in cap.values()],
        [np.nanmean(y) for _, y in cap.values()])
    res["fills"]["adverse_per_fill_did"] = paired([
        a - b for a, b in zip(
            pair_diff(data, "v400_ofi_on", "v10_ofi_on", "adverse_per_fill"),
            pair_diff(data, "v400_ofi_off", "v10_ofi_off", "adverse_per_fill"), strict=False)])
    return res


# ------------------------------------------------------------------ report ---

def f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) \
        else f"{x:.{nd}f}"


def report(data: dict, res: dict) -> str:
    L: list[str] = []
    ap = L.append

    ap("=== GATE 1: control reproduction, v50_ofi_off vs rescale `outward` ===")
    ap("(rescale outward, per eval trajectory: d_equity -721, spread 16.94, fills 316)")
    ctl = res["dose_response"].get("v50_ofi_off", {})
    for k, target in (("d_equity_per_eval", -721.0), ("spread_cents", 16.94),
                      ("fills_per_eval", 316.0)):
        d = ctl.get(k, {})
        ap(f"  {k:<20} p2 {f(d.get('mean'),2):>10}  +/- sd {f(d.get('sd'),2):>8}"
           f"   rescale {target:>10.2f}")

    ap("")
    ap("=== RESOLUTION CONTEXT: spread of the per-seed values themselves ===")
    ap(f"{'quantity':<22}{'min seed':>10}{'max seed':>10}{'ratio':>8}"
       f"{'mean cross-seed sd':>21}")
    for k, nd in (("spread_cents", 2), ("tick_mean", 2), ("withdrawn_pct", 1),
                  ("two_sided_pct", 1), ("adverse_per_fill", 3)):
        vals = [r[k] for c in data["cells"].values() if "seeds" in c
                for r in c["seeds"] if math.isfinite(r[k])]
        sds = [res["dose_response"][c][k]["sd"] for c in CELLS
               if c in res["dose_response"]]
        lo, hi = min(vals), max(vals)
        ratio = hi / lo if lo not in (0,) and lo * hi > 0 else float("nan")
        ap(f"{k:<22}{f(lo,nd):>10}{f(hi,nd):>10}{f(ratio,2):>8}"
           f"{f(float(np.mean(sds)),3):>21}")

    ap("")
    ap("=== GATE 2: dispersion (>=12 of 16 ticks, tick sd > 2) ===")
    ap(f"{'cell':<14}{'distinct':>9}{'tick sd':>9}{'tick mean':>11}"
       f"{'floor%':>9}{'ceil%':>9}{'seeds pass':>12}{'min dist':>10}{'min sd':>9}")
    for cell in CELLS:
        g = res["gates"]["dispersion"].get(cell)
        if not g:
            continue
        ap(f"{cell:<14}{g['pooled_distinct_ticks']:>9}{f(g['pooled_tick_sd'],2):>9}"
           f"{f(g['pooled_tick_mean'],2):>11}{f(g['pooled_floor_share_pct'],1):>9}"
           f"{f(g['pooled_ceil_share_pct'],1):>9}"
           f"{str(g['seeds_passing'])+'/'+str(g['n_seeds']):>12}"
           f"{g['min_seed_distinct']:>10}{f(g['min_seed_tick_sd'],2):>9}")

    ap("")
    ap("=== GATE 3: feature health, OFI channel std (target ~0.4) ===")
    ap(f"{'cell':<14}{'ofi_last sd':>13}{'range':>18}{'ofi_ewma sd':>13}{'range':>18}")
    for cell in CELLS:
        g = res["gates"].get("feature_health", {}).get(cell)
        if not g:
            continue
        ap(f"{cell:<14}{f(g['ofi_last']['mean']):>13}"
           f"{f(g['ofi_last']['min'])+'-'+f(g['ofi_last']['max']):>18}"
           f"{f(g['ofi_ewma']['mean']):>13}"
           f"{f(g['ofi_ewma']['min'])+'-'+f(g['ofi_ewma']['max']):>18}")

    ap("")
    ap("=== DOSE-RESPONSE (per-seed mean +/- cross-seed sd, n=20) ===")
    for k in ("spread_cents", "withdrawn_pct", "two_sided_pct", "tick_mean",
              "capture_per_fill", "adverse_per_fill", "capture_per_share_c",
              "adverse_per_share_c", "fills_per_eval", "d_equity_per_eval"):
        ap(f"-- {k}")
        ap(f"{'arm':<10}" + "".join(f"{'v'+str(v):>22}" for v in DOSES))
        for arm in ARMS:
            cells = [f"v{v}_{arm}" for v in DOSES]
            vals = []
            for c in cells:
                d = res["dose_response"].get(c, {}).get(k, {})
                vals.append(f"{f(d.get('mean'),3)} +/- {f(d.get('sd'),3)}")
            ap(f"{arm:<10}" + "".join(f"{v:>22}" for v in vals))

    ap("")
    ap("=== CONTRAST v400 - v10, paired by seed ===")
    ap(f"{'DV':<16}{'arm':<10}{'mean':>10}{'sem':>9}{'95% CI':>22}"
       f"{'p':>9}{'+/-/0':>12}{'cross-seed sd':>15}")
    for k in ("spread_cents", "withdrawn_pct", "two_sided_pct"):
        for arm in ARMS:
            s = res["contrast"][k][arm]
            sds = [res["dose_response"][f"v{v}_{arm}"][k]["sd"] for v in DOSES]
            signs = f"{s['n_pos']:d}/{s['n_neg']:d}/{s['n_zero']:d}"
            ap(f"{k:<16}{arm:<10}{f(s['mean']):>10}{f(s['sem']):>9}"
               f"{'['+f(s['ci_lo'])+', '+f(s['ci_hi'])+']':>22}{f(s['p'],4):>9}"
               f"{signs:>12}{f(float(np.mean(sds))):>15}")

    ap("")
    ap("=== ADJACENT DOSE STEPS, paired by seed (monotonicity check) ===")
    ap(f"{'DV':<16}{'step':<20}{'mean':>10}{'sem':>9}{'95% CI':>22}{'p':>9}{'+/-/0':>12}")
    for k in ("spread_cents", "withdrawn_pct", "two_sided_pct"):
        for name, st in res["contrast"][k]["steps"].items():
            signs = f"{st['n_pos']:d}/{st['n_neg']:d}/{st['n_zero']:d}"
            ap(f"{k:<16}{name:<20}{f(st['mean']):>10}{f(st['sem']):>9}"
               f"{'['+f(st['ci_lo'])+', '+f(st['ci_hi'])+']':>22}"
               f"{f(st['p'],4):>9}{signs:>12}")

    ap("")
    ap("=== DIFFERENCE-IN-DIFFERENCES  (v400-v10 | on) - (v400-v10 | off) ===")
    ap(f"{'DV':<16}{'mean':>10}{'sem':>9}{'95% CI':>24}{'p':>9}{'+/-/0':>12}")
    for k in ("spread_cents", "withdrawn_pct", "two_sided_pct"):
        s = res["did"][k]
        signs = f"{s['n_pos']:d}/{s['n_neg']:d}/{s['n_zero']:d}"
        ap(f"{k:<16}{f(s['mean']):>10}{f(s['sem']):>9}"
           f"{'['+f(s['ci_lo'])+', '+f(s['ci_hi'])+']':>24}{f(s['p'],4):>9}"
           f"{signs:>12}")

    ap("")
    a = res["fills"]["adverse_per_fill_did"]
    signs = f"{a['n_pos']:d}/{a['n_neg']:d}/{a['n_zero']:d}"
    ap(f"adverse_per_fill DiD:  mean {f(a['mean'])}  sem {f(a['sem'])}"
       f"  95% CI [{f(a['ci_lo'])}, {f(a['ci_hi'])}]  p {f(a['p'], 4)}  {signs}")

    ap("")
    ap("=== ADVERSE SELECTION PER FILL vs QUOTED SPREAD ===")
    ap("slope is $/fill per cent of quoted spread; adverse/fill is negative")
    ap("(a loss), so a POSITIVE slope = widening reduces adverse selection,")
    ap("which is the pre-registered falsifier.")
    ap(f"{'regression':<26}{'n':>5}{'slope':>10}{'se':>9}{'95% CI':>22}"
       f"{'p':>9}{'r':>8}")
    for name, key in (("between cells (8 pts)", "between_cell"),
                      ("all runs, pooled", "all_runs"),
                      ("within cell (all 8)", "within_cell_pooled"),
                      ("within cell (v400 only)", "within_cell_v400"),
                      ("capture: between cells", "capture_between_cell"),
                      ("capture: within cell", "capture_within_cell_pooled")):
        o = res["fills"][key]
        tc = t_crit(o["n"] - 2)
        lo, hi = o["slope"] - tc * o["se"], o["slope"] + tc * o["se"]
        ap(f"{name:<26}{o['n']:>5}{f(o['slope'],4):>10}{f(o['se'],4):>9}"
           f"{'['+f(lo,4)+', '+f(hi,4)+']':>22}{f(o['p'],4):>9}{f(o['r'],3):>8}")
    ap("")
    ap("per-cell within-cell slopes (adverse $/fill per cent of spread):")
    ap(f"{'cell':<14}{'n':>4}{'slope':>10}{'se':>9}{'p':>9}{'r':>8}"
       f"{'spread sd':>11}{'adv/fill sd':>13}")
    for cell in CELLS:
        o = res["fills"]["within_cell"].get(cell)
        if not o:
            continue
        ap(f"{cell:<14}{o['n']:>4}{f(o['slope'],4):>10}{f(o['se'],4):>9}"
           f"{f(o['p'],4):>9}{f(o['r'],3):>8}{f(o.get('x_sd'),2):>11}"
           f"{f(o.get('y_sd'),3):>13}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=None,
                    help="directory containing the p2/ suite; default <repo>/runs")
    ap.add_argument("--suite", default="p2")
    ap.add_argument("--json", default=None,
                    help="load a previously written per-seed dump instead of "
                         "re-reading the trajectories")
    ap.add_argument("--out", default=None, help="write the per-seed dump here")
    ap.add_argument("--analysis-out", default=None)
    ap.add_argument("--report-out", default=None)
    ap.add_argument("--max-size", type=int, default=100)
    args = ap.parse_args()

    if args.json:
        data = json.loads(Path(args.json).read_text())
    else:
        repo = Path(__file__).resolve().parent.parent
        root = Path(args.runs_root).expanduser() if args.runs_root else repo / "runs"
        data = collect(root / args.suite, max_size=args.max_size)
        if args.out:
            Path(args.out).write_text(json.dumps(data))

    res = analyse(data)
    if args.analysis_out:
        Path(args.analysis_out).write_text(json.dumps(res, indent=1))
    text = report(data, res)
    print(text)
    if args.report_out:
        Path(args.report_out).write_text(text + "\n")


if __name__ == "__main__":
    main()
