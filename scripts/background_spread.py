#!/usr/bin/env python3
"""Measure the background market's spread and depth geometry in cents.

The RL agent quotes an integer offset from mid, so the question this answers is
purely descriptive: what scale of offsets does the *background* rmsc03_simple
market actually produce? No trained agent and no MARL agents are involved - this
runs the background config alone.

Method
------
`book_logging=True` is passed through `config_kwargs` (rmsc03_simple supports
this per-run override; the file's default stays OFF, nothing is edited). The
exchange then snapshots the top `book_log_depth` levels of the book on every
change, and `OrderBook.get_L2_snapshots` hands them back.

Those snapshots are in *event* time, so they over-weight busy moments. The
script therefore reports two samplings:

  event : one row per book change, as logged
  clock : the last snapshot at or before each point on a fixed grid
          (default 1s, matching the env's `wakeup_interval`), i.e. the book as
          the agent would see it at its decision points

Padded levels (ABIDES pads a short book with synthetic prices at zero size) are
dropped by requiring qty > 0.

    python scripts/background_spread.py --seeds 1,2,3 --end 10:30:00
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
from abides_core import abides
from abides_core.utils import ns_date, str_to_ns

from marl_lob.configs import rmsc03_simple

NLEVELS = 10


def run_one(seed: int, start: str, end: str, book_log_depth: int = NLEVELS):
    """Run the background sim once and return per-snapshot arrays."""
    config = rmsc03_simple.build_config(
        start_time=start,
        end_time=end,
        seed=seed,
        book_logging=True,          # per-run override, file default stays off
        book_log_depth=book_log_depth,
    )
    end_state = abides.run(config)
    book = end_state["agents"][0].order_books["ABM"]
    L2 = book.get_L2_snapshots(nlevels=NLEVELS)
    return L2


def snapshot_rows(L2):
    """-> (times_ns, list of (bid_levels, ask_levels)) keeping only qty>0."""
    times = np.asarray(L2["times"], dtype=np.int64)
    out = []
    for i in range(len(times)):
        b = np.asarray(L2["bids"][i])
        a = np.asarray(L2["asks"][i])
        b = b[b[:, 1] > 0] if b.size else b
        a = a[a[:, 1] > 0] if a.size else a
        out.append((b, a))
    return times, out


def clock_subsample(times, rows, grid_ns):
    """Last snapshot at or before each grid point. Returns (times, rows)."""
    if len(times) == 0:
        return times, rows
    t0, t1 = times[0], times[-1]
    grid = np.arange(t0, t1 + 1, grid_ns, dtype=np.int64)
    # searchsorted 'right' - 1 gives the index of the last snapshot <= g
    idx = np.searchsorted(times, grid, side="right") - 1
    idx = idx[idx >= 0]
    return grid[: len(idx)], [rows[i] for i in idx]


def measure(times, rows, max_levels=5):
    """Compute the spread / distance-from-mid distributions."""
    spreads = []
    d_bid, d_ask = [], []
    lvl_bid = {i: [] for i in range(max_levels)}
    lvl_ask = {i: [] for i in range(max_levels)}
    n_one_sided = 0

    for b, a in rows:
        if len(b) == 0 or len(a) == 0:
            n_one_sided += 1
            continue
        bb, ba = int(b[0, 0]), int(a[0, 0])
        mid = (bb + ba) / 2.0
        spreads.append(ba - bb)
        d_bid.append(mid - bb)
        d_ask.append(ba - mid)
        for i in range(max_levels):
            if i < len(b):
                lvl_bid[i].append(mid - int(b[i, 0]))
            if i < len(a):
                lvl_ask[i].append(int(a[i, 0]) - mid)

    return {
        "n_snapshots": len(rows),
        "n_two_sided": len(spreads),
        "n_one_sided": n_one_sided,
        "t_first_ns": int(times[0]) if len(times) else None,
        "t_last_ns": int(times[-1]) if len(times) else None,
        "spread": np.asarray(spreads, dtype=float),
        "d_bid": np.asarray(d_bid, dtype=float),
        "d_ask": np.asarray(d_ask, dtype=float),
        "lvl_bid": {i: np.asarray(v, dtype=float) for i, v in lvl_bid.items()},
        "lvl_ask": {i: np.asarray(v, dtype=float) for i, v in lvl_ask.items()},
    }


PCTS = [10, 25, 50, 75, 90, 99]


def qtable(x):
    if len(x) == 0:
        return {}
    d = {f"p{p}": float(np.percentile(x, p)) for p in PCTS}
    d.update(mean=float(np.mean(x)), min=float(np.min(x)),
             max=float(np.max(x)), n=int(len(x)))
    return d


def report(tag, m, max_levels=5):
    print(f"\n===== {tag} =====")
    print(f"snapshots: {m['n_snapshots']}  two-sided: {m['n_two_sided']}  "
          f"one-sided/empty: {m['n_one_sided']}")
    if m["t_first_ns"] is not None:
        t0 = m["t_first_ns"] - ns_date(m["t_first_ns"])
        t1 = m["t_last_ns"] - ns_date(m["t_last_ns"])
        print(f"window: {t0/1e9:.1f}s .. {t1/1e9:.1f}s since midnight "
              f"({(t1-t0)/1e9:.1f}s span)")
    for name, arr in (("spread", m["spread"]), ("mid->bestbid", m["d_bid"]),
                      ("mid->bestask", m["d_ask"])):
        q = qtable(arr)
        if q:
            print(f"{name:>14}: mean {q['mean']:7.3f}  " +
                  "  ".join(f"p{p}={q[f'p{p}']:g}" for p in PCTS) +
                  f"  min={q['min']:g} max={q['max']:g}")
    c = Counter(int(v) for v in m["spread"])
    tot = max(sum(c.values()), 1)
    print("  spread histogram (cents: count, pct):")
    for v, n in sorted(c.items())[:25]:
        print(f"    {v:6d}: {n:7d}  {100*n/tot:6.2f}%")
    if len(c) > 25:
        print(f"    ... {len(c)-25} further distinct values")
    print("  cumulative: P(spread <= s) and P(one-sided dist from mid <= d)")
    for t in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 50):
        ps = 100.0 * np.mean(m["spread"] <= t) if len(m["spread"]) else float("nan")
        pdm = 100.0 * np.mean(m["d_ask"] <= t) if len(m["d_ask"]) else float("nan")
        print(f"    <= {t:3d}c : spread {ps:6.2f}%   dist-from-mid {pdm:6.2f}%")
    print("  depth profile, distance from mid (cents):")
    for i in range(max_levels):
        for side, d in (("bid", m["lvl_bid"]), ("ask", m["lvl_ask"])):
            q = qtable(d[i])
            if q:
                print(f"    L{i+1} {side}: mean {q['mean']:8.2f}  " +
                      "  ".join(f"p{p}={q[f'p{p}']:g}" for p in PCTS) +
                      f"  n={q['n']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--start", default="09:30:00")
    ap.add_argument("--end", default="10:30:00")
    ap.add_argument("--grid", default="1s", help="clock-sampling interval")
    ap.add_argument("--warmup", default="60s",
                    help="drop snapshots this far after market open")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--npz-out", default=None,
                    help="dump the raw clock-sample spread/distance arrays")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    grid_ns = str_to_ns(args.grid)
    warm_ns = str_to_ns(args.warmup)

    ev_t, ev_r, ck_t, ck_r = [], [], [], []
    for seed in seeds:
        L2 = run_one(seed, args.start, args.end)
        t, r = snapshot_rows(L2)
        if len(t):
            open_ns = ns_date(t[0]) + str_to_ns(args.start)
            keep = t >= open_ns + warm_ns
            t, r = t[keep], [r[i] for i in np.flatnonzero(keep)]
        ct, cr = clock_subsample(t, r, grid_ns)
        print(f"seed {seed}: {len(t)} event snapshots (post-warmup), "
              f"{len(ct)} clock samples")
        ev_t.append(t)
        ev_r.extend(r)
        ck_t.append(ct)
        ck_r.extend(cr)

    for tag, ts, rs in (("EVENT-TIME (pooled)", ev_t, ev_r),
                        (f"CLOCK-TIME @{args.grid} (pooled)", ck_t, ck_r)):
        cat = (np.concatenate([x for x in ts if len(x)])
               if any(len(x) for x in ts) else np.array([], dtype=np.int64))
        m = measure(cat, rs)
        report(tag, m)
        if args.json_out:
            pass

    if args.npz_out:
        cat = np.concatenate([x for x in ck_t if len(x)])
        mm = measure(cat, ck_r)
        np.savez(args.npz_out, spread=mm["spread"], d_bid=mm["d_bid"],
                 d_ask=mm["d_ask"],
                 **{f"lvl_bid_{i+1}": v for i, v in mm["lvl_bid"].items()},
                 **{f"lvl_ask_{i+1}": v for i, v in mm["lvl_ask"].items()})
        print(f"wrote {args.npz_out}")

    if args.json_out:
        cat = np.concatenate([x for x in ck_t if len(x)])
        m = measure(cat, ck_r)
        blob = {
            "seeds": seeds, "start": args.start, "end": args.end,
            "grid": args.grid, "warmup": args.warmup,
            "n_snapshots": m["n_snapshots"], "n_two_sided": m["n_two_sided"],
            "spread": qtable(m["spread"]),
            "d_bid": qtable(m["d_bid"]), "d_ask": qtable(m["d_ask"]),
            "spread_hist": dict(sorted(Counter(int(v) for v in m["spread"]).items())),
            "lvl_bid": {i + 1: qtable(v) for i, v in m["lvl_bid"].items()},
            "lvl_ask": {i + 1: qtable(v) for i, v in m["lvl_ask"].items()},
        }
        with open(args.json_out, "w") as f:
            json.dump(blob, f, indent=2, default=str)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
