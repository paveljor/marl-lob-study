#!/usr/bin/env python3
"""Replay a suite's saved eval actions through the live quantisation.

Δequity says how much money a policy made; it does not say whether the policy
was making markets. This does. It takes the raw actions written to each eval
trajectory and pushes them through the same rounding that ``actions.py`` applies
before orders reach ABIDES, then reports how often a genuine two-sided quote
actually reached the book.

The distinction that matters: an action vector is not a quote. Offsets are
continuous cents rounded to whole ticks, sizes are rounded to integers, and
``translate_action`` voids *both* sides when the rounded bid would cross the
rounded ask. A policy emitting sub-tick offsets and sub-unit sizes therefore
posts nothing at all while still producing a full-looking action array.

The offset mapping and the mid-rounding mode are read per run from its
run_meta.json and printed, because the reconstruction below is only right if
it matches how the run actually quoted - and nothing else would notice if it
did not.

Handles both action layouts. Under the gated layout ``withdrawn%`` is the
share of steps on which the policy *chose* not to quote - a real behavioural
measurement - whereas under the legacy layout not quoting is mostly an
artifact of sub-tick rounding, and ``crossed%`` counts the self-voided steps
that gating makes structurally impossible.

    python scripts/quote_stats.py retrain
    python scripts/quote_stats.py grid --cells v10_n1,v400_n4
"""
import argparse
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np

from marl_lob.quote_analysis import QuoteMode, participation_stats


def cell_stats(npz_paths, max_size=100):
    """Average the participation stats over every eval file in one cell.

    Each file is replayed under the action semantics **its own run** used,
    read from that run's run_meta.json. Replaying a run under the wrong
    semantics produces plausible, wrong numbers with no error, so the mode is
    also reported in the table rather than assumed: a file with no run_meta
    reads as the legacy mapping, which is what pre-2026-09-05 runs were.
    """
    agg = defaultdict(list)
    modes = set()
    for f in npz_paths:
        d = np.load(f)
        if "actions" not in d:
            continue
        a = d["actions"]
        mode = QuoteMode.for_trajectory(f)
        modes.add(mode.label())
        m = participation_stats(a, max_size, **mode.kwargs())
        for key in ("two_sided_pct", "any_quote_pct", "crossed_pct",
                    "withdrawn_pct", "spread_cents"):
            agg[key].append(m[key])
        agg["gated"].append(float(m["gated"]))
        # Raw means are reported over the offset and size columns only, which
        # sit at different indices in the two layouts.
        off_cols, size_cols = ((2, 4), (4, 6)) if m["gated"] else ((0, 2), (2, 4))
        agg["raw_offset"].append(float(a[:, off_cols[0]:off_cols[1]].mean()))
        agg["raw_size"].append(float(a[:, size_cols[0]:size_cols[1]].mean()))
        agg["fills"].append(
            int(len(d["fill_timestamps"])) if "fill_timestamps" in d else 0
        )
    if not agg:
        return {}
    out = {k: float(np.mean(v)) for k, v in agg.items()}
    out["mode"] = modes.pop() if len(modes) == 1 else "MIXED"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suite")
    ap.add_argument("--cells", help="comma-separated subset; default all")
    ap.add_argument("--max-size", type=int, default=100)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    root = repo / "runs" / args.suite
    if not root.is_dir():
        raise SystemExit(f"no runs at {root} - has the suite been run?")

    cells = (args.cells.split(",") if args.cells
             else sorted(p.name for p in root.iterdir() if p.is_dir()))

    hdr = ("cell", "gated", "mode", "two-sided%", "anyquote%", "withdrawn%",
           "crossed%", "spread(c)", "rawoff", "rawsize", "fills")
    print("{!s:<16}{!s:>7}{!s:>22}{!s:>11}{!s:>11}{!s:>12}{!s:>10}"
          "{!s:>11}{!s:>9}{!s:>9}{!s:>8}".format(*hdr))
    for cell in cells:
        paths = sorted(glob.glob(str(root / cell / "*" / "eval" / "*.npz")))
        if not paths:
            continue
        m = cell_stats(paths, args.max_size)
        if not m:
            continue
        # Keep the numeric rendering identical to historical report tables.
        print(
            "%-16s%7s%22s%11.1f%11.1f%12.1f%10.1f%11.2f%9.3f%9.2f%8.0f" % (  # noqa: UP031
            cell, "yes" if m["gated"] > 0.5 else "no", m["mode"],
            m["two_sided_pct"], m["any_quote_pct"], m["withdrawn_pct"],
            m["crossed_pct"], m["spread_cents"], m["raw_offset"],
            m["raw_size"], m["fills"]))


if __name__ == "__main__":
    main()
