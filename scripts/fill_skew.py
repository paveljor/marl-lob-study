#!/usr/bin/env python3
"""Buy/sell fill skew, inventory drift, and fills per submitted order.

Two questions, one pass over the eval trajectories.

**Q1 - sell-side skew.** ``MarlChild.act_on_wakeup`` computes its quoting
reference as ``mid = (best_bid + best_ask) // 2``. Floor division. When the
book's spread is odd the true mid is a half-integer and the floor lands on the
bid, so a symmetric offset ``off`` posts ``bid = mid - off`` and
``ask = mid + off`` with the ask one tick CLOSER to the true mid than the bid.
Prediction: sells exceed buys and inventory drifts short. Because the exact
integer spread is recoverable from the saved observations (feature ``4K+2`` is
``(ask-bid)/mid_true``), every fill can be attributed to the parity of the
spread at its quote step - which is the discriminating test, not just the
pooled tally.

**Q2 - fills per submitted order.** ``translate_action`` reconciles by
cancel-all-then-replace on every wake-up, so the number of orders the agent
submitted on a side is exactly the number of steps on which that side's
quantised quote had positive size. That is recoverable from the saved
``actions`` array through the live quantisation, so fills per submitted order
needs no new logging.

Sign conventions, taken from the code and not assumed:

* ``marl_agents.snapshot_fills``: ``sign = +1 if order.side.is_bid()``, so a
  resting bid that executed means the agent BOUGHT. **fill_side +1 = buy,
  -1 = sell.**
* ``inventory`` is signed shares, positive = long.
* ``marl_agents._aggregate_fills`` collapses all fills in a step into ONE net
  signed quantity, and ``Trajectory.from_tuples`` then records a single Fill
  whose side is the sign of that net. So a "fill record" is a step with
  non-zero NET fill, not an execution, and the volumes below are net volumes.
  A step that bought 10 and sold 10 records nothing at all. This is the main
  caveat on every Q1 number here.

Usage (root = a directory of ``<cell>/<seed>/eval/*.npz``):

    python scripts/fill_skew.py results/artifacts/latency
    python scripts/fill_skew.py results/artifacts/latency --json /tmp/fs.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from marl_lob.quote_analysis import (
    QuoteMode,
    effective_quotes,
    quote_prices_from_actions,
)

K = 10                 # book levels per side in the observation
SPREAD_FEATURE = 4 * K + 2
BEST_BID_FEATURE = 0
BEST_ASK_FEATURE = 2 * K


def recover_spread(obs: np.ndarray, mid_floor: np.ndarray) -> np.ndarray:
    """Integer spread in cents per snapshot, from the observation's spread
    feature and the floored mid stored in the trajectory.

    ``extract_obs`` stores ``(best_ask - best_bid) / mid_true`` with
    ``mid_true = (B+A)/2`` as a float, while the trajectory stores
    ``mid_floor = (B+A)//2``. The two differ by at most 0.5 cents out of
    ~100,000, so multiplying the feature by ``mid_floor`` and rounding
    recovers the exact integer spread; float32 carries ~7 digits, which is two
    orders of magnitude more than needed for a 1-cent resolution here.
    """
    return np.rint(obs[:, SPREAD_FEATURE].astype(np.float64) * mid_floor).astype(np.int64)


def book_from_spread(mid_floor: np.ndarray, spread: np.ndarray):
    """(best_bid, best_ask, mid_true) from the floored mid and the spread.

    ``mid_floor = B + s//2`` (since ``floor((2B+s)/2) = B + floor(s/2)``), so
    ``B = mid_floor - s//2`` and ``A = B + s``. The true mid sits half a cent
    above the floored one exactly when the spread is odd.
    """
    bid = mid_floor - spread // 2
    ask = bid + spread
    mid_true = mid_floor + 0.5 * (spread % 2)
    return bid, ask, mid_true


def analyse(path: Path, max_size: int = 100) -> dict:
    d = np.load(path)
    ts = d["timestamps"]
    mid = d["mid_price"].astype(np.int64)          # FLOORED mid, as quoted from
    inv = d["inventory"].astype(np.int64)
    obs = d["observations"]
    acts = d["actions"]

    spread = recover_spread(obs, mid)
    best_bid, best_ask, mid_true = book_from_spread(mid, spread)
    odd = (spread % 2) == 1

    # ---- posted orders (denominator for Q2) -------------------------------
    # Cancel-all-then-replace every wake-up => one submitted order per side
    # per step on which the quantised side has positive size. actions[j] is
    # the action taken from snapshot j, and its fills land in snapshot j+1.
    # Which offset mapping and mid-rounding mode this run quoted under. Read
    # from its own run_meta.json; absent (a pre-2026-09-05 artifact) means the
    # legacy semantics, which is what those runs used.
    mode = QuoteMode.for_trajectory(path)
    n_act = len(acts)
    q = effective_quotes(acts, max_size, **mode.kwargs())
    bid_qty, ask_qty = q[:, 2], q[:, 3]
    posted_bid, posted_ask = bid_qty > 0, ask_qty > 0

    # Where the posted quote sat relative to the real touch. Only defined on
    # steps where the book was two-sided (spread > 0).
    two_sided_book = spread[:n_act] > 0
    # Prices come from the shared resolver, not from mid -/+ offset: that
    # identity holds only under mid_rounding="floor". Under "outward" the ask
    # sits a cent higher whenever the spread is odd, which is exactly the
    # asymmetry this script measures, so reconstructing it by hand here would
    # report the fix as though it had not been applied. Zero on a side that
    # posted nothing; every use below is masked by posted_bid / posted_ask.
    doubled_mid = (best_bid + best_ask)[:n_act]
    posted_px = quote_prices_from_actions(
        acts, doubled_mid, max_size, **mode.kwargs())
    post_bid_px, post_ask_px = posted_px[:, 0], posted_px[:, 1]
    at_or_better_bid = posted_bid & two_sided_book & (post_bid_px >= best_bid[:n_act])
    at_or_better_ask = posted_ask & two_sided_book & (post_ask_px <= best_ask[:n_act])
    odd_a = odd[:n_act]
    # Distance from the TRUE mid, in cents. The asymmetry lives here.
    bid_dist = (mid_true[:n_act] - post_bid_px)[posted_bid & two_sided_book]
    ask_dist = (post_ask_px - mid_true[:n_act])[posted_ask & two_sided_book]

    # ---- fills (numerator for Q2, everything for Q1) ----------------------
    ft = d["fill_timestamps"]
    fs = d["fill_side"].astype(np.int64)
    fp = d["fill_price"].astype(np.int64)
    fq = d["fill_quantity"].astype(np.int64)

    # A Fill's timestamp IS the snapshot timestamp of the row it was recorded
    # on, and fills recorded on row i executed against orders submitted from
    # row i-1's action. searchsorted(..., "left") - 1 therefore returns the
    # QUOTE step, which is the index into actions/observations we want. Same
    # convention as scripts/markout.py.
    qi = np.searchsorted(ts, ft, side="left") - 1
    keep = qi >= 0
    qi, fs, fp, fq = qi[keep], fs[keep], fp[keep], fq[keep]
    qi = np.clip(qi, 0, n_act - 1)

    buy, sell = fs > 0, fs < 0
    mq_floor = mid[qi].astype(np.float64)
    mq_true = mid_true[qi]
    odd_q = odd[qi]

    def m(a):
        return float(np.mean(a)) if len(a) else float("nan")

    rec = {
        "file": str(path),
        "quote_mode": mode.label(),
        "n_steps": int(n_act),
        # --- Q1: fill tallies (records = steps with non-zero NET fill) ---
        "buy_fills": int(buy.sum()),
        "sell_fills": int(sell.sum()),
        "buy_vol": int(fq[buy].sum()),
        "sell_vol": int(fq[sell].sum()),
        # --- Q1: inventory ---
        "inv_mean": float(inv.mean()),
        "inv_median": float(np.median(inv)),
        "inv_final": int(inv[-1]),
        "inv_min": int(inv.min()),
        "inv_max": int(inv.max()),
        "frac_steps_short": float((inv < 0).mean()),
        # --- Q1: realised edge vs the mid at the quote step, per side ---
        # Positive = favourable (bought below mid / sold above mid).
        "edge_buy_floor_c": m(mq_floor[buy] - fp[buy]),
        "edge_sell_floor_c": m(fp[sell] - mq_floor[sell]),
        "edge_buy_true_c": m(mq_true[buy] - fp[buy]),
        "edge_sell_true_c": m(fp[sell] - mq_true[sell]),
        # --- Q1: the discriminating split, by spread parity at quote step ---
        "buy_fills_odd": int((buy & odd_q).sum()),
        "sell_fills_odd": int((sell & odd_q).sum()),
        "buy_fills_even": int((buy & ~odd_q).sum()),
        "sell_fills_even": int((sell & ~odd_q).sum()),
        # --- book / quote geometry ---
        "spread_mean_c": float(spread.mean()),
        "frac_spread_odd": float(odd.mean()),
        "frac_spread_eq1": float((spread == 1).mean()),
        "post_bid_dist_true_c": m(bid_dist),
        "post_ask_dist_true_c": m(ask_dist),
        "frac_bid_at_touch": (float(at_or_better_bid.sum()
                                   / max(int((posted_bid & two_sided_book).sum()), 1))),
        "frac_ask_at_touch": (float(at_or_better_ask.sum()
                                   / max(int((posted_ask & two_sided_book).sum()), 1))),
        # --- Q2: submitted orders ---
        "posted_bids": int(posted_bid.sum()),
        "posted_asks": int(posted_ask.sum()),
        "posted_both": int((posted_bid & posted_ask).sum()),
        "posted_any": int((posted_bid | posted_ask).sum()),
        "posted_bid_vol": int(bid_qty[posted_bid].sum()),
        "posted_ask_vol": int(ask_qty[posted_ask].sum()),
    }
    # per-parity posted counts, so fills-per-order can be split the same way
    rec["posted_bids_odd"] = int((posted_bid & odd_a).sum())
    rec["posted_asks_odd"] = int((posted_ask & odd_a).sum())
    rec["posted_bids_even"] = int((posted_bid & ~odd_a).sum())
    rec["posted_asks_even"] = int((posted_ask & ~odd_a).sum())
    # at-touch split the same way, to show the mechanism directly
    for tag, msk in (("odd", odd_a), ("even", ~odd_a)):
        db = posted_bid & two_sided_book & msk
        da = posted_ask & two_sided_book & msk
        rec[f"frac_bid_at_touch_{tag}"] = safe_div(
            int((at_or_better_bid & msk).sum()), int(db.sum()))
        rec[f"frac_ask_at_touch_{tag}"] = safe_div(
            int((at_or_better_ask & msk).sum()), int(da.sum()))
    rec["mean_posted_bid_size"] = m(bid_qty[posted_bid])
    rec["mean_posted_ask_size"] = m(ask_qty[posted_ask])
    return rec


def cell_seed(path: Path, root: Path):
    rel = path.relative_to(root).parts
    return rel[0], rel[1], re.sub(r"\D", "", path.stem[-3:]) or path.stem


def safe_div(a, b):
    return (a / b) if b else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="dir of <cell>/<seed>/eval/*.npz")
    ap.add_argument("--max-size", type=int, default=100)
    ap.add_argument("--json", help="write per-file records here")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    paths = sorted(root.glob("*/*/eval/*.npz"))
    if not paths:
        raise SystemExit(f"no npz under {root}/*/*/eval/")

    recs = []
    for p in paths:
        r = analyse(p, args.max_size)
        cell, seed, eseed = cell_seed(p, root)
        r.update(cell=cell, seed=seed, eval_seed=eseed)
        recs.append(r)

    if args.json:
        Path(args.json).write_text(json.dumps(recs, indent=1))

    by_cell = defaultdict(list)
    for r in recs:
        by_cell[r["cell"]].append(r)

    S = lambda rs, k: sum(r[k] for r in rs)          # noqa: E731
    # nanmean: a file where the policy never posted an ask (or never sold)
    # has an undefined per-side rate, and one such file must not blank the
    # whole cell. The count of such files is reported below.
    M = lambda rs, k: float(np.nanmean([r[k] for r in rs]))  # noqa: E731

    print("=== Q1  buy/sell split, inventory (pooled per cell) ===")
    hdr = ("cell", "files", "buyF", "sellF", "sell%", "buyVol", "sellVol",
           "invMean", "invMed", "invEnd", "short%")
    print("{!s:<8}{!s:>7}{!s:>9}{!s:>9}{!s:>8}{!s:>10}{!s:>10}"
          "{!s:>10}{!s:>9}{!s:>9}{!s:>8}".format(*hdr))
    for cell, rs in sorted(by_cell.items()):
        b, s = S(rs, "buy_fills"), S(rs, "sell_fills")
        # Keep legacy numeric rendering stable for historical report comparisons.
        print(
            "%-8s%7d%9d%9d%8.1f%10d%10d%10.1f%9.1f%9.1f%8.1f" % (  # noqa: UP031
            cell, len(rs), b, s, 100 * safe_div(s, b + s),
            S(rs, "buy_vol"), S(rs, "sell_vol"),
            M(rs, "inv_mean"), M(rs, "inv_median"), M(rs, "inv_final"),
            100 * M(rs, "frac_steps_short")))

    print("\n=== Q1  per-training-seed sign counts (3 eval seeds pooled per seed) ===")
    print("{!s:<8}{!s:>14}{!s:>16}{!s:>16}".format(
        "cell", "sell>buy /20", "invMean<0 /20", "invEnd<0 /20"))
    for cell, rs in sorted(by_cell.items()):
        by_seed = defaultdict(list)
        for r in rs:
            by_seed[r["seed"]].append(r)
        n = len(by_seed)
        sk = sum(S(g, "sell_fills") > S(g, "buy_fills") for g in by_seed.values())
        im = sum(M(g, "inv_mean") < 0 for g in by_seed.values())
        ie = sum(M(g, "inv_final") < 0 for g in by_seed.values())
        print("{!s:<8}{!s:>14}{!s:>16}{!s:>16}".format(
            cell, f"{sk}/{n}", f"{im}/{n}", f"{ie}/{n}"))

    print("\n=== Q1  parity split: sell share of fills by spread parity at quote step ===")
    print("{!s:<8}{!s:>10}{!s:>12}{!s:>12}{!s:>12}{!s:>12}".format(
        "cell", "odd%", "sell%(odd)", "sell%(even)", "spread(c)", "spr==1%"))
    for cell, rs in sorted(by_cell.items()):
        bo, so = S(rs, "buy_fills_odd"), S(rs, "sell_fills_odd")
        be, se = S(rs, "buy_fills_even"), S(rs, "sell_fills_even")
        # Preserve legacy numeric report rendering.
        print("%-8s%10.1f%12.1f%12.1f%12.2f%12.1f" % (  # noqa: UP031
            cell, 100 * M(rs, "frac_spread_odd"),
            100 * safe_div(so, bo + so), 100 * safe_div(se, be + se),
            M(rs, "spread_mean_c"), 100 * M(rs, "frac_spread_eq1")))

    print("\n=== Q1  quote geometry and realised edge (cents, vs TRUE mid) ===")
    print("{!s:<8}{!s:>12}{!s:>12}{!s:>12}{!s:>12}{!s:>12}"
          "{!s:>12}{!s:>12}{!s:>12}".format(
        "cell", "bidDistMid", "askDistMid", "bid@touch%", "ask@touch%",
        "edgeBuy", "edgeSell", "edgeBuyFlr", "edgeSellFlr"))
    for cell, rs in sorted(by_cell.items()):
        # Preserve legacy numeric report rendering.
        print(
            "%-8s%12.3f%12.3f%12.1f%12.1f%12.3f%12.3f%12.3f%12.3f" % (  # noqa: UP031
            cell, M(rs, "post_bid_dist_true_c"), M(rs, "post_ask_dist_true_c"),
            100 * M(rs, "frac_bid_at_touch"), 100 * M(rs, "frac_ask_at_touch"),
            M(rs, "edge_buy_true_c"), M(rs, "edge_sell_true_c"),
            M(rs, "edge_buy_floor_c"), M(rs, "edge_sell_floor_c")))

    print("\n=== Q1  at-touch share by spread parity (the flooring mechanism) ===")
    print("{!s:<8}{!s:>14}{!s:>14}{!s:>14}{!s:>14}".format(
        "cell", "bid@t odd%", "ask@t odd%", "bid@t even%", "ask@t even%"))
    for cell, rs in sorted(by_cell.items()):
        # Preserve legacy numeric report rendering.
        print("%-8s%14.1f%14.1f%14.1f%14.1f" % (  # noqa: UP031
            cell, 100 * M(rs, "frac_bid_at_touch_odd"),
            100 * M(rs, "frac_ask_at_touch_odd"),
            100 * M(rs, "frac_bid_at_touch_even"),
            100 * M(rs, "frac_ask_at_touch_even")))

    print("\n=== Q2  fills per submitted order ===")
    print("{!s:<8}{!s:>9}{!s:>9}{!s:>9}{!s:>9}{!s:>10}{!s:>10}{!s:>10}".format(
        "cell", "fills", "postBid", "postAsk", "postAny",
        "f/order", "buy/bid", "sell/ask"))
    for cell, rs in sorted(by_cell.items()):
        b, s = S(rs, "buy_fills"), S(rs, "sell_fills")
        pb, pa = S(rs, "posted_bids"), S(rs, "posted_asks")
        # Preserve legacy numeric report rendering.
        print("%-8s%9d%9d%9d%9d%10.4f%10.4f%10.4f" % (  # noqa: UP031
            cell, b + s, pb, pa, S(rs, "posted_any"),
            safe_div(b + s, pb + pa), safe_div(b, pb), safe_div(s, pa)))

    print("\n=== Q2  fills per submitted order, split by side x spread parity ===")
    print("{!s:<8}{!s:>14}{!s:>14}{!s:>14}{!s:>14}{!s:>10}{!s:>10}".format(
        "cell", "buy/bid odd", "sell/ask odd", "buy/bid evn", "sell/ask evn",
        "postSzB", "postSzA"))
    for cell, rs in sorted(by_cell.items()):
        # Preserve legacy numeric report rendering.
        print("%-8s%14.4f%14.4f%14.4f%14.4f%10.2f%10.2f" % (  # noqa: UP031
            cell,
            safe_div(S(rs, "buy_fills_odd"), S(rs, "posted_bids_odd")),
            safe_div(S(rs, "sell_fills_odd"), S(rs, "posted_asks_odd")),
            safe_div(S(rs, "buy_fills_even"), S(rs, "posted_bids_even")),
            safe_div(S(rs, "sell_fills_even"), S(rs, "posted_asks_even")),
            M(rs, "mean_posted_bid_size"), M(rs, "mean_posted_ask_size")))

    print("\n=== Q1/Q2  parity diff-in-diff: (sell/ask - buy/bid) odd minus even ===")
    print("{!s:<8}{!s:>12}{!s:>12}{!s:>12}{!s:>14}".format(
        "cell", "gap odd", "gap even", "DiD", "DiD>0 /seeds"))
    for cell, rs in sorted(by_cell.items()):
        def gap(g):
            return (safe_div(S(g, "sell_fills_odd"), S(g, "posted_asks_odd"))
                    - safe_div(S(g, "buy_fills_odd"), S(g, "posted_bids_odd")),
                    safe_div(S(g, "sell_fills_even"), S(g, "posted_asks_even"))
                    - safe_div(S(g, "buy_fills_even"), S(g, "posted_bids_even")))
        go, ge = gap(rs)
        by_seed = defaultdict(list)
        for r in rs:
            by_seed[r["seed"]].append(r)
        n = 0
        for g in by_seed.values():
            o, e = gap(g)
            n += bool(o > e)
        # Preserve legacy numeric report rendering.
        print("%-8s%12.4f%12.4f%12.4f%14s" % (  # noqa: UP031
            cell, go, ge, go - ge, f"{n}/{len(by_seed)}"))

    print("\n=== degenerate files (undefined per-side rates) ===")
    for cell, rs in sorted(by_cell.items()):
        # Preserve legacy numeric report rendering.
        print("  %s: no sell fills %d/%d, no buy fills %d/%d, "  # noqa: UP031
              "no posted asks %d/%d, no posted bids %d/%d" % (
                  cell,
                  sum(r["sell_fills"] == 0 for r in rs), len(rs),
                  sum(r["buy_fills"] == 0 for r in rs), len(rs),
                  sum(r["posted_asks"] == 0 for r in rs), len(rs),
                  sum(r["posted_bids"] == 0 for r in rs), len(rs)))

    print("\n=== Q2  per-side per-seed sign counts: sell/ask > buy/bid ===")
    for cell, rs in sorted(by_cell.items()):
        by_seed = defaultdict(list)
        for r in rs:
            by_seed[r["seed"]].append(r)
        n = sum(1 for g in by_seed.values()
                if safe_div(S(g, "sell_fills"), S(g, "posted_asks"))
                > safe_div(S(g, "buy_fills"), S(g, "posted_bids")))
        print(f"  {cell}: {n}/{len(by_seed)}")

    print("\n=== Q2  per-training-seed monotonicity input: mean fills/order per seed ===")
    cells = sorted(by_cell)
    print(f"{'seed':<8}" + "".join(f"{c!s:>12}" for c in cells))
    seeds = sorted({r["seed"] for r in recs}, key=lambda s: int(re.sub(r"\D", "", s) or 0))
    per_seed = {}
    for sd in seeds:
        row = []
        for c in cells:
            g = [r for r in by_cell[c] if r["seed"] == sd]
            v = safe_div(S(g, "buy_fills") + S(g, "sell_fills"),
                         S(g, "posted_bids") + S(g, "posted_asks")) if g else float("nan")
            row.append(v)
        per_seed[sd] = row
        # Preserve legacy numeric report rendering, including undefined rates.
        print("{!s:<8}{}".format(sd, "".join("%12.4f" % v for v in row)))  # noqa: UP031
    print("\npairwise sign counts on fills/order (rows: A vs B, count of seeds A>B):")
    for i, a in enumerate(cells):
        for bcell in cells[i + 1:]:
            j = cells.index(bcell)
            n = sum(1 for sd in seeds if per_seed[sd][i] > per_seed[sd][j])
            print(f"  {a} > {bcell}: {n}/{len(seeds)}")


if __name__ == "__main__":
    main()
