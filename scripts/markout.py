"""Exact decomposition of market-maker PnL into spread capture vs adverse selection.

The agent quotes at a wake-up using the mid it observes *then*, and fills arrive
over the following interval. So the correct reference price for a fill is the
mid at the PREVIOUS wake-up (the quote-time mid), not the mid recorded at or
after the fill -- that one already contains the price move that caused the fill,
which is why a naive version reports negative spread capture for a market maker.

With inventory starting at 0 this decomposition is exact:

    d_equity = capture + adverse(to session end)
    capture      = sum s*q*(mid_quote - p)        # earned vs your own quote reference
    adverse(h)   = sum s*q*(mid(t+h) - mid_quote) # how the mid moved against you

s = +1 buy / -1 sell, q = size, p = fill price. Integer cents in, dollars out.
"""
import glob
import sys

import numpy as np

LABELS = {"baseline": "F spread10_size100 (default)", "eval": "PPO (exp1 ckpt)"}


def step_mid(ts, mid, targets):
    idx = np.searchsorted(ts, targets, side="right") - 1
    return mid[np.clip(idx, 0, len(mid) - 1)].astype(np.int64)


def analyse(path, horizons=(60.0, 300.0, 900.0)):
    d = np.load(path)
    ts, mid = d["timestamps"], d["mid_price"].astype(np.int64)
    inv, cash = d["inventory"].astype(np.int64), d["cash"].astype(np.int64)
    ft, fs = d["fill_timestamps"], d["fill_side"].astype(np.int64)
    fp, fq = d["fill_price"].astype(np.int64), d["fill_quantity"].astype(np.int64)

    eq = cash + inv * mid
    d_eq = (eq[-1] - eq[0]) / 100.0

    # quote-time mid: the wake-up strictly before the fill
    qi = np.clip(np.searchsorted(ts, ft, side="left") - 1, 0, len(mid) - 1)
    mid_q = mid[qi]

    shares = int(fq.sum())
    capture_c = (fs * fq * (mid_q - fp)).sum()
    capture = capture_c / 100.0
    half_spread = capture_c / shares if shares else float("nan")  # cents per share

    row = {"file": path, "n_fills": len(ft), "shares": shares, "d_equity": d_eq,
           "capture": capture, "half_spread_c": half_spread}
    for h in horizons:
        row[f"adv_{int(h)}"] = (fs * fq * (step_mid(ts, mid, ft + h) - mid_q)).sum() / 100.0
    row["adv_end"] = (fs * fq * (mid[-1] - mid_q)).sum() / 100.0
    row["residual"] = d_eq - (capture + row["adv_end"])
    row["final_inv"] = int(inv[-1])
    return row


def label(path):
    parts = path.split("/")
    cfg = LABELS.get(parts[-2], parts[-2])
    agent = parts[-1].split("_")[-2] if "ppo" in parts[-1] else parts[-1].split("_")[1]
    return f"{cfg}/{agent}"


def main():
    files = sorted(glob.glob("runs/baseline_sweep/*/trajectory_*_seed42.npz")) \
        + sorted(glob.glob("runs/baseline/trajectory_*_seed42.npz")) \
        + sorted(glob.glob("runs/eval/ppo_trajectory_*_seed42.npz"))
    rows = []
    for f in files:
        if "fill_timestamps" not in np.load(f):
            print(f"skipping {f}: no fill arrays on this path "
                  "(only the F baseline logs fills; MarlChild does not yet)")
            continue
        rows.append(analyse(f))
    if not rows:
        sys.exit("no trajectories with fill data found")

    print(f"{'config / agent':<34} {'fills':>6} {'shares':>7} {'Δequity':>9} "
          f"{'capture':>9} {'c/share':>8} {'adv60s':>9} {'adv300s':>9} {'advEnd':>9} {'resid':>7}")
    print("-" * 128)
    for r in rows:
        print(f"{label(r['file']):<34} {r['n_fills']:>6} {r['shares']:>7} {r['d_equity']:>9,.0f} "
              f"{r['capture']:>9,.0f} {r['half_spread_c']:>8.2f} {r['adv_60']:>9,.0f} "
              f"{r['adv_300']:>9,.0f} {r['adv_end']:>9,.0f} {r['residual']:>7,.1f}")
    print("\n'resid' is Δequity - (capture + advEnd); it must be ~0 for the")
    print("decomposition to be exact. 'c/share' is realised half-spread in cents.")


if __name__ == "__main__":
    main()
