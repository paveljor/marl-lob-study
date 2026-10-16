#!/usr/bin/env python3
"""Measure how much the parameter-sharing agents actually differ.

SuperSuit presents the N PettingZoo agents to SB3 as N parallel copies of a
single-agent env, so one ``MlpPolicy`` is trained on the pooled experience and
every agent evaluates the *same weights*. Whether that makes them behave as
distinct competitors is an empirical question: 40 of the 44 observation
features are the shared order book, and only ``inventory_norm`` and
``cash_norm`` can differentiate one agent from another.

That matters for any claim resting on ``n_agents``. If the agents emit
identical actions, varying ``n_agents`` varies the number of *copies* of one
policy in the book, not the amount of competition between market makers.

This script quantifies it three ways, all pairwise between agents:

* **raw actions** — correlation and difference of the continuous action
  vectors (either layout; the width selects the column names);
* **effective quotes** — the same comparison *after* ``translate_action``'s
  quantisation, which is what the exchange actually sees;
* **observations** — which of the 44 features ever diverge, and by how much.

    python scripts/agent_divergence.py runs/divergence/vecnorm_only/eval
    python scripts/agent_divergence.py runs/retrain/min_size_10/seed0/eval
"""
from __future__ import annotations

import argparse
import glob
import itertools
import re
from pathlib import Path

import numpy as np

from marl_lob.quote_analysis import QuoteMode, action_names, effective_quotes

# ── pure helpers ────────────────────────────────────────────────────────────

def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation, returning nan when either side is constant.

    A frozen policy emits a constant action, and ``np.corrcoef`` both warns
    and returns nan there. Constant-vs-constant is the case we most want to
    report rather than crash on, so it is handled explicitly.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.std() == 0.0 or y.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def pairwise_report(a: np.ndarray, b: np.ndarray, names=None) -> dict:
    """Compare two (n, k) arrays column by column.

    ``identical_pct`` is the headline: the share of steps on which the two
    agents emitted bit-identical vectors. ``rel_diff`` scales the mean
    absolute difference by the pooled std, so a value near 0 means the agents
    differ by far less than either one varies over time.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if names is None:
        names = action_names(a.shape[1])

    per_dim = {}
    for i, name in enumerate(names):
        pooled = np.concatenate([a[:, i], b[:, i]]).std()
        diff = float(np.abs(a[:, i] - b[:, i]).mean())
        per_dim[name] = {
            "corr": safe_corr(a[:, i], b[:, i]),
            "mean_abs_diff": diff,
            "rel_diff": float(diff / pooled) if pooled > 0 else 0.0,
            "mean_a": float(a[:, i].mean()),
            "mean_b": float(b[:, i].mean()),
        }
    return {
        "n_steps": int(n),
        "identical_pct": float((a == b).all(axis=1).mean() * 100),
        "per_dim": per_dim,
    }


def obs_divergence(o_a: np.ndarray, o_b: np.ndarray, names: list[str]) -> list[tuple]:
    """Return (name, max_abs_diff, std_a) for observation dims that ever differ."""
    o_a = np.asarray(o_a, dtype=float)
    o_b = np.asarray(o_b, dtype=float)
    n = min(len(o_a), len(o_b))
    d = np.abs(o_a[:n] - o_b[:n]).max(axis=0)
    return [(names[i], float(d[i]), float(o_a[:n, i].std()))
            for i in np.where(d > 0)[0]]


def group_by_seed(paths: list[str]) -> dict[str, dict[int, str]]:
    """Bucket eval trajectory files by seed, then by agent index.

    Filenames look like ``ppo_trajectory_<agent>_seed<seed>.npz``; agents are
    only comparable within the same eval seed, since a different seed is a
    different market.
    """
    out: dict[str, dict[int, str]] = {}
    for p in paths:
        m = re.search(r"trajectory_(\d+)_seed(\w+)\.npz$", Path(p).name)
        if m:
            out.setdefault(m.group(2), {})[int(m.group(1))] = p
    return out


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_dir", type=Path,
                    help="directory of eval trajectories (…/eval)")
    ap.add_argument("--max-size", type=int, default=100)
    args = ap.parse_args()

    paths = sorted(glob.glob(str(args.eval_dir / "*.npz")))
    if not paths:
        raise SystemExit(f"no .npz under {args.eval_dir}")

    from marl_lob.observation_extractor import describe_obs

    for seed, by_agent in sorted(group_by_seed(paths).items()):
        if len(by_agent) < 2:
            print(f"seed {seed}: only {len(by_agent)} agent(s), nothing to compare")
            continue
        print(f"\n{'='*72}\neval seed {seed} — {len(by_agent)} agents\n{'='*72}")
        data = {i: np.load(p) for i, p in sorted(by_agent.items())}
        # Replay under the semantics the run used. Under the legacy mapping a
        # span-trained policy's offsets all sit in [0, 1] and would every one
        # of them quantise to a single tick, which would report two agents as
        # quoting identically whatever they actually did.
        mode = {i: QuoteMode.for_trajectory(p) for i, p in sorted(by_agent.items())}

        for i, j in itertools.combinations(sorted(data), 2):
            da, db = data[i], data[j]
            if "actions" not in da or "actions" not in db:
                print(f"  agents {i}/{j}: no saved actions in this trajectory")
            else:
                for label, xa, xb in (
                    ("raw actions", da["actions"], db["actions"]),
                    ("effective quotes",
                     effective_quotes(da["actions"], args.max_size,
                                      **mode[i].kwargs()),
                     effective_quotes(db["actions"], args.max_size,
                                      **mode[j].kwargs())),
                ):
                    r = pairwise_report(xa, xb)
                    print(f"\n  [{label}] agents {i} vs {j} "
                          f"({r['n_steps']} steps)")
                    print(f"    identical on {r['identical_pct']:.2f}% of steps")
                    print(f"    {'field':<12}{'corr':>10}{'mean|Δ|':>12}"
                          f"{'Δ/std':>10}{'mean '+str(i):>10}{'mean '+str(j):>10}")
                    for name, m in r["per_dim"].items():
                        print(f"    {name:<12}{m['corr']:>10.5f}"
                              f"{m['mean_abs_diff']:>12.5f}{m['rel_diff']:>10.4f}"
                              f"{m['mean_a']:>10.4f}{m['mean_b']:>10.4f}")

            if "observations" in da and "observations" in db:
                # The identity one-hot widens the observation, so k cannot be
                # inferred from the width alone. eval.py records the suffix
                # width; files written before it did have no suffix.
                id_width = int(da["agent_id_width"]) if "agent_id_width" in da else 0
                k = (da["observations"].shape[1] - 4 - id_width) // 4
                diverging = obs_divergence(da["observations"],
                                           db["observations"],
                                           describe_obs(k, id_width))
                total = da["observations"].shape[1]
                print(f"\n  [observations] {len(diverging)}/{total} features "
                      f"ever differ between agents {i} and {j}")
                for name, mx, sd in diverging:
                    print(f"    {name:<24} max|Δ|={mx:.6f}   std={sd:.6f}")

            inv_a = da["inventory"].astype(float)
            inv_b = db["inventory"].astype(float)
            n = min(len(inv_a), len(inv_b))
            print(f"\n  [inventory] corr={safe_corr(inv_a[:n], inv_b[:n]):.5f}  "
                  f"final={inv_a[-1]:.0f}/{inv_b[-1]:.0f}  "
                  f"mean|Δ|={np.abs(inv_a[:n]-inv_b[:n]).mean():.2f}")
            fills_a = len(da["fill_timestamps"]) if "fill_timestamps" in da else 0
            fills_b = len(db["fill_timestamps"]) if "fill_timestamps" in db else 0
            print(f"  [fills] {fills_a} / {fills_b}")


if __name__ == "__main__":
    main()
