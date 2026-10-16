"""Plot participation and share flows from describe_trader_behavior.py output."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("behavior", type=Path)
    parser.add_argument("--scenario", default="sampled_pair")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.behavior)
    frame = frame[frame.scenario == args.scenario]
    if frame.empty:
        parser.error("No rows for the requested scenario")
    metrics = ["bid_gate_pct", "ask_gate_pct", "bought_shares", "sold_shares"]
    if not np.isfinite(frame[metrics].to_numpy()).all():
        raise ValueError("Participation and share flows must be finite")
    # Each dot is one market, even when multiple action draws were evaluated.
    markets = frame.groupby(["training_run", "agent", "seed"])[metrics].mean()
    pairs = list(markets.index.droplevel("seed").unique())
    labels = [f"{run.removeprefix('finite_seed')} / {'A' if agent == 'mm_0' else 'B'}"
              for run, agent in pairs]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    panels = [
        (("bid_gate_pct", "Bid enabled", "#167d8d"),
         ("ask_gate_pct", "Ask enabled", "#bf6541")),
        (("bought_shares", "Shares bought", "#167d8d"),
         ("sold_shares", "Shares sold", "#bf6541")),
    ]
    for ax, panel in zip(axes, panels, strict=True):
        for metric, legend, color in panel:
            shift = -0.18 if metric in ("bid_gate_pct", "bought_shares") else 0.18
            for index, pair in enumerate(pairs):
                values = markets.loc[pair, metric].to_numpy()
                ax.bar(index + shift, values.mean(), width=0.32, color=color,
                       alpha=0.8, label=legend if index == 0 else None)
                ax.scatter(index + shift + np.linspace(-0.09, 0.09, len(values)),
                           values, color="#222222", alpha=0.4, s=12, zorder=3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="upper right")
    axes[0].set_ylabel("Decisions enabling a side (%)")
    axes[0].set_ylim(0, 105)
    axes[1].set_ylabel("Shares per session")
    axes[1].set_xticks(range(len(pairs)), labels)
    axes[1].set_xlabel("Training seed / trader")
    fig.suptitle(
        "Quote activity does not tell the inventory story\n"
        f"{args.scenario.replace('_', ' ').capitalize()}; bars: means; dots: markets "
        "(action draws averaged)", fontsize=13,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out}; {len(pairs)} trader/run groups")


if __name__ == "__main__":
    main()
