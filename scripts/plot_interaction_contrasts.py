"""Plot market-paired effects from reconciled interaction-analysis JSON.

Uses the existing per-market contrasts, including averaging action replicates
within each market. It does not pool markets across trained policies or invent
independent training replications from repeated evaluations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", help="one label per analyzed training run")
    args = parser.parse_args()
    results = json.loads(args.analysis.read_text())["results"]
    labels = args.labels or [
        Path(row["metadata"]["parent_run"]).name.replace("_", " ") for row in results
    ]
    if len(labels) != len(results):
        parser.error("Provide one label per analysis result")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for ax, agent, name, color in zip(
        axes, ("mm_0", "mm_1"), ("Trader A", "Trader B"),
        ("#167d8d", "#bf6541"), strict=True,
    ):
        for index, row in enumerate(results):
            effects = [
                e for e in row["paired_effects"]
                if e["agent"] == agent and e["metric"] == "pnl_dollars"
                and e["contrast"] == "sampled_pair minus marginal_b"
            ]
            if len(effects) != 1:
                raise ValueError(f"Expected one state-blind B contrast for {row['evaluation']}")
            effect = effects[0]
            values = np.array(list(effect["seed_differences"].values()), dtype=float)
            if not np.isfinite(values).all() or not len(values):
                raise ValueError("Contrast must contain finite market-level differences")
            if not np.isclose(values.mean(), effect["mean_difference"]):
                raise ValueError("Reported mean does not match market-level differences")
            positions = index + np.linspace(-0.12, 0.12, len(values))
            ax.scatter(positions, values, color=color, alpha=0.6, s=42, zorder=3)
            ax.plot([index - 0.22, index + 0.22], [values.mean()] * 2,
                    color=color, linewidth=4, zorder=4)
        ax.axhline(0, color="#555555", linewidth=1)
        ax.set_xticks(range(len(results)), labels)
        ax.set_xlim(-0.5, len(results) - 0.5)
        ax.set_title(name, fontsize=13)
        ax.set_ylabel("Difference in marked P&L ($)")
        ax.grid(axis="y", alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Learned B versus B's state-blind action mix\n"
        "Points: paired markets; bars: means; learned A held fixed within each comparison",
        fontsize=13,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out}; {len(results)} training-run comparisons")


if __name__ == "__main__":
    main()
