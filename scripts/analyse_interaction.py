"""Summarize interaction controls, paired differences, and exact account decomposition."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from two_trader import write_json


def enrich(directory):
    frame = pd.read_csv(directory / "summary.csv")
    # Secondary sensitivity: compare all conditions within a market/action
    # replicate at their shortest observed duration, before any early risk stop.
    # Full episode P&L and risk outcomes remain the primary recorded results.
    frame["common_horizon_steps"] = frame.groupby(["seed", "replicate"])["steps"].transform("min")
    for idx, row in frame.iterrows():
        path = directory / "eval" / row.scenario / f"seed{row.seed}_rep{row.replicate}"
        with np.load(path / f"{row.agent}.npz") as data:
            inventory, cash, mid = (
                data[name].astype(np.int64) for name in ("inventory", "cash", "mid_price")
            )
            exposure = int(np.sum(inventory[:-1] * np.diff(mid)))
            execution = int(np.sum(np.diff(cash) + np.diff(inventory) * mid[1:]))
            total = int(cash[-1] + inventory[-1] * mid[-1] - cash[0] - inventory[0] * mid[0])
            if exposure + execution != total or not np.isclose(total / 100, row.pnl_dollars):
                raise ValueError(f"P&L decomposition does not reconcile: {path}")
            common_step = int(row.common_horizon_steps)
            frame.loc[idx, "common_horizon_pnl_dollars"] = (
                float(
                    cash[common_step]
                    + inventory[common_step] * mid[common_step]
                    - cash[0]
                    - inventory[0] * mid[0]
                )
                / 100
            )
            signed_fills = data["fill_side"] * data["fill_quantity"]
            if int(signed_fills.sum()) != int(inventory[-1] - inventory[0]):
                raise ValueError(f"Fill inventory mismatch: {path}")
            if int(-(signed_fills * data["fill_price"]).sum()) != int(cash[-1] - cash[0]):
                raise ValueError(f"Fill cash mismatch: {path}")
            frame.loc[idx, "inventory_marking_dollars"] = exposure / 100
            frame.loc[idx, "execution_vs_step_end_mid_dollars"] = execution / 100
            frame.loc[idx, "mean_abs_inventory"] = float(np.abs(inventory).mean())
            frame.loc[idx, "duration_seconds"] = float(
                data["timestamps"][-1] - data["timestamps"][0]
            )
    frame["evaluation_dir"] = str(directory)
    return frame


def paired_effect(frame, control, metric, agent):
    sub = frame[frame.agent == agent]
    by_market = sub.groupby(["scenario", "seed"])[metric].mean().unstack("scenario")
    if control not in by_market or "sampled_pair" not in by_market:
        return None
    differences = (by_market.sampled_pair - by_market[control]).dropna()
    if not len(differences):
        return None
    rng = np.random.default_rng(12345)
    bootstrap = rng.choice(differences.to_numpy(), (10000, len(differences))).mean(axis=1)
    replicate_effects = []
    for replicate, group in sub.groupby("replicate"):
        values = group.groupby(["scenario", "seed"])[metric].mean().unstack("scenario")
        if control not in values or "sampled_pair" not in values:
            continue
        delta = (values.sampled_pair - values[control]).dropna()
        if not len(delta):
            continue
        replicate_effects.append({
            "replicate": int(replicate),
            "n_market_seeds": len(delta),
            "mean_difference": float(delta.mean()),
            "positive_markets": int((delta > 0).sum()),
            "seed_differences": {int(k): float(v) for k, v in delta.items()},
        })
    return {
        "contrast": f"sampled_pair minus {control}",
        "agent": agent,
        "metric": metric,
        "n_market_seeds": len(differences),
        "mean_difference": float(differences.mean()),
        "median_difference": float(differences.median()),
        "positive_markets": int((differences > 0).sum()),
        "paired_market_bootstrap_95pct": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "seed_differences": {int(k): float(v) for k, v in differences.items()},
        "action_replicate_effects": replicate_effects,
        "scope": "conditional on this trained pair; action replicates averaged within market",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluations", nargs="+", type=Path)
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    frames, results = [], []
    metrics = [
        "common_horizon_pnl_dollars",
        "pnl_dollars",
        "any_quote_pct",
        "gross_fills",
        "final_inventory",
        "max_abs_inventory",
        "mean_abs_inventory",
        "duration_seconds",
        "inventory_marking_dollars",
        "execution_vs_step_end_mid_dollars",
    ]
    for directory in args.evaluations:
        meta = json.loads((directory / "run_meta.json").read_text())
        if meta["status"] != "complete":
            raise ValueError(f"Incomplete evaluation: {directory}")
        frame = enrich(directory)
        frames.append(frame)
        grouped = frame.groupby(["scenario", "agent"])
        summaries = []
        for (scenario, agent), group in grouped:
            summary = {
                "scenario": scenario,
                "agent": agent,
                "episodes": len(group),
                "risk_stops": int(group.risk_terminated.sum()),
                "direct_pair_matches": int(group.direct_pair_matches.sum()),
            }
            summary.update({metric: float(group[metric].mean()) for metric in metrics})
            summaries.append(summary)
        effects = [
            effect
            for control in frame.scenario.unique()
            if control != "sampled_pair"
            for agent in ("mm_0", "mm_1")
            for metric in ("pnl_dollars", "any_quote_pct", "gross_fills")
            if (effect := paired_effect(frame, control, metric, agent)) is not None
        ]
        results.append(
            {
                "evaluation": str(directory),
                "metadata": meta,
                "input_csv_sha256": hashlib.sha256(
                    (directory / "summary.csv").read_bytes()
                ).hexdigest(),
                "summary": summaries,
                "paired_effects": effects,
            }
        )
        print(f"\n{directory}")
        print(
            grouped[
                [
                    "pnl_dollars",
                    "any_quote_pct",
                    "inventory_marking_dollars",
                    "execution_vs_step_end_mid_dollars",
                ]
            ]
            .mean()
            .round(2)
            .to_string()
        )
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(args.out_prefix.with_suffix(".csv"), index=False)
    write_json(
        args.out_prefix.with_suffix(".json"),
        {
            "results": results,
            "reconciled_agent_episodes": len(combined),
            "limitations": [
                "Exploratory intervals, no multiple-comparison correction",
                "Intervals condition on each trained pair, not training uncertainty",
                "Execution component uses next recorded midpoint, not spread profit",
                "P&L marks inventory; no modeled fees or terminal liquidation",
            ],
        },
    )
    fig, axes = plt.subplots(len(frames), 2, figsize=(12, max(4, 3.5 * len(frames))), squeeze=False)
    for index, frame in enumerate(frames):
        scenarios = list(frame.scenario.unique())
        for col, (metric, label) in enumerate(
            (("pnl_dollars", "Marked P&L ($)"), ("any_quote_pct", "Decisions with a quote (%)"))
        ):
            ax = axes[index, col]
            for agent, shift, color in (("mm_0", -0.12, "#167d8d"), ("mm_1", 0.12, "#bf6541")):
                sub = frame[frame.agent == agent]
                for n, scenario in enumerate(scenarios):
                    values = sub[sub.scenario == scenario].groupby("seed")[metric].mean()
                    ax.scatter(
                        np.full(len(values), n + shift),
                        values,
                        color=color,
                        alpha=0.45,
                        s=18,
                        label=agent if n == 0 else None,
                    )
                    ax.plot(
                        [n + shift - 0.08, n + shift + 0.08],
                        [values.mean()] * 2,
                        color=color,
                        linewidth=3,
                    )
            ax.axhline(0, color="grey", linewidth=0.6)
            ax.set_xticks(
                range(len(scenarios)),
                [s.replace("_", " ") for s in scenarios],
                rotation=25,
                ha="right",
            )
            ax.set_ylabel(label)
            ax.set_title(str(args.evaluations[index].name))
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False)
    fig.suptitle(
        "Two independent traders: learned behavior and controls\n"
        "Dots: market seeds (action repeats averaged); bars: means",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(args.out_prefix.with_suffix(".png"), dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
