"""Summarize a completed independent-policy pilot without significance claims."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "initial_a_vs_script": "Initial A / Script",
    "trained_a_vs_script": "Trained A / Script",
    "trained_a_vs_initial_b": "Trained A / Initial B",
    "trained_a_vs_trained_b": "Trained A / Trained B",
    "script_vs_trained_b": "Script / Trained B",
    "inactive_a_vs_trained_b": "Inactive A / Trained B",
}
METRICS = ("pnl_dollars", "any_quote_pct", "two_sided_pct", "gross_fills",
           "gross_shares", "final_inventory", "max_abs_inventory",
           "direct_pair_matches", "direct_pair_shares")


def summarize(run):
    meta = json.loads((run / "run_meta.json").read_text())
    if meta["status"] != "complete":
        raise ValueError("Run must be complete before analysis")
    with (run / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    labels = dict(LABELS)
    sampling_path = run / "sampling_check" / "summary.csv"
    sampling_meta = None
    if sampling_path.exists():
        sampling_meta = json.loads((sampling_path.parent / "run_meta.json").read_text())
        if sampling_meta["status"] != "complete":
            raise ValueError("Sampling diagnostic is incomplete")
        with sampling_path.open() as stream:
            rows.extend(csv.DictReader(stream))
        labels["trained_a_vs_sampled_b"] = "Trained A / Sampled B†"
    summary = {}
    for scenario in labels:
        summary[scenario] = {}
        for agent in ("mm_0", "mm_1"):
            group = [r for r in rows if r["scenario"] == scenario and r["agent"] == agent]
            if not group:
                raise ValueError(f"Missing {scenario}/{agent}")
            summary[scenario][agent] = {
                key: {"mean": float(np.mean([float(r[key]) for r in group])),
                      "values_by_seed": {r["seed"]: float(r[key]) for r in group}}
                for key in METRICS
            }
            summary[scenario][agent]["risk_terminated_episodes"] = sum(
                r["risk_terminated"] == "True" for r in group)
            summary[scenario][agent]["steps_by_seed"] = {
                r["seed"]: int(r["steps"]) for r in group}
    contrasts = {}
    for name, agent, newer, reference in (
        ("a_training", "mm_0", "trained_a_vs_script", "initial_a_vs_script"),
        ("b_training_against_a", "mm_1", "trained_a_vs_trained_b", "trained_a_vs_initial_b"),
        ("a_change_when_b_trained", "mm_0", "trained_a_vs_trained_b", "trained_a_vs_initial_b"),
        ("b_with_trained_vs_inactive_a", "mm_1",
         "trained_a_vs_trained_b", "inactive_a_vs_trained_b"),
    ):
        contrasts[name] = {}
        for metric in ("pnl_dollars", "any_quote_pct", "gross_fills"):
            new = summary[newer][agent][metric]["values_by_seed"]
            old = summary[reference][agent][metric]["values_by_seed"]
            if new.keys() != old.keys():
                raise ValueError("Unpaired evaluation seeds")
            differences = {s: new[s] - old[s] for s in new}
            contrasts[name][metric] = {
                "mean_difference": float(np.mean(list(differences.values()))),
                "differences_by_seed": differences,
            }
    return {"run": str(run), "run_meta": meta, "labels": labels,
            "sampling_diagnostic_meta": sampling_meta, "scenarios": summary,
            "paired_descriptive_contrasts": contrasts}


def plot(summary, run, out):
    scenarios = summary["scenarios"]
    labels = dict(summary["labels"])
    for scenario in labels:
        if any(a["risk_terminated_episodes"] for a in scenarios[scenario].values()):
            labels[scenario] += "*"
    x = np.arange(len(labels))
    colors = {"mm_0": "#176b9b", "mm_1": "#bf651d"}
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), constrained_layout=True)
    for panel, metric, ylabel in ((axes[0], "pnl_dollars", "Marked P&L ($)"),
                                  (axes[1], "any_quote_pct", "Decisions with a quote (%)")):
        for agent, shift, name in (("mm_0", -.18, "Trader A slot"),
                                   ("mm_1", .18, "Trader B slot")):
            values = [scenarios[s][agent][metric]["mean"] for s in labels]
            panel.bar(x + shift, values, width=.32, color=colors[agent], alpha=.65, label=name)
            for i, s in enumerate(labels):
                raw = list(scenarios[s][agent][metric]["values_by_seed"].values())
                panel.scatter(i + shift + np.linspace(-.07, .07, len(raw)), raw,
                              color=colors[agent], edgecolors="white", linewidths=.5, zorder=3)
        panel.axhline(0, color="#666", linewidth=.8)
        panel.set_xticks(x, list(labels.values()), rotation=14, ha="right")
        panel.set_ylabel(ylabel)
        panel.spines[["top", "right"]].set_visible(False)
        panel.grid(axis="y", alpha=.15)
    axes[0].legend(frameon=False, ncols=2)
    axes[1].set_ylim(0, 105)
    for seed in summary["run_meta"]["args"]["eval_seeds"]:
        for agent, name, style in (("mm_0", "A", "--"), ("mm_1", "B", "-")):
            path = run / "eval" / "trained_a_vs_trained_b" / f"seed{seed}" / f"{agent}.npz"
            with np.load(path) as d:
                axes[2].plot(d["timestamps"], d["inventory"], style, color=colors[agent],
                             alpha=.65, label=f"{name}, market seed {seed}")
    axes[2].set(xlabel="Simulated seconds", ylabel="Inventory (shares)",
                title="Deterministic trained A vs trained B: inventory through each evaluation")
    axes[2].axhline(0, color="#666", linewidth=.8)
    axes[2].spines[["top", "right"]].set_visible(False)
    axes[2].legend(frameon=False, fontsize=8, ncols=3)
    axes[2].grid(alpha=.15)
    fig.suptitle("Two independent trading policies — exploratory pilot\n"
                 "Bars: mean across evaluation seeds; dots: individual markets. "
                 "One training seed per role.\n"
                 "* Includes an early inventory-limit stop. "
                 "† Post-pilot stochastic evaluation; same B checkpoint.", fontsize=12)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.run)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.out_prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(summary, args.run, args.out_prefix.with_suffix(".png"))
    print(json.dumps(summary["paired_descriptive_contrasts"], indent=2))


if __name__ == "__main__":
    main()
