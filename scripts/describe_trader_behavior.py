"""Describe quote direction, requested offsets, gross flows and inventory.

Reads complete saved evaluations; does not simulate new markets. Differences
between conditions are descriptive and do not isolate a causal mechanism.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from marl_lob.actions import resolve_offset_cents


def describe(evaluation: Path, scenarios: list[str]) -> list[dict]:
    meta = json.loads((evaluation / "run_meta.json").read_text())
    if meta["status"] != "complete":
        raise ValueError(f"Incomplete evaluation: {evaluation}")
    config = meta["environment_config"]
    if not config.get("gated_actions"):
        raise ValueError("This diagnostic requires gated actions")
    summary = pd.read_csv(evaluation / "summary.csv")
    if set(scenarios) - set(summary.scenario):
        raise ValueError(f"Requested scenarios absent from {evaluation}")
    rows = []
    for record in summary[summary.scenario.isin(scenarios)].itertuples():
        source = (evaluation / "eval" / record.scenario
                  / f"seed{record.seed}_rep{record.replicate}" / f"{record.agent}.npz")
        with np.load(source) as data:
            actions = data["actions"]
            inventory = data["inventory"].astype(np.int64)
            side, quantity = data["fill_side"], data["fill_quantity"]
            if actions.ndim != 2 or actions.shape[1] != 6:
                raise ValueError(f"Expected six-dimensional saved actions: {source}")
            if len(actions) != record.steps or len(inventory) != len(actions) + 1:
                raise ValueError(f"Decision/account lengths disagree: {source}")
            if not np.isfinite(actions).all():
                raise ValueError(f"Nonfinite action: {source}")
            buys = int(quantity[side > 0].sum())
            sells = int(quantity[side < 0].sum())
            if buys - sells != inventory[-1] - inventory[0]:
                raise ValueError(f"Signed gross flow does not reconcile: {source}")
            row = {
                "training_run": Path(meta["parent_run"]).name,
                "scenario": record.scenario, "seed": record.seed,
                "replicate": record.replicate, "agent": record.agent,
                "pnl_dollars": record.pnl_dollars,
                "risk_terminated": record.risk_terminated,
                "bought_shares": buys, "sold_shares": sells,
                "mean_inventory": float(inventory.mean()),
                "mean_abs_inventory": float(np.abs(inventory).mean()),
                "final_inventory": int(inventory[-1]),
                "source": str(source),
            }
            for column, name in enumerate(("bid", "ask")):
                active = actions[:, column] > 0
                offsets = [
                    resolve_offset_cents(
                        float(value), tick_size=config.get("tick_size", 1),
                        min_offset_ticks=config.get("min_offset_ticks", 1),
                        offset_span=config.get("offset_span"),
                    )
                    for value in actions[active, column + 2]
                ]
                row[f"{name}_gate_pct"] = float(active.mean() * 100)
                row[f"{name}_active_offset_cents"] = (
                    float(np.mean(offsets)) if offsets else float("nan")
                )
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluations", nargs="+", type=Path)
    parser.add_argument("--scenarios", nargs="+", default=["sampled_pair", "initial_pair"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [row for path in args.evaluations for row in describe(path, args.scenarios)]
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    columns = [
        "bid_gate_pct", "ask_gate_pct", "bid_active_offset_cents", "ask_active_offset_cents",
        "bought_shares", "sold_shares", "mean_inventory", "mean_abs_inventory",
        "final_inventory", "pnl_dollars",
    ]
    print(frame.groupby(["training_run", "scenario", "agent"])[columns].mean().round(2)
          .to_string())
    print("Offsets describe requested active-side actions, before half-tick outward rounding.")
    print("Group offsets are means of episode means; missing inactive sides are omitted.")
    print("Gate choices are not exchange acknowledgments. Flow totals reconcile to inventory.")


if __name__ == "__main__":
    main()
