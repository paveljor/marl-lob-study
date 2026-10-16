"""Post-hoc critic values at recorded closes; not original training bonuses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from two_trader import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    meta = json.loads((args.evaluation / "run_meta.json").read_text())
    if meta["status"] != "complete" or meta["args"].get("checkpoint") is not None:
        parser.error("Requires a complete final-checkpoint evaluation")
    parent = Path(meta["parent_run"])
    normalizers = json.loads((parent / "normalizers.json").read_text())
    rows = []
    torch.set_num_threads(1)
    for index, role in enumerate(("a", "b")):
        agent = f"mm_{index}"
        model = PPO.load(parent / role / "policy.zip", device="cpu")
        for path in sorted((args.evaluation / "eval" / "sampled_pair").glob(f"*/{agent}.npz")):
            market = json.loads((path.parent / "market.json").read_text())
            if market["risk_terminated"]:
                continue
            with np.load(path) as data:
                observation = data["observations"][-1]
                timestamp = float(data["timestamps"][-1])
            with torch.no_grad():
                tensor, _ = model.policy.obs_to_tensor(observation)
                value = float(model.policy.predict_values(tensor).item())
            rows.append(
                {
                    "agent": agent,
                    "source": str(path),
                    "timestamp_seconds": timestamp,
                    "critic_value_normalized": value,
                    "hypothetical_legacy_bonus_normalized": model.gamma * value,
                    "approx_dollars_at_final_reward_scale": (
                        model.gamma * value * np.sqrt(normalizers[agent]["var"]) / 100
                    ),
                }
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out,
        {
            "evaluation": str(args.evaluation),
            "rows": rows,
            "limitation": "Final frozen critic on held-out close states, not bonuses logged during "
            "training. Dollar conversion uses final reward variance only; no cash "
            "charge or additional market P&L occurred. Correct continuation at the "
            "task's actual close is zero.",
        },
    )
    print(f"Saved {len(rows)} close-state diagnostics to {args.out}")


if __name__ == "__main__":
    main()
