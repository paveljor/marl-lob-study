"""Measure how a frozen Gaussian policy changes across observed market states.

These are policy-distribution diagnostics, not evidence of profitable
adaptation. They reuse saved observations and execute no new market trades.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from two_trader import write_json


def distribution(model, observations):
    with torch.no_grad():
        tensor, _ = model.policy.obs_to_tensor(observations)
        normal = model.policy.get_distribution(tensor).distribution
        p_off = normal.cdf(torch.zeros_like(normal.mean)).cpu().numpy()[:, :2]
        return (
            normal.mean.cpu().numpy(),
            normal.stddev.cpu().numpy(),
            np.column_stack([1 - p_off, 1 - np.prod(p_off, axis=1)]),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    meta = json.loads((args.evaluation / "run_meta.json").read_text())
    parent = Path(meta["parent_run"])
    checkpoint = meta["args"].get("checkpoint")
    result = {"evaluation": str(args.evaluation), "roles": {}}
    torch.set_num_threads(1)
    for index, role in enumerate(("a", "b")):
        agent = f"mm_{index}"
        path = (
            parent / role / "policy.zip"
            if checkpoint is None
            else parent / "checkpoints" / f"step{checkpoint}" / f"{agent}.zip"
        )
        model = PPO.load(path, device="cpu")
        states = []
        for source in sorted((args.evaluation / "eval" / "sampled_pair").glob(f"*/{agent}.npz")):
            with np.load(source) as data:
                states.append(data["observations"][:-1])
        observations = np.concatenate(states)
        mean, std, probability = distribution(model, observations)
        probabilities = {}
        for column, name in enumerate(("bid", "ask", "either")):
            values = probability[:, column]
            probabilities[name] = {
                "mean": float(values.mean()),
                "std_over_states": float(values.std()),
                "quantiles_0_5_50_95_100": np.quantile(values, [0, 0.05, 0.5, 0.95, 1]).tolist(),
            }
        # Direct policy interventions on individual observation fields. These
        # can be off-distribution and are not market outcome counterfactuals.
        base = 4 * meta["environment_config"]["k"]
        interventions = {}
        for name, columns in (("inventory_zero", [base]), ("ofi_zero", [base + 4, base + 5])):
            if name == "ofi_zero" and not meta["environment_config"].get("include_ofi"):
                continue
            counterfactual = observations.copy()
            counterfactual[:, columns] = 0
            _, _, alternative = distribution(model, counterfactual)
            interventions[name] = {
                "mean_abs_change_bid_ask_either_probability": np.abs(alternative - probability)
                .mean(axis=0)
                .tolist(),
                "max_abs_change_bid_ask_either_probability": np.abs(alternative - probability)
                .max(axis=0)
                .tolist(),
            }
        result["roles"][agent] = {
            "observed_states": len(observations),
            "quote_probabilities": probabilities,
            "raw_action_mean_average": mean.mean(axis=0).tolist(),
            "raw_action_mean_std_over_states": mean.std(axis=0).tolist(),
            "gaussian_sampling_std": std[0].tolist(),
            "deterministic_any_quote_fraction": float((mean[:, :2] > 0).any(axis=1).mean()),
            "observation_interventions": interventions,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
