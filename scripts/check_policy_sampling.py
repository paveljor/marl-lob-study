"""Post-pilot diagnostic: distinguish deterministic withdrawal from sampled behavior.

No training. Keeps A deterministic and samples B's existing Gaussian policy,
with a recorded action-sampling seed per held-out market. Also calculates
quoting probabilities on the saved deterministic trajectories.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from two_trader import evaluate, write_json

from marl_lob.independent import FrozenPPOPolicy, fixed_size_action


class SamplingPolicy(FrozenPPOPolicy):
    def act(self, observation):
        action, _ = self.model.predict(observation, deterministic=False)
        return fixed_size_action(action, self.quote_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    meta = json.loads((args.run / "run_meta.json").read_text())
    if meta["status"] != "complete":
        parser.error("The parent pilot must be complete")
    for role in ("a", "b"):
        policy_meta = json.loads((args.run / role / "run_meta.json").read_text())
        if policy_meta["norm_obs"]:
            parser.error("This diagnostic requires raw-observation policies")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "diagnostic_source.py").write_text(Path(__file__).read_text())
    torch.set_num_threads(1)
    a = PPO.load(args.run / "a/policy", device="cpu")
    b = PPO.load(args.run / "b/policy", device="cpu")
    quote_size = meta["args"]["quote_size"]
    diagnostics = []
    for scenario in ("trained_a_vs_trained_b", "script_vs_trained_b",
                     "inactive_a_vs_trained_b"):
        for seed in meta["args"]["eval_seeds"]:
            path = args.run / "eval" / scenario / f"seed{seed}" / "mm_1.npz"
            with np.load(path) as data:
                observations = data["observations"][:-1]
            with torch.no_grad():
                normal = b.policy.get_distribution(torch.as_tensor(observations)).distribution
                means = normal.mean[:, :2]
                p_off = normal.cdf(torch.zeros_like(normal.mean))[:, :2]
                p_any = 1 - p_off[:, 0] * p_off[:, 1]
            diagnostics.append({
                "scenario": scenario, "market_seed": seed,
                "mean_bid_gate": float(means[:, 0].mean()),
                "mean_ask_gate": float(means[:, 1].mean()),
                "gate_std": normal.stddev[0, :2].tolist(),
                "expected_any_quote_probability_on_saved_states": float(p_any.mean()),
            })
    write_json(args.out_dir / "saved_state_probabilities.json", diagnostics)
    policies = {"mm_0": FrozenPPOPolicy(a, quote_size), "mm_1": SamplingPolicy(b, quote_size)}
    rows = []
    for seed in meta["args"]["eval_seeds"]:
        torch.manual_seed(50000 + seed)
        result = evaluate(meta["environment_config"], policies, seed,
                          "trained_a_vs_sampled_b", args.out_dir / "eval" / f"seed{seed}")
        rows.extend(result)
        print(f"market seed {seed}: " + ", ".join(
            f"{r['agent']} ${r['pnl_dollars']:+.2f}, {r['gross_fills']} fills, "
            f"{r['any_quote_pct']:.1f}% quoting" for r in result), flush=True)
    with (args.out_dir / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.out_dir / "run_meta.json", {
        "status": "complete", "parent_run": str(args.run.resolve()),
        "environment_config": meta["environment_config"],
        "sampling_seeds": {s: 50000 + s for s in meta["args"]["eval_seeds"]},
        "a_deterministic": True, "b_deterministic": False,
        "purpose": "exploratory diagnostic prompted by deterministic withdrawal; no retraining",
        "diagnostic_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })


if __name__ == "__main__":
    main()
