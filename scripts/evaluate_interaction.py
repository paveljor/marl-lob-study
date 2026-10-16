"""Evaluate learned pairs and interventions using disjoint calibration markets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from two_trader import evaluate, save_sources, write_json

from marl_lob.independent import ConstantPolicy
from marl_lob.policy_controls import GaussianPolicy, MarginalPolicy

SCENARIOS = ("sampled_pair", "mean_pair", "initial_pair", "initial_b", "marginal_b", "inactive_b")


def action_seed(market_seed, replicate, role):
    return int(np.random.SeedSequence([918273, market_seed, replicate, role]).generate_state(1)[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--market-seeds", nargs="+", type=int, default=list(range(30001, 30007)))
    parser.add_argument(
        "--calibration-seeds", nargs="+", type=int, default=list(range(20001, 20005))
    )
    parser.add_argument("--replicates", nargs="+", type=int, default=[0])
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--a-mode", choices=["sampled", "mean"], default="sampled")
    parser.add_argument("--checkpoint", type=int, help="paired checkpoint step instead of final")
    args = parser.parse_args()
    if set(args.market_seeds) & set(args.calibration_seeds):
        parser.error("Calibration and test markets must be disjoint")
    for values in (args.market_seeds, args.calibration_seeds, args.replicates, args.scenarios):
        if len(values) != len(set(values)):
            parser.error("Duplicate evaluation conditions are not allowed")
    parent = json.loads((args.run / "run_meta.json").read_text())
    config = parent["environment_config"]
    quote_size = parent["args"]["quote_size"]
    args.out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    paths = {}
    for agent, role in enumerate(("a", "b")):
        paths[f"initial_{role}"] = args.run / role / "initial.zip"
        paths[role] = (
            args.run / role / "policy.zip"
            if args.checkpoint is None
            else args.run / "checkpoints" / f"step{args.checkpoint}" / f"mm_{agent}.zip"
        )
    models = {role: PPO.load(path, device="cpu") for role, path in paths.items()}
    for model in models.values():
        if model.observation_space.shape != models["a"].observation_space.shape:
            raise ValueError("Checkpoint observation spaces differ")
    meta = {
        "status": "running",
        "parent_run": str(args.run.resolve()),
        "environment_config": config,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "checkpoint_sha256": {
            k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()
        },
        "action_rng": (
            "per-agent numpy default_rng; seed=SeedSequence([918273,market,replicate,role])"
        ),
        "marginal_control": (
            "IID complete rows from separate calibration markets; breaks state/time link"
        ),
        "norm_obs": False,
    }
    write_json(args.out_dir / "run_meta.json", meta)
    save_sources(args.out_dir)
    started = time.monotonic()

    def policy(role, seed, replicate, initial=False, mean=False):
        return GaussianPolicy(
            models[("initial_" if initial else "") + role],
            quote_size,
            action_seed(seed, replicate, int(role == "b")),
            deterministic=mean or (role == "a" and args.a_mode == "mean"),
        )

    pool = None
    if "marginal_b" in args.scenarios:
        pools = []
        for seed in args.calibration_seeds:
            directory = args.out_dir / "calibration" / f"seed{seed}"
            evaluate(
                config,
                {"mm_0": policy("a", seed, 0), "mm_1": policy("b", seed, 0)},
                seed,
                "calibration",
                directory,
            )
            with np.load(directory / "mm_1.npz") as data:
                pools.append(data["actions"])
            print(f"calibrated market {seed}", flush=True)
        pool = np.concatenate(pools)
        np.save(args.out_dir / "b_action_pool.npy", pool)

    completed = 0
    with (args.out_dir / "summary.csv").open("w") as stream:
        writer = None
        for seed in args.market_seeds:
            for replicate in args.replicates:
                for scenario in args.scenarios:
                    policies = {
                        "mm_0": policy(
                            "a",
                            seed,
                            replicate,
                            initial=scenario == "initial_pair",
                            mean=scenario == "mean_pair",
                        ),
                        "mm_1": policy(
                            "b",
                            seed,
                            replicate,
                            initial=scenario in ("initial_pair", "initial_b"),
                            mean=scenario == "mean_pair",
                        ),
                    }
                    if scenario == "inactive_b":
                        policies["mm_1"] = ConstantPolicy(quote_size=quote_size, active=False)
                    elif scenario == "marginal_b":
                        policies["mm_1"] = MarginalPolicy(pool, action_seed(seed, replicate, 1))
                    directory = args.out_dir / "eval" / scenario / f"seed{seed}_rep{replicate}"
                    rows = evaluate(config, policies, seed, scenario, directory)
                    for row in rows:
                        row["replicate"] = replicate
                        row["training_run"] = str(args.run)
                        row["checkpoint"] = args.checkpoint or "final"
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                        writer.writeheader()
                    writer.writerows(rows)
                    stream.flush()
                    completed += 1
                    print(
                        f"{seed}/{replicate} {scenario}: "
                        + ", ".join(
                            f"{r['agent']} ${r['pnl_dollars']:+.2f}, "
                            f"{r['any_quote_pct']:.1f}% quotes"
                            for r in rows
                        ),
                        flush=True,
                    )
                    meta.update(
                        completed_markets=completed, wall_seconds=time.monotonic() - started
                    )
                    write_json(args.out_dir / "run_meta.json", meta)
    meta.update(status="complete", wall_seconds=time.monotonic() - started)
    write_json(args.out_dir / "run_meta.json", meta)


if __name__ == "__main__":
    main()
