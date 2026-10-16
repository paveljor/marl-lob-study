"""Train two independent PPO networks from shared market rollouts."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
from two_trader import policy_digest, save_sources, write_json

from marl_lob.concurrent import ConcurrentPPOTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--config-from", type=Path, default=Path("experiments/concurrent_base.json")
    )
    parser.add_argument("--steps", type=int, default=32768, help="samples per trader")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--quote-size", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=8192)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument(
        "--legacy-close-bootstrap",
        action="store_true",
        help="Reproduce the old artificial continuation value at exchange close",
    )
    parser.add_argument(
        "--episode-seconds", type=int, help="override session length for smoke runs"
    )
    args = parser.parse_args()
    if min(args.steps, args.save_every, args.n_steps) < 1:
        parser.error("Steps and checkpoint interval must be positive")
    if args.save_every % args.n_steps:
        parser.error("save-every must be a multiple of n-steps")
    config = json.loads(args.config_from.read_text())["environment_config"]
    if args.episode_seconds is not None:
        from datetime import datetime, timedelta

        start = datetime.strptime(config["start_time"], "%H:%M:%S")
        if not 1 <= args.episode_seconds <= 23400:
            parser.error("episode-seconds must be in 1..23400")
        config["end_time"] = (start + timedelta(seconds=args.episode_seconds)).strftime("%H:%M:%S")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    meta = {
        "status": "running",
        "environment_config": config,
        "norm_obs": False,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "platform": platform.platform(),
        "algorithm": "concurrent independent PPO; shared market, separate buffers and optimizers",
        "policy_seeds": {"mm_0": args.seed, "mm_1": args.seed + 1},
        "market_close_bootstrap": args.legacy_close_bootstrap,
        "limitations": [
            "mark-to-mid PnL, no fees or liquidation",
            "legacy-close-bootstrap, if enabled, changes the finite-session learning objective",
            "snapshots preserve policies/optimizers/stats, not live ABIDES kernel state",
        ],
    }
    write_json(args.out_dir / "run_meta.json", meta)
    save_sources(args.out_dir)
    protocol = Path("notes/concurrent_protocol.md")
    (args.out_dir / "protocol.md").write_text(protocol.read_text())
    started = time.monotonic()
    trainer = ConcurrentPPOTrainer(
        config,
        args.quote_size,
        args.seed,
        n_steps=args.n_steps,
        batch_size=min(64, args.n_steps),
        bootstrap_at_market_close=args.legacy_close_bootstrap,
        device="cpu",
    )
    for agent, role in zip(trainer.agents, ("a", "b"), strict=True):
        directory = args.out_dir / role
        directory.mkdir()
        trainer.models[agent].save(directory / "initial")
    meta["initial_policy_sha256"] = {
        agent: policy_digest(model) for agent, model in trainer.models.items()
    }
    last_report = started

    def checkpoint(current, step):
        nonlocal last_report
        final = step >= args.steps
        if step % args.save_every == 0 or final:
            directory = args.out_dir / "checkpoints" / f"step{step}"
            directory.mkdir(parents=True)
            for agent, model in current.models.items():
                model.save(directory / agent)
            write_json(directory / "normalizers.json", current.normalizer_state())
        # Save run evidence after every completed optimizer update.
        meta.update(
            actual_timesteps=step,
            episode_seeds=current.episode_seeds,
            wall_seconds=time.monotonic() - started,
        )
        write_json(args.out_dir / "run_meta.json", meta)
        write_json(args.out_dir / "training_diagnostics.json", current.diagnostics)
        if time.monotonic() - last_report >= 30 or final:
            print(
                f"seed {args.seed}: {step} shared steps; {len(current.episode_seeds)} episodes; "
                f"{time.monotonic() - started:.1f}s",
                flush=True,
            )
            last_report = time.monotonic()

    try:
        trainer.train(args.steps, checkpoint)
        for agent, role in zip(trainer.agents, ("a", "b"), strict=True):
            trainer.models[agent].save(args.out_dir / role / "policy")
        meta.update(
            status="complete",
            final_policy_sha256={
                agent: policy_digest(model) for agent, model in trainer.models.items()
            },
        )
        if any(
            meta["initial_policy_sha256"][a] == meta["final_policy_sha256"][a]
            for a in trainer.agents
        ):
            raise RuntimeError("At least one learner did not change weights")
        write_json(args.out_dir / "normalizers.json", trainer.normalizer_state())
    except BaseException as error:
        meta.update(status="failed", error=repr(error))
        raise
    finally:
        meta["wall_seconds"] = time.monotonic() - started
        write_json(args.out_dir / "run_meta.json", meta)
        trainer.close()


if __name__ == "__main__":
    main()
