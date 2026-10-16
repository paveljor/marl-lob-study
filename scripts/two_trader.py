"""Small sequential-learning experiment with two independently owned PPO models.

Train A vs a scripted quoter, freeze A, train B vs A, then evaluate fixed
matchups on held-out market seeds. This is a pilot, not an exploitation test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
import platform
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from marl_lob.env import MarlLobEnv
from marl_lob.independent import ConstantPolicy, FrozenPPOPolicy, SingleLearnerEnv
from marl_lob.quote_analysis import participation_stats
from marl_lob.trajectory import Fill, Trajectory, save_trajectory


def write_json(path, data):
    def numpy_scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Cannot serialize {type(value).__name__}")
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False,
                                    default=numpy_scalar) + "\n")


def policy_digest(model):
    digest = hashlib.sha256()
    for key, tensor in sorted(model.policy.state_dict().items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def train(config, learner, opponent, quote_size, seed, steps, directory):
    directory.mkdir()
    base = SingleLearnerEnv(config, learner, opponent, quote_size, seed)
    env = VecNormalize(VecMonitor(DummyVecEnv([lambda: base])),
                       norm_obs=False, norm_reward=True, clip_reward=10)
    model = PPO("MlpPolicy", env, seed=seed, device="cpu", n_steps=256,
                batch_size=64, verbose=0)
    model.set_logger(configure(str(directory), ["csv"]))
    model.save(directory / "initial")
    initial_digest = policy_digest(model)
    started = time.monotonic()
    print(f"training {learner}: {steps} requested steps", flush=True)
    last_report = started

    def progress(locals_, globals_):
        nonlocal last_report
        if time.monotonic() - last_report >= 30:
            print(f"  {learner}: {model.num_timesteps} steps, "
                  f"{time.monotonic() - started:.0f}s", flush=True)
            last_report = time.monotonic()
        return True

    try:
        model.learn(total_timesteps=steps, callback=progress)
        model.save(directory / "policy")
        env.save(str(directory / "vecnormalize.pkl"))
        meta = {
            "environment_config": config,
            "algorithm": "PPO, independent optimizer, frozen opponent",
            "learner": learner, "quote_size": quote_size,
            "policy_action_layout": "bid_gate,ask_gate,bid_offset,ask_offset",
            "norm_obs": False, "seed": seed, "episode_seeds": base.episode_seeds,
            "requested_timesteps": steps, "actual_timesteps": model.num_timesteps,
            "initial_policy_sha256": initial_digest,
            "final_policy_sha256": policy_digest(model),
            "train_seconds": time.monotonic() - started,
        }
        write_json(directory / "run_meta.json", meta)
    finally:
        env.close()
    print(f"finished {learner}: {meta['train_seconds']:.1f}s", flush=True)
    # Reload inference models from disk to exercise the actual checkpoint path.
    return (FrozenPPOPolicy(PPO.load(directory / "policy", device="cpu"), quote_size),
            FrozenPPOPolicy(PPO.load(directory / "initial", device="cpu"), quote_size))


def evaluate(config, policies, seed, scenario, directory):
    env = MarlLobEnv(**config)
    try:
        obs, infos = env.reset(seed=seed)
        agents = env.possible_agents
        rows = {a: [infos[a]["traj_row"]] for a in agents}
        observations = {a: [obs[a]] for a in agents}
        actions = {a: [] for a in agents}
        executions = {a: [] for a in agents}
        submissions = {a: [] for a in agents}
        risk_terminated = False
        while env.agents:
            acts = {a: policies[a].act(obs[a]) for a in env.agents}
            for a, action in acts.items():
                actions[a].append(action)
            obs, _, terms, truncs, infos = env.step(acts)
            for a in agents:
                rows[a].append(infos[a]["traj_row"])
                observations[a].append(obs[a])
                executions[a].extend(infos[a]["executions"])
                submissions[a].extend(infos[a]["submissions"])
            risk_terminated = any(terms.values())
            if risk_terminated or any(truncs.values()):
                break

        # This is analyst-only data, never exposed to either policy.
        ids = {child.id: name for child, name in zip(env._children, agents, strict=True)}
        exchange = env.kernel.agents[0]
        matches = [dict(e) for e in exchange.order_books[env.symbol].history
                   if e["type"] == "EXEC"
                   and (e["agent_id"] in ids or e["oppos_agent_id"] in ids)]
        direct = [e for e in matches if e["agent_id"] in ids
                  and e["oppos_agent_id"] in ids
                  and e["agent_id"] != e["oppos_agent_id"]]
        directory.mkdir(parents=True)
        write_json(directory / "market.json", {
            "seed": seed, "environment_config": config, "agent_ids": ids,
            "risk_terminated": risk_terminated,
            "exchange_matches": matches,
            "executions": executions, "submissions": submissions,
        })
        summaries = []
        for a in agents:
            trajectory = Trajectory.from_tuples(rows[a])
            # Replace net-per-step fills with gross executions received by the
            # trader. Keep arrival time explicit; exchange match times are above.
            fills = [Fill((e["received_time_ns"] - env._mkt_open_ns) / 1e9,
                          1 if e["signed_quantity"] > 0 else -1,
                          e["price_cents"], abs(e["signed_quantity"]))
                     for e in executions[a]]
            trajectory = Trajectory(trajectory.timestamps, trajectory.inventory,
                                    trajectory.cash, trajectory.mid_price, fills)
            acts = np.asarray(actions[a], dtype=np.float32)
            save_trajectory(directory / f"{a}.npz", trajectory, actions=acts,
                            observations=np.asarray(observations[a], dtype=np.float32))
            stats = participation_stats(
                acts, max_size=config["max_size"],
                offset_span=config["offset_span"], mid_rounding=config["mid_rounding"])
            eq = trajectory.equity()
            inv_change = int(trajectory.inventory[-1] - trajectory.inventory[0])
            fill_inv = sum(e["signed_quantity"] for e in executions[a])
            cash_change = int(trajectory.cash[-1] - trajectory.cash[0])
            fill_cash = -sum(e["signed_quantity"] * e["price_cents"]
                             for e in executions[a])
            if (inv_change, cash_change) != (fill_inv, fill_cash):
                raise RuntimeError(f"Gross fills do not reconcile for {a}")
            summaries.append({
                "scenario": scenario, "seed": seed, "agent": a,
                "steps": len(acts), "risk_terminated": risk_terminated,
                "pnl_dollars": float(eq[-1] - eq[0]) / 100,
                "final_inventory": int(trajectory.inventory[-1]),
                "max_abs_inventory": int(np.abs(trajectory.inventory).max()),
                "gross_fills": len(fills), "gross_shares": sum(f.quantity for f in fills),
                "submitted_orders": len(submissions[a]),
                "direct_pair_matches": len(direct),
                "direct_pair_shares": sum(e["quantity"] for e in direct),
                "two_sided_pct": stats["two_sided_pct"],
                "any_quote_pct": stats["any_quote_pct"],
                "withdrawn_pct": stats["withdrawn_pct"],
                # Sum of offsets; outward half-tick rounding can add one cent.
                "mean_two_sided_offset_sum_cents": stats["spread_cents"],
                "gross_fill_cash_residual_cents": cash_change - fill_cash,
            })
        return summaries
    finally:
        env.close()


def simulator_sources():
    """Read the two patched sources from the active imported ABIDES modules."""
    sources = {}
    for name, rel in (
        ("abides_core.utils", "abides-core/abides_core/utils.py"),
        ("abides_markets.configs.rmsc03", "abides-markets/abides_markets/configs/rmsc03.py"),
    ):
        path = None
        try:
            module = importlib.import_module(name)
            if getattr(module, "__file__", None):
                path = Path(module.__file__).resolve()
            source = inspect.getsourcefile(module)
            if source is None:
                raise ValueError("module has no Python source file")
            path = Path(source).resolve()
            data = path.read_bytes()
        except (ImportError, TypeError, ValueError, OSError) as exc:
            location = f" at {path}" if path is not None else ""
            raise RuntimeError(
                f"Cannot snapshot runtime module {name}{location}: {exc}. "
                "Install ABIDES with readable Python source files."
            ) from exc
        sources["abides-local/" + rel] = (path, data)
    return sources


def simulator_git_sha():
    """Return the active sources' shared checkout SHA, or None outside a checkout."""
    paths = [path for path, _ in simulator_sources().values()]
    checkouts = set()
    for path in paths:
        checkout = next((p for p in path.parents if (p / ".git").exists()), None)
        checkouts.add(checkout)
    if None in checkouts or len(checkouts) != 1:
        return None
    checkout = checkouts.pop()
    try:
        # A source install inside an unrelated project is not that project's ABIDES checkout.
        subprocess.check_output(
            ["git", "-C", str(checkout), "ls-files", "--error-unmatch", "--",
             *(str(path.relative_to(checkout)) for path in paths)],
            text=True, stderr=subprocess.DEVNULL,
        )
        return subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def save_sources(out):
    """Archive source bytes and hash those same bytes, including active ABIDES sources."""
    simulator = simulator_sources()
    root = Path(__file__).resolve().parents[1]
    paths = sorted(root.glob("src/**/*.py")) + sorted(root.glob("scripts/*.py"))
    paths += [root / "pyproject.toml"]
    manifest = {}
    with zipfile.ZipFile(out / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for path in paths:
            rel = str(path.relative_to(root))
            data = path.read_bytes()
            manifest[rel] = hashlib.sha256(data).hexdigest()
            z.writestr(rel, data)
        for rel, (_, data) in simulator.items():
            manifest[rel] = hashlib.sha256(data).hexdigest()
            z.writestr(rel, data)
    write_json(out / "source_manifest.json", manifest)
    (out / "pip-freeze.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16_384, help="per learner")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[10001, 10002, 10003])
    parser.add_argument("--episode-seconds", type=int, default=600)
    parser.add_argument("--quote-size", type=int, default=10)
    parser.add_argument("--num-value-agents", type=int, default=50)
    parser.add_argument("--num-noise-agents", type=int, default=2000)
    args = parser.parse_args()
    if args.steps < 1 or not 1 <= args.episode_seconds <= 23_400:
        parser.error("positive steps and 1..23400 episode seconds required")
    if not 1 <= args.quote_size <= 100:
        parser.error("quote-size must be in 1..100")
    # Do not silently replace an earlier run, even if incomplete.
    args.out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started = time.monotonic()
    config = {
        "n_agents": 2, "starting_cash": 10_000_000, "k": 10,
        "max_size": 100, "max_inventory": 1000,
        "inventory_penalty": 1e-4, "termination_penalty": -1.0,
        "start_time": "09:30:00",
        "end_time": (datetime(2020, 6, 3, 9, 30) + timedelta(
            seconds=args.episode_seconds)).strftime("%H:%M:%S"),
        "wakeup_interval": "1s", "gated_actions": True,
        "offset_span": 15, "mid_rounding": "outward",
        "agent_id_obs": True, "agent_id_width": 2, "include_ofi": True,
        "latency_edge": 1.0,
        "config_kwargs": {"num_noise_agents": args.num_noise_agents,
                          "num_value_agents": args.num_value_agents,
                          "num_momentum_agents": 10},
    }
    meta = {
        "status": "running", "environment_config": config,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "platform": platform.platform(),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "abides_git_sha": simulator_git_sha(),
        "versions": {p: importlib.metadata.version(p)
                     for p in ("numpy", "torch", "stable-baselines3", "gymnasium")},
        "protocol": "A learns vs constant5; freeze A; B learns vs A; deterministic cross-play",
        "limitations": ["one training seed per role", "short pilot budget",
                        "mark-to-mid PnL; no fees or terminal liquidation",
                        "shared background; matched seeds do not imply identical market paths"],
    }
    write_json(args.out_dir / "run_meta.json", meta)
    save_sources(args.out_dir)
    constant = ConstantPolicy(5, args.quote_size)
    a, fresh_a = train(config, "mm_0", constant, args.quote_size,
                       args.seed, args.steps, args.out_dir / "a")
    a_hash = policy_digest(a.model)
    b, fresh_b = train(config, "mm_1", a, args.quote_size,
                       args.seed + 1, args.steps, args.out_dir / "b")
    if policy_digest(a.model) != a_hash:
        raise RuntimeError("A changed while B trained")
    meta["frozen_a_verified"] = True
    scenarios = {
        "initial_a_vs_script": (fresh_a, constant),
        "trained_a_vs_script": (a, constant),
        "trained_a_vs_initial_b": (a, fresh_b),
        "trained_a_vs_trained_b": (a, b),
        "script_vs_trained_b": (constant, b),
        "inactive_a_vs_trained_b": (ConstantPolicy(5, args.quote_size, active=False), b),
    }
    summary = []
    for scenario, pair in scenarios.items():
        for seed in args.eval_seeds:
            rows = evaluate(config, dict(zip(("mm_0", "mm_1"), pair, strict=True)), seed,
                            scenario, args.out_dir / "eval" / scenario / f"seed{seed}")
            summary.extend(rows)
            print(f"{scenario} seed={seed}: " + ", ".join(
                f"{r['agent']} ${r['pnl_dollars']:+.2f}, {r['gross_fills']} fills"
                for r in rows), flush=True)
            with (args.out_dir / "summary.csv").open("w") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary[0]))
                writer.writeheader()
                writer.writerows(summary)
    meta.update(status="complete", wall_seconds=time.monotonic() - started,
                evaluation_rows=len(summary))
    write_json(args.out_dir / "run_meta.json", meta)
    print(f"complete in {meta['wall_seconds']:.1f}s: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
