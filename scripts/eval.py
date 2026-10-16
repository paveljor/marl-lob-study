"""Roll a trained PPO checkpoint through MarlLobEnv and compare against F.

Side-by-side metric printout per (seed, agent), reading F's saved
trajectories from `runs/baseline/`. The "one command, two Sharpe numbers"
deliverable for chunk 9.

Assumes scripts/run_baseline.py has been run on the same seeds first
(otherwise: pass --no-baseline to just print PPO numbers).
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import NamedTuple

import numpy as np
from stable_baselines3 import PPO

from marl_lob.checkpoint_config import evaluation_env_kwargs
from marl_lob.env import MarlLobEnv
from marl_lob.metrics import compute_all
from marl_lob.trajectory import Trajectory, load_trajectory, save_trajectory


def load_obs_normalizer(checkpoint: Path):
    """Return raw-obs -> policy-space-obs, or None if the model wants raw obs.

    A policy trained with --norm-obs expects VecNormalize-standardised
    observations. Feeding it raw ones evaluates a different input distribution
    than it was trained on: no crash, no warning, just meaningless numbers.
    train.py saves the running statistics next to the checkpoint precisely so
    eval can reproduce them.
    """
    pkl = checkpoint.parent / "vecnormalize.pkl"
    if not pkl.is_file():
        return None
    with pkl.open("rb") as fh:
        vn = pickle.load(fh)
    if not getattr(vn, "norm_obs", False):
        return None  # norm_reward only - observations were raw during training
    rms, clip, eps = vn.obs_rms, vn.clip_obs, vn.epsilon

    def normalize(obs: np.ndarray) -> np.ndarray:
        return np.clip(
            (obs - rms.mean) / np.sqrt(rms.var + eps), -clip, clip
        ).astype(np.float32)

    return normalize


class Rollout(NamedTuple):
    """One agent's episode: the metrics trajectory plus the raw policy I/O.

    `observations` has one row per snapshot (the reset row first); `actions`
    has one row per step, so `actions[i]` is the action taken from
    `observations[i]` that produced snapshot `i + 1`. Both are kept because
    the interesting failures are visible only here: whether the policy is
    frozen at initialisation, and where in the book it chose to quote.
    """
    traj: Trajectory
    actions: np.ndarray
    observations: np.ndarray


def rollout_ppo(env: MarlLobEnv, model, seed: int, max_steps: int,
                normalize=None) -> dict[str, Rollout]:
    """Run a deterministic-policy rollout, return per-agent Rollout.

    `normalize` maps raw observations into the space the policy was trained
    in; observations are logged raw either way, since that is what the
    market actually showed the agent."""
    to_policy = normalize if normalize is not None else (lambda o: o)
    obs, info = env.reset(seed=seed)
    rows: dict[str, list[tuple]] = {
        a: [info[a]["traj_row"]] for a in env.possible_agents
    }
    obs_log: dict[str, list[np.ndarray]] = {
        a: [np.asarray(obs[a], dtype=np.float32)] for a in env.possible_agents if a in obs
    }
    act_log: dict[str, list[np.ndarray]] = {a: [] for a in env.possible_agents}
    for _ in range(max_steps):
        if not env.agents:
            break
        actions = {
            a: model.predict(to_policy(obs[a]), deterministic=True)[0].astype(np.float32)
            for a in env.agents
        }
        for a, act in actions.items():
            act_log[a].append(act)
        obs, _r, _t, trunc, info = env.step(actions)
        for a in env.possible_agents:
            if a in info:
                rows[a].append(info[a]["traj_row"])
            if a in obs:
                obs_log.setdefault(a, []).append(np.asarray(obs[a], dtype=np.float32))
        if any(trunc.values()):
            break

    def stack(seq: list[np.ndarray], width_from: list[np.ndarray]) -> np.ndarray:
        if seq:
            return np.stack(seq).astype(np.float32)
        width = width_from[0].shape[0] if width_from else 0
        return np.zeros((0, width), dtype=np.float32)

    return {
        a: Rollout(
            traj=Trajectory.from_tuples(rows[a]),
            actions=stack(act_log.get(a, []), obs_log.get(a, [])),
            observations=stack(obs_log.get(a, []), obs_log.get(a, [])),
        )
        for a in env.possible_agents
    }


def infer_dt_seconds(traj: Trajectory) -> float:
    """Median Δt between snapshots — used to annualize Sharpe."""
    if len(traj) < 2:
        return 1.0
    return float(np.median(np.diff(traj.timestamps)))


def metric_block(label: str, traj: Trajectory) -> None:
    if len(traj) == 0:
        print(f"    {label}: no snapshots")
        return
    dt = infer_dt_seconds(traj)
    m = compute_all(traj, dt_seconds=dt)
    eq = traj.equity()
    print(
        f"    {label:>5}: snaps={len(traj):>4}  fills={len(traj.fills):>4}  "
        f"Sharpe={m['sharpe']:>+8.2f}  "
        f"MaxDD={m['max_drawdown'] * 100:>6.2f}%  "
        f"Δeq={int(eq[-1] - eq[0]):>+10d}c  "
        f"finalInv={int(traj.inventory[-1]):>+5d}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path,
                        help="path to a .zip checkpoint from scripts/train.py")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--n-agents", type=int, default=None,
                        help="override the number of agents saved with the checkpoint")
    parser.add_argument("--max-steps", type=int, default=3600)
    parser.add_argument("--baseline-dir", type=Path, default=Path("runs/baseline"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/eval"))
    parser.add_argument("--no-baseline", action="store_true",
                        help="skip F comparison (just print PPO numbers)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = PPO.load(args.checkpoint)
    print(f"loaded checkpoint: {args.checkpoint}")
    normalize = load_obs_normalizer(args.checkpoint)
    print("obs normalisation: "
          + ("VecNormalize stats from vecnormalize.pkl"
             if normalize is not None else "none (policy trained on raw obs)"))

    try:
        env_kwargs = evaluation_env_kwargs(
            args.checkpoint,
            action_space=model.action_space,
            observation_space=model.observation_space,
            n_agents=args.n_agents,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    n_agents = env_kwargs["n_agents"]
    gated = env_kwargs["gated_actions"]
    agent_id_width = env_kwargs.get("agent_id_width")
    if not env_kwargs.get("agent_id_obs", True):
        agent_id_width = 0
    elif agent_id_width is None:
        agent_id_width = n_agents
    include_ofi = env_kwargs.get("include_ofi", False)
    print(f"environment: {env_kwargs}")
    (args.out_dir / "eval_meta.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint.resolve()),
        "seeds": args.seeds,
        "max_steps": args.max_steps,
        "n_agents": n_agents,
        "environment_config": env_kwargs,
    }, indent=2) + "\n")

    for seed in args.seeds:
        env = MarlLobEnv(**env_kwargs)
        rollouts = rollout_ppo(env, model, seed, args.max_steps, normalize)
        env.close()

        print(f"\n== seed {seed} ==")
        for i in range(n_agents):
            agent = f"mm_{i}"
            roll = rollouts[agent]
            ppo = roll.traj
            print(f"  agent {agent}")
            metric_block("PPO", ppo)

            if not args.no_baseline:
                f_path = args.baseline_dir / f"trajectory_{i}_seed{seed}.npz"
                if f_path.exists():
                    f_traj = load_trajectory(f_path)
                    metric_block("F", f_traj)
                else:
                    print(f"    F   : no baseline at {f_path}; "
                          f"run scripts/run_baseline.py --seed {seed}")

            # Save the full PPO rollout. Fills drive the markout
            # decomposition, actions show where the policy chose to quote,
            # observations show the magnitudes it was fed - none of which can
            # be recovered without re-running the episode.
            save_trajectory(
                args.out_dir / f"ppo_trajectory_{i}_seed{seed}.npz",
                ppo,
                actions=roll.actions,
                observations=roll.observations,
                # Layout metadata: the column meaning of `actions` and the
                # width of the identity suffix in `observations` cannot be
                # recovered from the arrays alone, and old and new files sit
                # side by side in the same runs/ tree.
                gated=np.array(int(gated)),
                agent_id_width=np.array(agent_id_width),
                include_ofi=np.array(int(bool(include_ofi))),
                n_agents=np.array(int(n_agents)),
            )

    print(f"\nPPO trajectories saved → {args.out_dir}/")


if __name__ == "__main__":
    main()
