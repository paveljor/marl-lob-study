"""Train PPO on MarlLobEnv (the real thing).

Wiring: PettingZoo parallel env -> SuperSuit vec env -> SB3 PPO with parameter
sharing across the learning agents.

Notes:
- Default num_vec_envs=1 because each ABIDES sim spawns ~2000+ background
  agents; running 8 in parallel is heavy. Bump for real training runs.
- max_inventory bumped to 10_000 so random/early-policy rollouts don't
  trip the per-agent termination cap (which would shrink the agent set
  and break SuperSuit's fixed-agent assumption).
- Defaults are kept exactly as the May 2026 experiments ran them, so results
  in notes/experiments_results.md stay reproducible. In particular
  --inventory-penalty defaults to 0.0 (the env default is 1e-4) and
  observation normalisation is OFF unless --norm-obs is passed.
- device defaults to cpu: the policy is a tiny MLP on a 44-dim observation and
  the bottleneck is the single-threaded ABIDES kernel, so a GPU does not help.
"""

import argparse
import json
import platform
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor, VecNormalize

from marl_lob.actions import (
    DEFAULT_MID_ROUNDING,
    DEFAULT_MIN_OFFSET_TICKS,
    DEFAULT_MIN_QUOTE_SIZE,
    DEFAULT_OFFSET_SPAN,
    MID_ROUNDING_MODES,
)
from marl_lob.env import MarlLobEnv


class BoolDones(VecEnvWrapper):
    """Cast SuperSuit's uint8 done flags to bool.

    SuperSuit builds terminations/truncations with dtype=np.uint8. SB3's
    VecNormalize then runs `self.returns[dones] = 0`, which on an integer
    dtype is fancy indexing, not boolean masking:

      n_agents=1, dones=[1]   -> returns[1] -> IndexError (hard crash)
      n_agents>1, dones=[0,0] -> zeroes returns[0] every non-terminal step
      n_agents>1, dones=[1,1] -> zeroes only returns[1], leaving the rest

    train.py sets norm_reward=True, so the corrupted `returns` feeds ret_rms,
    which scales every reward. The corrupted fraction is 1/n_agents, so the
    distortion tracks n_agents - the grid's own independent variable.
    """

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        return obs, rewards, np.asarray(dones, dtype=bool), infos


def make_env(
    num_vec_envs: int,
    *,
    n_agents: int = 2,
    inventory_penalty: float,
    vecnormalize: bool,
    norm_obs: bool = False,
    min_size: int = 0,
    max_offset_cents: int = 50,
    gated_actions: bool = True,
    min_offset_ticks: int = DEFAULT_MIN_OFFSET_TICKS,
    min_quote_size: int = DEFAULT_MIN_QUOTE_SIZE,
    offset_span: int | None = DEFAULT_OFFSET_SPAN,
    mid_rounding: str = DEFAULT_MID_ROUNDING,
    agent_id_obs: bool = True,
    agent_id_width: int | None = None,
    include_ofi: bool = False,
    latency_edge: float | None = None,
    config_kwargs: dict | None = None,
):
    env = MarlLobEnv(
        n_agents=n_agents,
        max_inventory=10_000,
        inventory_penalty=inventory_penalty,
        min_size=min_size,
        max_offset_cents=max_offset_cents,
        gated_actions=gated_actions,
        min_offset_ticks=min_offset_ticks,
        min_quote_size=min_quote_size,
        offset_span=offset_span,
        mid_rounding=mid_rounding,
        agent_id_obs=agent_id_obs,
        agent_id_width=agent_id_width,
        include_ofi=include_ofi,
        latency_edge=latency_edge,
        config_kwargs=config_kwargs,
    )
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env, num_vec_envs=num_vec_envs, num_cpus=1, base_class="stable_baselines3"
    )
    env = BoolDones(env)
    env = VecMonitor(env)
    if vecnormalize:
        env = VecNormalize(
            env, norm_obs=norm_obs, norm_reward=True, clip_reward=10.0
        )
    return env


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--num-vec-envs", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/marl_lob_ppo"))
    parser.add_argument("--n-steps", type=int, default=512,
                        help="PPO rollout length per env. ABIDES is slow, so "
                             "smaller rollouts give finer-grained TB logs.")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed for PPO + env reset")
    parser.add_argument("--inventory-penalty", type=float, default=0.0,
                        help="passed to MarlLobEnv (env default is 1e-4)")
    parser.add_argument("--no-vecnormalize", action="store_true",
                        help="disable VecNormalize wrapper")

    # --- population / competition ---
    parser.add_argument("--n-agents", type=int, default=2,
                        help="number of learning market makers")

    # --- market composition (forwarded to rmsc03_simple.build_config) ---
    parser.add_argument("--num-value-agents", type=int, default=None,
                        help="informed traders (build_config default 50)")
    parser.add_argument("--num-noise-agents", type=int, default=None,
                        help="uninformed volume (build_config default 2000); "
                             "below ~1000 the L1 barely moves")
    parser.add_argument("--num-momentum-agents", type=int, default=None,
                        help="trend followers (build_config default 10)")

    # --- action-space constraints ---
    parser.add_argument("--min-size", type=int, default=0,
                        help="raise action_space.low on the size dims so "
                             "quoting nothing is impossible")
    parser.add_argument("--max-offset-cents", type=int, default=50,
                        help="cap how far from mid the agent may quote; clamp "
                             "this together with --min-size to close both "
                             "escape hatches at once")

    # --- observation scaling ---
    parser.add_argument("--legacy-actions", action="store_true",
                        help="use the pre-2026-09 4-tuple action space "
                             "(bid_offset, ask_offset, bid_size, ask_size). "
                             "Sub-tick offsets and sub-unit sizes round to "
                             "zero there, so 'post nothing' is a large "
                             "zero-gradient region the policy initialises "
                             "inside. Kept as an experimental control.")
    parser.add_argument("--min-offset-ticks", type=int,
                        default=DEFAULT_MIN_OFFSET_TICKS,
                        help="floor on a gated-on side's offset, in ticks. "
                             "At >=1 the two sides can never cross. 0 reopens "
                             "the degenerate region.")
    parser.add_argument("--min-quote-size", type=int,
                        default=DEFAULT_MIN_QUOTE_SIZE,
                        help="floor on a gated-on side's size, in shares")
    parser.add_argument("--offset-span", type=int, default=DEFAULT_OFFSET_SPAN,
                        help="rescale the offset dimensions: the [0, 1] "
                             "action range maps onto min_offset_ticks + [0, "
                             "SPAN] ticks, one tick per 1/SPAN. Unset (or 0) "
                             "keeps the legacy mapping, where the raw value "
                             "is read as cents and max(round(raw), 1) sends "
                             "every raw offset in [0, 1.5) to one tick - "
                             "measured, the policy left that interval on "
                             "0.06-1.4%% of steps, so its offset choice was a "
                             "no-op on ~99%% of them. Gated layout only.")
    parser.add_argument("--mid-rounding", choices=MID_ROUNDING_MODES,
                        default=DEFAULT_MID_ROUNDING,
                        help="how to round the quoting reference off the "
                             "touch. floor (default, legacy) is "
                             "(bid+ask)//2, which lands on the bid when the "
                             "spread is odd - ~56-57%% of snapshots - so a "
                             "symmetric offset posts the ask one tick closer "
                             "to the true mid than the bid. ceil is the "
                             "mirror of that bias and exists as a causal "
                             "check, not a fix. outward rounds each side away "
                             "from the exact mid and is symmetric.")
    parser.add_argument("--no-agent-id-obs", action="store_true",
                        help="omit the agent-identity one-hot. Without it the "
                             "shared policy cannot tell the agents apart and "
                             "they quote near-identically - measured at "
                             "97-99.9%% identical quotes.")
    parser.add_argument("--agent-id-width", type=int, default=None,
                        help="pin the identity one-hot width (default: "
                             "n_agents). Pin it across a grid that varies "
                             "n_agents so every cell shares one obs width.")
    parser.add_argument("--ofi", action="store_true",
                        help="add two order-flow-imbalance features to the "
                             "observation: last-step OFI and its EWMA. "
                             "Everything else in the observation is a snapshot, "
                             "so without this the agent cannot see the book "
                             "CHANGE - and adverse selection is a statement "
                             "about flow. grid2's P1 (no widening against "
                             "informed flow) may be a fact about this gap "
                             "rather than about market making.")
    parser.add_argument("--latency-edge", type=float, default=None,
                        help="pin our agents' latency to the exchange at this "
                             "multiple of the background median. 1.0 = a "
                             "typical participant, 0.1 = ten times faster, "
                             "0.0 = co-located, >1 = a disadvantage. Left "
                             "unset, ABIDES drops us at a RANDOM point on its "
                             "Seattle-to-NYC line, so latency has been an "
                             "uncontrolled variable in every run so far - "
                             "setting this to 1.0 is a control, not a no-op.")
    parser.add_argument("--norm-obs", action="store_true",
                        help="normalise observations in VecNormalize. OFF by "
                             "default to match the May runs, but observations "
                             "are raw cents (~1e5), the same magnitude problem "
                             "that froze the policy via the reward.")

    parser.add_argument("--device", default="cpu",
                        help="torch device; cpu is correct here (tiny MLP, "
                             "simulator-bound)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        set_random_seed(args.seed)

    config_kwargs = {
        k: v
        for k, v in (
            ("num_value_agents", args.num_value_agents),
            ("num_noise_agents", args.num_noise_agents),
            ("num_momentum_agents", args.num_momentum_agents),
        )
        if v is not None
    }

    env = make_env(
        args.num_vec_envs,
        gated_actions=not args.legacy_actions,
        min_offset_ticks=args.min_offset_ticks,
        min_quote_size=args.min_quote_size,
        offset_span=args.offset_span,
        mid_rounding=args.mid_rounding,
        agent_id_obs=not args.no_agent_id_obs,
        agent_id_width=args.agent_id_width,
        include_ofi=args.ofi,
        latency_edge=args.latency_edge,
        n_agents=args.n_agents,
        inventory_penalty=args.inventory_penalty,
        vecnormalize=not args.no_vecnormalize,
        norm_obs=args.norm_obs,
        min_size=args.min_size,
        max_offset_cents=args.max_offset_cents,
        config_kwargs=config_kwargs or None,
    )
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=args.n_steps,
        device=args.device,
    )
    # Diagnostics go to progress.csv as well as tensorboard: value_loss,
    # approx_kl, explained_variance and the policy std are the instruments
    # that diagnose the reward-scale failure, and a plain CSV survives being
    # copied off a borrowed machine and read without a tensorboard install.
    model.set_logger(configure(str(args.out_dir), ["stdout", "csv", "tensorboard"]))

    started = time.time()
    model.learn(total_timesteps=args.total_timesteps, progress_bar=False)
    elapsed = time.time() - started

    model.save(args.out_dir / "ppo_marl_lob")
    # The running obs/reward statistics are part of the trained model. Without
    # them the checkpoint cannot be re-evaluated later - it would have to be
    # retrained - so they are saved next to it.
    if isinstance(env, VecNormalize):
        env.save(str(args.out_dir / "vecnormalize.pkl"))

    # provenance: without this, results collected on different machines at
    # different times cannot be compared or reproduced
    meta = {
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
        "config_kwargs": config_kwargs,
        "git_sha": _git_sha(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "train_seconds": round(elapsed, 1),
    }
    (args.out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"saved checkpoint -> {args.out_dir / 'ppo_marl_lob.zip'}")
    print(f"trained {args.total_timesteps} steps in {elapsed/60:.1f} min "
          f"on {meta['machine']} ({meta['host']})")


if __name__ == "__main__":
    main()
