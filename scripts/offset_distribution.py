#!/usr/bin/env python3
"""Distribution of the quote offsets a run's policy actually emits.

The aggregate mean of the raw offset action cannot separate two very different
things. Under a rescaled mapping ``raw = 0`` is the *midpoint* of the offset
range, and a freshly initialised SB3 Gaussian policy also sits at ``raw ~ 0``.
So "explored the dimension and settled in the middle" and "never left its
initialisation" both report a mean near zero. Dispersion and shape separate
them, and that is what this script measures.

Four action populations are compared, per arm:

``trained_det``
    The actions saved in each eval trajectory. ``scripts/eval.py`` rolls out
    with ``deterministic=True``, so these are the Gaussian's *mean* - their
    spread across steps is how much the mean network moves with the
    observation, not the width of the policy's distribution.
``trained_stoch``
    The same trained checkpoint re-sampled with ``deterministic=False`` over
    the same saved observations. This is the population that shows the
    Gaussian's width, and it is the like-for-like partner of ``init_stoch``.
``init_det`` / ``init_stoch``
    A freshly constructed ``PPO("MlpPolicy", ...)`` built against **that arm's
    own** observation and action spaces, seeded with the run's seed, evaluated
    over the same observations. Nothing is trained; this is forward passes
    through an untouched network.

Comparing ``trained_det`` against ``init_stoch`` would conflate "the Gaussian
narrowed" with "eval is deterministic", so both pairings are reported.

The offset in ticks is never recomputed here. It comes from
``quote_analysis.effective_quotes``, under the ``QuoteMode`` read from each
run's own ``run_meta.json``, which is the same arithmetic the live agent runs.
Each arm in this suite has different action semantics and replaying one under
another's mapping produces plausible, wrong numbers silently.

    python scripts/offset_distribution.py --suite rescale --out offsets.json
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import gymnasium as gym
import numpy as np

warnings.filterwarnings("ignore")

from marl_lob.actions import DEFAULT_MIN_OFFSET_TICKS  # noqa: E402
from marl_lob.quote_analysis import QuoteMode, effective_quotes  # noqa: E402

VARIANTS = ("trained_det", "trained_stoch", "init_det", "init_stoch")
SIDES = ("bid", "ask")
MAX_TICK = 4096  # histogram width; offsets are bounded well below this


class Accum:
    """Running histogram + moments for one (arm, variant, side) population.

    Kept as counts rather than raw samples: the pooled populations are ~10^5
    steps per arm per variant, and every statistic wanted here (mean, sd,
    distinct values, floor and ceiling share, full histogram) is recoverable
    from a tick histogram plus the raw action's first two moments.
    """

    def __init__(self) -> None:
        self.hist = np.zeros(MAX_TICK, dtype=np.int64)
        self.n_quote = 0          # steps on which this side quoted
        self.n_total = 0          # steps seen
        self.raw_sum = 0.0        # raw action, quoting steps only
        self.raw_sumsq = 0.0
        self.raw_all_sum = 0.0    # raw action, every step
        self.raw_all_sumsq = 0.0

    def add(self, ticks: np.ndarray, raw: np.ndarray) -> None:
        quoting = ticks > 0
        t = ticks[quoting]
        if t.size:
            self.hist += np.bincount(t, minlength=MAX_TICK)[:MAX_TICK]
        r = raw[quoting]
        self.n_quote += int(quoting.sum())
        self.n_total += int(ticks.size)
        self.raw_sum += float(r.sum())
        self.raw_sumsq += float((r * r).sum())
        self.raw_all_sum += float(raw.sum())
        self.raw_all_sumsq += float((raw * raw).sum())

    def merge(self, other: Accum) -> None:
        self.hist += other.hist
        self.n_quote += other.n_quote
        self.n_total += other.n_total
        self.raw_sum += other.raw_sum
        self.raw_sumsq += other.raw_sumsq
        self.raw_all_sum += other.raw_all_sum
        self.raw_all_sumsq += other.raw_all_sumsq

    def summary(self, floor_tick: int, ceil_tick: int | None) -> dict:
        n = self.n_quote
        if n == 0:
            return {"n_quote_steps": 0, "n_steps": self.n_total}
        ticks = np.nonzero(self.hist)[0]
        counts = self.hist[ticks].astype(float)
        mean = float((ticks * counts).sum() / n)
        var = float((counts * (ticks - mean) ** 2).sum() / n)
        raw_mean = self.raw_sum / n
        raw_var = max(self.raw_sumsq / n - raw_mean * raw_mean, 0.0)
        raw_all_mean = self.raw_all_sum / self.n_total
        raw_all_var = max(
            self.raw_all_sumsq / self.n_total - raw_all_mean * raw_all_mean, 0.0)
        return {
            "n_steps": self.n_total,
            "n_quote_steps": n,
            "quote_share_pct": 100.0 * n / self.n_total,
            "tick_mean": mean,
            "tick_sd": float(np.sqrt(var)),
            "tick_min": int(ticks.min()),
            "tick_max": int(ticks.max()),
            "distinct_ticks": int(ticks.size),
            "floor_share_pct": 100.0 * float(self.hist[floor_tick]) / n,
            "ceil_share_pct": (
                100.0 * float(self.hist[ceil_tick]) / n
                if ceil_tick is not None and ceil_tick < MAX_TICK else None),
            "raw_mean": raw_mean,
            "raw_sd": float(np.sqrt(raw_var)),
            "raw_mean_all_steps": raw_all_mean,
            "raw_sd_all_steps": float(np.sqrt(raw_all_var)),
            "histogram": {int(t): int(c) for t, c in
                          zip(ticks, self.hist[ticks], strict=False)},
        }


def arm_spaces(args: dict):
    """The exact observation and action spaces this run was trained against.

    Built by instantiating ``MarlLobEnv`` with the run's own recorded flags -
    the spaces are set in ``__init__`` and no ABIDES kernel is created until
    ``reset``, so this costs nothing and cannot drift from the real thing the
    way a re-derived Box would.
    """
    from marl_lob.env import MarlLobEnv

    env = MarlLobEnv(
        n_agents=int(args.get("n_agents", 1)),
        max_inventory=10_000,
        inventory_penalty=float(args.get("inventory_penalty", 0.0)),
        min_size=int(args.get("min_size", 0)),
        max_offset_cents=int(args.get("max_offset_cents", 50)),
        gated_actions=not bool(args.get("legacy_actions", False)),
        min_offset_ticks=int(args.get("min_offset_ticks", 1)),
        min_quote_size=int(args.get("min_quote_size", 1)),
        offset_span=args.get("offset_span"),
        mid_rounding=args.get("mid_rounding") or "floor",
        agent_id_obs=not bool(args.get("no_agent_id_obs", False)),
        agent_id_width=args.get("agent_id_width"),
        include_ofi=bool(args.get("ofi", False)),
        latency_edge=args.get("latency_edge"),
        config_kwargs=args.get("config_kwargs") or None,
    )
    agent = env.possible_agents[0]
    return env.observation_space(agent), env.action_space(agent)


class _SpaceEnv(gym.Env):
    """Minimal gym-shaped stand-in carrying only the two spaces.

    SB3 builds its policy from the spaces alone; nothing here is stepped, so
    no simulator is started. Used only to hand ``PPO`` the same spaces the arm
    trained under.
    """

    metadata: dict = {}
    render_mode = None

    def __init__(self, obs_space, act_space):
        self.observation_space = obs_space
        self.action_space = act_space

    def reset(self, *, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}

    def close(self):
        pass


def fresh_model(obs_space, act_space, seed: int):
    """A PPO whose policy has never seen a gradient step."""
    from stable_baselines3 import PPO

    return PPO(
        "MlpPolicy",
        _SpaceEnv(obs_space, act_space),
        n_steps=512,
        device="cpu",
        seed=seed,
        verbose=0,
    )


def batched_predict(model, obs: np.ndarray, deterministic: bool,
                    batch: int = 8192) -> np.ndarray:
    """``model.predict`` over a stack of observations.

    SB3 clips to the action-space box inside ``predict``, which is what the
    rollout collector does too, so the returned values are on the same scale
    as the actions saved in the trajectories.
    """
    out = []
    for i in range(0, len(obs), batch):
        a, _ = model.predict(obs[i:i + batch], deterministic=deterministic)
        out.append(np.asarray(a, dtype=np.float32))
    return np.concatenate(out) if out else np.zeros((0, 6), dtype=np.float32)


def offset_columns(actions: np.ndarray) -> tuple[int, int]:
    """Raw bid/ask offset column indices for this action layout."""
    width = actions.shape[1]
    if width == 6:
        return 2, 3
    if width == 4:
        return 0, 1
    raise ValueError(f"unrecognised action width {width}")


def side_ticks(actions: np.ndarray, mode: QuoteMode, max_size: int = 100):
    """Resolved (bid, ask) offsets in ticks, zero where the side is silent."""
    quotes = effective_quotes(actions, max_size, **mode.kwargs())
    return quotes[:, 0], quotes[:, 1]


def progress_std(run_dir: Path) -> dict | None:
    """Start/end of ``train/std`` from progress.csv, or None if absent.

    SB3 logs a single ``train/std`` - ``exp(log_std).mean()`` over *all* action
    dimensions - so this is not the offset dimensions' std on its own. The
    per-dimension values are read from the checkpoint instead.
    """
    csv = run_dir / "progress.csv"
    if not csv.is_file():
        return None
    import csv as _csv

    rows = list(_csv.DictReader(csv.open()))
    vals = [(int(float(r["time/total_timesteps"])), float(r["train/std"]))
            for r in rows
            if r.get("train/std") not in (None, "") and
            r.get("time/total_timesteps") not in (None, "")]
    if not vals:
        return None
    return {"first": vals[0][1], "last": vals[-1][1], "n_updates": len(vals),
            "first_step": vals[0][0], "last_step": vals[-1][0]}


def checkpoint_log_std(model) -> list[float]:
    """Per-dimension std of the trained Gaussian, in raw action units."""
    import numpy as _np

    return [float(x) for x in _np.exp(
        model.policy.log_std.detach().cpu().numpy())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="rescale")
    ap.add_argument("--runs-root", default=None,
                    help="default <repo>/runs")
    ap.add_argument("--arms", default=None, help="comma-separated subset")
    ap.add_argument("--max-size", type=int, default=100)
    ap.add_argument("--sample-seed", type=int, default=12345,
                    help="RNG seed for the stochastic re-sampling passes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-init", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    root = Path(args.runs_root) if args.runs_root else repo / "runs"
    root = root / args.suite
    if not root.is_dir():
        raise SystemExit(f"no runs at {root}")

    arms = (args.arms.split(",") if args.arms
            else sorted(p.name for p in root.iterdir() if p.is_dir()))

    from stable_baselines3 import PPO

    result = {"suite": args.suite, "runs_root": str(root), "arms": {}}

    for arm in arms:
        arm_dir = root / arm
        seeds = sorted((p for p in arm_dir.iterdir() if p.is_dir()),
                       key=lambda p: (len(p.name), p.name))
        pooled = {v: {s: Accum() for s in SIDES} for v in VARIANTS}
        per_seed = []
        modes = set()
        arm_args = None
        obs_space = act_space = None
        std_rows = []
        final_std = []

        for run_dir in seeds:
            meta_path = run_dir / "run_meta.json"
            if not meta_path.is_file():
                print(f"  skip {run_dir} - no run_meta.json")
                continue
            meta = json.loads(meta_path.read_text())
            margs = meta.get("args", {})
            mode = QuoteMode.from_run_meta(meta)
            modes.add(mode.label())
            if arm_args is None:
                arm_args = margs
                obs_space, act_space = arm_spaces(margs)

            npzs = sorted((run_dir / "eval").glob("*.npz"))
            if not npzs:
                print(f"  skip {run_dir} - no eval trajectories")
                continue

            ckpt = run_dir / "ppo_marl_lob.zip"
            trained = PPO.load(ckpt, device="cpu") if ckpt.is_file() else None
            if trained is not None:
                trained.set_random_seed(args.sample_seed)
                final_std.append(checkpoint_log_std(trained))
            init = None
            if not args.skip_init:
                init = fresh_model(obs_space, act_space,
                                   int(margs.get("seed", 0)))
                init.set_random_seed(args.sample_seed)

            seed_acc = {v: {s: Accum() for s in SIDES} for v in VARIANTS}
            for npz in npzs:
                d = np.load(npz)
                acts = np.asarray(d["actions"], dtype=np.float32)
                obs = np.asarray(d["observations"], dtype=np.float32)
                obs = obs[:len(acts)]
                if len(obs) != len(acts):
                    print(f"  {npz}: obs/action length mismatch, skipping")
                    continue
                bcol, acol = offset_columns(acts)

                populations = {"trained_det": acts}
                if trained is not None:
                    populations["trained_stoch"] = batched_predict(
                        trained, obs, deterministic=False)
                if init is not None:
                    populations["init_det"] = batched_predict(
                        init, obs, deterministic=True)
                    populations["init_stoch"] = batched_predict(
                        init, obs, deterministic=False)

                for variant, a in populations.items():
                    bt, at = side_ticks(a, mode, args.max_size)
                    seed_acc[variant]["bid"].add(bt, a[:, bcol])
                    seed_acc[variant]["ask"].add(at, a[:, acol])

            floor_tick = int(margs.get("min_offset_ticks",
                                       DEFAULT_MIN_OFFSET_TICKS))
            span = margs.get("offset_span")
            ceil_tick = (floor_tick + int(span) if span
                         else int(margs.get("max_offset_cents", 50)))

            row = {"seed": int(margs.get("seed", -1)), "run_dir": str(run_dir),
                   "n_eval_files": len(npzs), "mode": mode.label(),
                   "train_std": progress_std(run_dir)}
            for variant in VARIANTS:
                for side in SIDES:
                    acc = seed_acc[variant][side]
                    if acc.n_total == 0:
                        continue
                    pooled[variant][side].merge(acc)
                    row[f"{variant}_{side}"] = acc.summary(floor_tick,
                                                           ceil_tick)
            if row["train_std"]:
                std_rows.append(row["train_std"])
            per_seed.append(row)
            print(f"  {arm}/{run_dir.name} done")

        floor_tick = int((arm_args or {}).get("min_offset_ticks",
                                              DEFAULT_MIN_OFFSET_TICKS))
        span = (arm_args or {}).get("offset_span")
        ceil_tick = (floor_tick + int(span) if span
                     else int((arm_args or {}).get("max_offset_cents", 50)))

        result["arms"][arm] = {
            "mode": modes.pop() if len(modes) == 1 else "MIXED",
            "args": {k: (arm_args or {}).get(k) for k in
                     ("offset_span", "mid_rounding", "min_offset_ticks",
                      "min_quote_size", "max_offset_cents", "legacy_actions",
                      "n_agents", "ofi", "latency_edge", "norm_obs")},
            "floor_tick": floor_tick,
            "ceil_tick": ceil_tick,
            "n_seeds": len(per_seed),
            "pooled": {
                f"{v}_{s}": pooled[v][s].summary(floor_tick, ceil_tick)
                for v in VARIANTS for s in SIDES
                if pooled[v][s].n_total},
            "train_std": ({
                "first_mean": float(np.mean([r["first"] for r in std_rows])),
                "last_mean": float(np.mean([r["last"] for r in std_rows])),
                "first_min": float(np.min([r["first"] for r in std_rows])),
                "first_max": float(np.max([r["first"] for r in std_rows])),
                "last_min": float(np.min([r["last"] for r in std_rows])),
                "last_max": float(np.max([r["last"] for r in std_rows])),
                "n_runs": len(std_rows),
                "n_updates": std_rows[0]["n_updates"],
            } if std_rows else None),
            "checkpoint_std_per_dim": ({
                "mean": np.mean(final_std, axis=0).tolist(),
                "min": np.min(final_std, axis=0).tolist(),
                "max": np.max(final_std, axis=0).tolist(),
                "n_runs": len(final_std),
            } if final_std else None),
            "per_seed": per_seed,
        }
        print(f"{arm}: {len(per_seed)} seeds")

    Path(args.out).write_text(json.dumps(result, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
