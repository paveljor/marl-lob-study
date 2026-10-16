"""Run RMSC03 + N constant-spread MMs (F) and report metrics.

End-to-end:
  1. Build rmsc03_simple config.
  2. Inject N LoggingConstantSpreadMM agents.
  3. Run the kernel to completion.
  4. For each F: build a Trajectory from snapshots, run D's compute_all,
     print Sharpe / MaxDD / final equity.
  5. Save trajectories to runs/<out>/trajectory_<i>.npz for chunk 9
     to compare against.

This is the floor for the May 10 demo: numbers F produces are the bar a
trained PPO has to beat.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
from abides_core import Kernel
from abides_core.utils import datetime_str_to_ns, str_to_ns, subdict
from abides_markets.utils import config_add_agents

from marl_lob.baseline_traj import LoggingConstantSpreadMM
from marl_lob.configs import rmsc03_simple
from marl_lob.metrics import compute_all
from marl_lob.trajectory import save_trajectory

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-agents", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-time", type=str, default="09:30:00")
    parser.add_argument("--end-time", type=str, default="10:30:00")
    parser.add_argument("--historical-date", type=str, default="20200603")
    parser.add_argument("--symbol", type=str, default="ABM")
    parser.add_argument("--starting-cash", type=int, default=10_000_000)
    parser.add_argument("--spread-ticks", type=int, default=10)
    parser.add_argument("--quote-size", type=int, default=100)
    parser.add_argument("--wake-up-freq", type=str, default="10s")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/baseline"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Background config + market window
    bg = rmsc03_simple.build_config(
        start_time=args.start_time, end_time=args.end_time, seed=args.seed,
    )
    date_ns = datetime_str_to_ns(args.historical_date)
    mkt_open_ns = date_ns + str_to_ns(args.start_time)
    next_id = len(bg["agents"])

    fs = [
        LoggingConstantSpreadMM(
            id=next_id + i,
            symbol=args.symbol,
            starting_cash=args.starting_cash,
            spread_ticks=args.spread_ticks,
            quote_size=args.quote_size,
            wake_up_freq=str_to_ns(args.wake_up_freq),
            mkt_open_ns=mkt_open_ns,
        )
        for i in range(args.n_agents)
    ]
    config = config_add_agents(bg, list(fs))

    kernel = Kernel(
        random_state=np.random.RandomState(seed=args.seed),
        **subdict(config, [
            "start_time", "stop_time", "agents",
            "agent_latency_model", "default_computation_delay",
            "custom_properties",
        ]),
    )
    kernel.initialize()
    kernel.runner()        # run to stop_time (no experimental agent → no pause)
    kernel.terminate()

    print(f"\nbaseline F × {args.n_agents}, seed={args.seed}, "
          f"window {args.start_time}-{args.end_time}")
    print("-" * 70)

    for i, f in enumerate(fs):
        traj = f.to_trajectory()
        if len(traj) == 0:
            print(f"  F[{i}]: no snapshots recorded (kernel may have exited "
                  f"before mkt_open)")
            continue
        # Infer step duration from the trajectory itself rather than the
        # default 1s — F's snapshot rate is its wakeup_freq (~10s).
        dt = float(np.median(np.diff(traj.timestamps))) if len(traj) > 1 else 1.0
        m = compute_all(traj, dt_seconds=dt)
        eq = traj.equity()
        print(f"  F[{i}]: snapshots={len(traj)} fills={len(traj.fills)}")
        print(f"    Sharpe       : {m['sharpe']:.3f}")
        print(f"    MaxDrawdown  : {m['max_drawdown'] * 100:.2f}% "
              f"(peak idx {m['max_drawdown_peak_idx']} → "
              f"trough idx {m['max_drawdown_trough_idx']})")
        print(f"    Final equity : {int(eq[-1]):>12} cents  "
              f"(Δ={int(eq[-1]-eq[0]):+d}, "
              f"min={int(eq.min())}, max={int(eq.max())})")
        print(f"    Final inv    : {int(traj.inventory[-1])} shares")

        # Save full trajectory (incl. fills) so chunk 9 can compare PPO
        # against F under identical metrics. Same writer as eval.py, so the
        # two paths cannot drift apart.
        save_trajectory(
            args.out_dir / f"trajectory_{i}_seed{args.seed}.npz", traj
        )

    print(f"\ntrajectories saved → {args.out_dir}/")


if __name__ == "__main__":
    main()
