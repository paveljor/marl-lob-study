"""Compare paired PPO checkpoints on identical saved market observations.

The states come from an evaluated sampled-pair trajectory. These diagnostics
measure policy response on those states, not counterfactual trading outcomes.
No new market is simulated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from inspect_policy_response import distribution
from stable_baselines3 import PPO


def _observations(evaluation: Path, agent: str) -> tuple[np.ndarray, list[str]]:
    files = sorted((evaluation / "eval" / "sampled_pair").glob(f"*/{agent}.npz"))
    if not files:
        raise FileNotFoundError(f"No sampled-pair observations for {agent} in {evaluation}")
    states = []
    for file in files:
        with np.load(file) as data:
            # The last observation has no matching decision in the trajectory.
            states.append(data["observations"][:-1].copy())
    return np.concatenate(states), [str(file) for file in files]


def _checkpoints(run: Path) -> dict[str, dict[str, Path]]:
    snapshots = {
        "initial": {
            "mm_0": run / "a" / "initial.zip",
            "mm_1": run / "b" / "initial.zip",
        }
    }
    for folder in sorted(
        (run / "checkpoints").glob("step*"),
        key=lambda path: int(path.name[4:]) if path.name[4:].isdigit() else -1,
    ):
        if not folder.name[4:].isdigit():
            continue
        pair = {agent: folder / f"{agent}.zip" for agent in ("mm_0", "mm_1")}
        if all(path.is_file() for path in pair.values()):
            snapshots[folder.name] = pair
    if not all(path.is_file() for path in snapshots["initial"].values()):
        raise FileNotFoundError("Initial paired checkpoints are incomplete")
    return snapshots


def _summary(model: PPO, states: np.ndarray) -> dict:
    mean, std, gates = distribution(model, states)
    gate_names = ("bid", "ask", "either")
    deterministic = np.column_stack([
        mean[:, 0] > 0,
        mean[:, 1] > 0,
        (mean[:, :2] > 0).any(axis=1),
    ])
    return {
        "n_states": len(states),
        "gate_probability": {
            name: {
                "mean": float(gates[:, i].mean()),
                "std_over_states": float(gates[:, i].std()),
                "min": float(gates[:, i].min()),
                "max": float(gates[:, i].max()),
            }
            for i, name in enumerate(gate_names)
        },
        "deterministic_gate_fraction": {
            name: float(deterministic[:, i].mean())
            for i, name in enumerate(gate_names)
        },
        "gaussian_std_mean": std.mean(axis=0).tolist(),
        "raw_mean_average": mean.mean(axis=0).tolist(),
        "raw_mean_std_over_states": mean.std(axis=0).tolist(),
        "raw_mean_min": mean.min(axis=0).tolist(),
        "raw_mean_max": mean.max(axis=0).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--training-run", type=Path,
        help="compare another run's checkpoints on these same saved observations",
    )
    args = parser.parse_args()
    evaluation = args.evaluation.resolve()
    meta = json.loads((evaluation / "run_meta.json").read_text())
    observation_run = Path(meta["parent_run"]).resolve()
    run = args.training_run.resolve() if args.training_run else observation_run
    run_meta = json.loads((run / "run_meta.json").read_text())
    if run_meta["environment_config"] != meta["environment_config"]:
        raise ValueError("Training and observation environment configurations differ")
    if run_meta.get("norm_obs", False) or meta.get("norm_obs", False):
        raise ValueError("These diagnostics require unnormalized saved observations")
    torch.set_num_threads(1)
    states = {}
    sources = {}
    for agent in ("mm_0", "mm_1"):
        states[agent], sources[agent] = _observations(evaluation, agent)
    result = {
        "evaluation": str(evaluation),
        "parent_run": str(run),
        "observation_run": str(observation_run),
        "observation_sources": sources,
        "limitation": (
            "All checkpoints are evaluated on the same states from the saved sampled-pair "
            "evaluation. These are policy-response comparisons, not market-outcome "
            "counterfactuals or states sampled from each checkpoint's own trajectory."
        ),
        "snapshots": {},
    }
    for name, pair in _checkpoints(run).items():
        result["snapshots"][name] = {}
        for agent, path in pair.items():
            model = PPO.load(path, device="cpu")
            result["snapshots"][name][agent] = {
                "checkpoint": str(path),
                **_summary(model, states[agent]),
            }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.out} ({len(result['snapshots'])} paired snapshots)")


if __name__ == "__main__":
    main()
