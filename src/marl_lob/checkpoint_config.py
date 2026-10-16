"""Rebuild the market contract used by a saved policy.

The observation width cannot identify the background population, penalties,
latency, or quote mapping. Evaluation must read these from run_meta.json.
"""
from __future__ import annotations

import inspect
import json
import warnings
from pathlib import Path

import numpy as np

from .env import MarlLobEnv
from .observation_extractor import obs_vector_size
from .quote_analysis import QuoteMode


def evaluation_env_kwargs(
    checkpoint: Path,
    *,
    action_space,
    observation_space,
    n_agents: int | None = None,
) -> dict:
    """Return validated ``MarlLobEnv`` kwargs for a saved checkpoint.

    New runs save the actual constructor kwargs under ``environment_config``.
    Older train.py runs save argparse values and ``config_kwargs`` separately.
    A checkpoint without metadata remains usable, but its invisible market
    settings cannot be recovered and the caller receives a warning.
    """
    checkpoint = Path(checkpoint)
    meta_path = checkpoint.parent / "run_meta.json"
    action_width = int(np.prod(action_space.shape))
    obs_width = int(np.prod(observation_space.shape))
    if action_width not in (4, 6):
        raise ValueError(f"unsupported checkpoint action width {action_width}")
    gated = action_width == 6

    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid checkpoint metadata {meta_path}: {exc}") from exc
        if not isinstance(meta, dict):
            raise ValueError(f"invalid checkpoint metadata {meta_path}: expected an object")
        if "environment_config" in meta:
            config = meta["environment_config"]
            if not isinstance(config, dict):
                raise ValueError("environment_config must be an object")
            kwargs = dict(config)
        else:
            args = meta.get("args")
            if not isinstance(args, dict):
                raise ValueError(f"invalid checkpoint metadata {meta_path}: args must be an object")
            population = meta.get("config_kwargs", {})
            if population is None:
                population = {}
            if not isinstance(population, dict):
                raise ValueError("config_kwargs must be an object")
            # Early run metadata did not carry the separate config_kwargs key.
            # Its argparse population fields are enough to reconstruct it.
            population = dict(population)
            for arg, key in (("num_value_agents", "num_value_agents"),
                             ("num_noise_agents", "num_noise_agents"),
                             ("num_momentum_agents", "num_momentum_agents")):
                if args.get(arg) is not None:
                    if key in population and population[key] != args[arg]:
                        raise ValueError(f"conflicting {key} in args and config_kwargs")
                    population[key] = args[arg]
            quote = QuoteMode.from_run_args(args)
            ofi = args.get("ofi", False)
            inferred_id_width = obs_width - obs_vector_size(include_ofi=ofi)
            has_identity = (not args["no_agent_id_obs"] if "no_agent_id_obs" in args
                            else inferred_id_width > 0)
            kwargs = {
                "n_agents": args.get("n_agents", 2),
                "max_inventory": 10_000,
                "inventory_penalty": args.get("inventory_penalty", 0.0),
                "min_size": args.get("min_size", 0),
                "max_offset_cents": args.get("max_offset_cents", 50),
                "gated_actions": not args.get("legacy_actions", not gated),
                "agent_id_obs": has_identity,
                "agent_id_width": args.get("agent_id_width", inferred_id_width),
                "include_ofi": ofi,
                "latency_edge": args.get("latency_edge"),
                "offset_span": quote.offset_span,
                "mid_rounding": quote.mid_rounding,
                "min_offset_ticks": quote.min_offset_ticks,
                "min_quote_size": quote.min_quote_size,
                "config_kwargs": population,
            }
    else:
        if gated and action_space.low[2] < 0:
            raise ValueError(
                f"No run_meta.json beside {checkpoint}; normalized offset "
                "bounds do not identify offset_span, so quote semantics "
                "cannot be reconstructed"
            )
        warnings.warn(
            f"No run_meta.json beside {checkpoint}; market population, penalties "
            "and other invisible settings cannot be verified. Using historical defaults.",
            UserWarning,
            stacklevel=2,
        )
        agent_id_width = obs_width - obs_vector_size()
        if agent_id_width < 0:
            raise ValueError(f"cannot infer observation layout from width {obs_width}")
        kwargs = {
            "n_agents": 2,
            "max_inventory": 10_000,
            "inventory_penalty": 0.0,
            "gated_actions": gated,
            "agent_id_obs": agent_id_width > 0,
            "agent_id_width": agent_id_width,
            "include_ofi": False,
            "min_size": int(action_space.low[-1]),
            "max_size": int(action_space.high[-1]),
            "max_offset_cents": 50 if gated and action_space.low[2] < 0
                                else int(action_space.high[0 if not gated else 2]),
            "offset_span": None,
        }

    if n_agents is not None:
        kwargs["n_agents"] = n_agents
    valid = set(inspect.signature(MarlLobEnv).parameters) - {"self"}
    unknown = set(kwargs) - valid
    if unknown:
        raise ValueError(f"unknown environment_config keys: {sorted(unknown)}")
    if "config_kwargs" in kwargs and kwargs["config_kwargs"] is not None \
            and not isinstance(kwargs["config_kwargs"], dict):
        raise ValueError("config_kwargs must be an object")
    try:
        env = MarlLobEnv(**kwargs)
        actual_action = env.action_space(env.possible_agents[0])
        actual_obs = env.observation_space(env.possible_agents[0])
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"invalid environment configuration: {exc}") from exc
    if actual_obs.shape != observation_space.shape:
        raise ValueError(
            f"observation layout mismatch: checkpoint {observation_space.shape}, "
            f"environment {actual_obs.shape}"
        )
    if actual_action.shape != action_space.shape or \
            not np.allclose(actual_action.low, action_space.low) or \
            not np.allclose(actual_action.high, action_space.high):
        raise ValueError("action layout or bounds mismatch between checkpoint and environment")
    return kwargs
