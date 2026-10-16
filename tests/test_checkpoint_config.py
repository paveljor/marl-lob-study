"""Evaluation must use the same market contract as training."""
import json

import pytest

from marl_lob.checkpoint_config import evaluation_env_kwargs
from marl_lob.env import MarlLobEnv


def spaces(kwargs):
    env = MarlLobEnv(**kwargs)
    return env.action_space("mm_0"), env.observation_space("mm_0")


def restore(tmp_path, kwargs, *, n_agents=None):
    action, observation = spaces(kwargs)
    return evaluation_env_kwargs(
        tmp_path / "ppo.zip", action_space=action,
        observation_space=observation, n_agents=n_agents,
    )


def test_full_environment_config_is_restored(tmp_path):
    config = {
        "n_agents": 2, "max_inventory": 1234, "inventory_penalty": 0.02,
        "min_size": 7, "max_size": 40, "max_offset_cents": 21,
        "gated_actions": True, "agent_id_obs": False, "include_ofi": True,
        "offset_span": 15, "mid_rounding": "outward",
        "min_offset_ticks": 2, "min_quote_size": 3, "latency_edge": 0.5,
        "start_time": "09:31:00", "end_time": "09:32:00",
        "config_kwargs": {"num_value_agents": 12, "num_noise_agents": 75,
                          "num_momentum_agents": 3},
    }
    (tmp_path / "run_meta.json").write_text(json.dumps({
        "environment_config": config,
        "args": {"n_agents": 100},  # explicit environment config takes precedence
    }))
    assert restore(tmp_path, config) == config


def test_legacy_train_metadata_restores_population_and_action_limits(tmp_path):
    training = {
        "n_agents": 2, "max_inventory": 10_000, "inventory_penalty": 0.0,
        "min_size": 8, "max_offset_cents": 17, "gated_actions": True,
        "agent_id_obs": True, "agent_id_width": 2, "include_ofi": True,
        "latency_edge": 1.0, "offset_span": 11,
        "config_kwargs": {"num_value_agents": 5, "num_noise_agents": 60},
    }
    (tmp_path / "run_meta.json").write_text(json.dumps({
        "args": {
            "n_agents": 2, "inventory_penalty": 0.0, "min_size": 8,
            "max_offset_cents": 17, "legacy_actions": False,
            "no_agent_id_obs": False, "agent_id_width": 2, "ofi": True,
            "latency_edge": 1.0, "offset_span": 11,
            "num_value_agents": 5, "num_noise_agents": 60,
        },
        "config_kwargs": {"num_value_agents": 5, "num_noise_agents": 60},
    }))
    restored = restore(tmp_path, training)
    for key, value in training.items():
        assert restored[key] == value


def test_old_checkpoint_without_metadata_warns_and_remains_usable(tmp_path):
    legacy = {"n_agents": 2, "max_inventory": 10_000,
              "inventory_penalty": 0.0, "gated_actions": False,
              "agent_id_obs": False}
    with pytest.warns(UserWarning, match="No run_meta.json"):
        restored = restore(tmp_path, legacy)
    assert restored["agent_id_obs"] is False
    assert restored["gated_actions"] is False


def test_missing_metadata_cannot_guess_normalized_offset_span(tmp_path):
    normalized = {"n_agents": 2, "gated_actions": True, "offset_span": 15}
    with pytest.raises(ValueError, match="cannot be reconstructed"):
        restore(tmp_path, normalized)


def test_early_metadata_infers_absent_identity_flag(tmp_path):
    training = {"n_agents": 2, "max_inventory": 10_000,
                "inventory_penalty": 0.0, "gated_actions": False,
                "agent_id_obs": False,
                "config_kwargs": {"num_noise_agents": 45}}
    (tmp_path / "run_meta.json").write_text(json.dumps({
        "args": {"n_agents": 2, "legacy_actions": True,
                 "num_noise_agents": 45},
    }))
    restored = restore(tmp_path, training)
    assert restored["agent_id_obs"] is False
    assert restored["config_kwargs"] == {"num_noise_agents": 45}


def test_invalid_existing_metadata_is_an_error(tmp_path):
    (tmp_path / "run_meta.json").write_text("{broken")
    with pytest.raises(ValueError, match="invalid checkpoint metadata"):
        restore(tmp_path, {})


def test_mismatched_full_config_fails(tmp_path):
    (tmp_path / "run_meta.json").write_text(json.dumps({
        "environment_config": {"n_agents": 2, "gated_actions": True,
                               "min_size": 7},
    }))
    with pytest.raises(ValueError, match="bounds mismatch"):
        restore(tmp_path, {"n_agents": 2, "gated_actions": True,
                           "min_size": 0})
