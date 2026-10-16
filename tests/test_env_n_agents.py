"""The N-agent path was never exercised: every run used n_agents=2.

These are construction-only checks (MarlLobEnv builds no ABIDES kernel until
reset), so they need no `abides` marker and run in milliseconds.
"""
import numpy as np
import pytest

from marl_lob.env import MarlLobEnv
from marl_lob.observation_extractor import obs_vector_size


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_agent_set_scales_with_n_agents(n):
    env = MarlLobEnv(n_agents=n)
    assert env.n_agents == n
    assert env.possible_agents == [f"mm_{i}" for i in range(n)]
    assert len(set(env.possible_agents)) == n


@pytest.mark.parametrize("n", [1, 3])
def test_spaces_defined_for_every_agent(n):
    """Default layout: gated 6-tuple action, identity one-hot on the obs."""
    env = MarlLobEnv(n_agents=n)
    for agent in env.possible_agents:
        obs_space = env.observation_space(agent)
        act_space = env.action_space(agent)
        assert obs_space.shape == (obs_vector_size(env.k, n),)
        assert act_space.shape == (6,)


@pytest.mark.parametrize("n", [1, 3])
def test_legacy_layout_still_available(n):
    """The pre-2026-09 contract stays reachable so old work can be re-run."""
    env = MarlLobEnv(n_agents=n, gated_actions=False, agent_id_obs=False)
    for agent in env.possible_agents:
        assert env.observation_space(agent).shape == (obs_vector_size(env.k),)
        assert env.action_space(agent).shape == (4,)


def test_min_size_raises_action_low_on_size_dims_only():
    env = MarlLobEnv(n_agents=3, min_size=7)
    low = env.action_space("mm_0").low
    assert low[0] == -1.0 and low[1] == -1.0, "gate dims span [-1, 1]"
    assert low[2] == 0.0 and low[3] == 0.0, "offset dims must stay at 0"
    assert low[4] == 7.0 and low[5] == 7.0, "size dims must be raised to min_size"


def test_gates_are_centred_on_zero():
    """A fresh policy emits ~0, which must sit on the quote/no-quote boundary.

    If the gate range were [0, 1] the initial policy would quote on every step;
    if it were [-1, 0] it would never quote and the dead zone would be back.
    """
    env = MarlLobEnv(n_agents=2)
    space = env.action_space("mm_0")
    assert (space.low[0], space.high[0]) == (-1.0, 1.0)
    assert (space.low[1], space.high[1]) == (-1.0, 1.0)


def test_max_offset_cents_caps_action_high():
    env = MarlLobEnv(n_agents=2, max_offset_cents=12, max_size=40)
    high = env.action_space("mm_1").high
    assert np.allclose(high, [1.0, 1.0, 12.0, 12.0, 40.0, 40.0])


def test_agent_id_width_can_be_pinned_across_a_grid():
    """A grid varying n_agents must keep one observation width, or its cells
    are not comparable and every checkpoint has a different input layer."""
    widths = {
        MarlLobEnv(n_agents=n, agent_id_width=4).observation_space("mm_0").shape
        for n in (1, 2, 4)
    }
    assert widths == {(obs_vector_size(10, 4),)}


def test_agent_id_width_defaults_to_n_agents():
    env = MarlLobEnv(n_agents=3)
    assert env.agent_id_width == 3
    assert env.observation_space("mm_0").shape == (obs_vector_size(10, 3),)


def test_config_kwargs_are_retained_for_forwarding():
    env = MarlLobEnv(n_agents=2, config_kwargs={"num_value_agents": 150})
    assert env.config_kwargs["num_value_agents"] == 150
