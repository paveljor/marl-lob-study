"""One market transition must feed two independent PPO optimizers."""
from copy import deepcopy

import gymnasium as gym
import numpy as np
import pytest
import torch as th

from marl_lob.concurrent import ConcurrentPPOTrainer

CONFIG = {"n_agents": 2, "gated_actions": True, "offset_span": 15, "max_size": 10}


class FakeMarket:
    possible_agents = ["mm_0", "mm_1"]

    def __init__(self, *, termination=None, truncation=None, market_closed=False):
        self.termination = termination
        self.truncation = truncation
        self.market_closed = market_closed
        self.seeds = []
        self.actions = []
        self.t = 0

    def observation_space(self, agent):
        return gym.spaces.Box(-100, 100, shape=(3,), dtype=np.float32)

    def reset(self, seed):
        self.seeds.append(seed)
        self.t = 0
        return self._obs(), {a: {} for a in self.possible_agents}

    def _obs(self):
        return {
            "mm_0": np.array([self.t, 0, 0], np.float32),
            "mm_1": np.array([self.t, 1, 0], np.float32),
        }

    def step(self, actions):
        self.actions.append(deepcopy(actions))
        self.t += 1
        terms = {a: a == self.termination for a in self.possible_agents}
        truncs = {a: a == self.truncation for a in self.possible_agents}
        return self._obs(), {"mm_0": 3.0, "mm_1": -7.0}, terms, truncs, {
            a: {"market_closed": self.market_closed} for a in self.possible_agents
        }

    def close(self):
        pass


def trainer(market, **kwargs):
    return ConcurrentPPOTrainer(
        CONFIG, quote_size=5, seed=18, n_steps=4, batch_size=4, n_epochs=1,
        market=market, **kwargs,
    )


def test_joint_rollout_routes_each_observation_reward_and_action():
    th.set_num_threads(1)
    market = FakeMarket()
    run = trainer(market)
    snapshots = []
    run.train(4, lambda current, step: snapshots.append((step, current.normalizer_state())))
    assert len(market.actions) == 4
    assert snapshots[0][0] == 4
    assert run.episode_seeds == market.seeds
    assert len(run.episode_seeds) == 1
    for action_pair in market.actions:
        for action in action_pair.values():
            assert action.shape == (6,)
            np.testing.assert_array_equal(action[-2:], [5, 5])
    for agent, identity in (("mm_0", 0), ("mm_1", 1)):
        buffer = run.models[agent].rollout_buffer
        np.testing.assert_array_equal(buffer.observations[:, 1], identity)
        sign = 1 if agent == "mm_0" else -1
        assert np.all(buffer.rewards[:, 0] * sign > 0)
        assert np.isfinite(buffer.log_probs).all()
        assert np.isfinite(buffer.values).all()
        assert np.isfinite(buffer.returns).all()
    assert run.normalizer_state()["mm_0"]["mean"] != run.normalizer_state()["mm_1"]["mean"]


def test_both_optimizers_update_on_the_same_joint_rollout():
    th.set_num_threads(1)
    run = trainer(FakeMarket())
    initial = {
        agent: {key: value.detach().clone() for key, value in model.policy.state_dict().items()}
        for agent, model in run.models.items()
    }
    assert any(
        not th.equal(initial["mm_0"][key], initial["mm_1"][key])
        for key in initial["mm_0"]
    )
    run.train(4)
    for agent, model in run.models.items():
        assert model.num_timesteps == 4
        assert model._n_updates == 1
        assert any(not th.equal(value, initial[agent][key])
                   for key, value in model.policy.state_dict().items())
        other = "mm_1" if agent == "mm_0" else "mm_0"
        assert model.policy.optimizer is not run.models[other].policy.optimizer
        assert "train/value_loss" in run.diagnostics[0]["train"][agent]


def test_batch_boundary_bootstraps_continuing_trajectory():
    th.set_num_threads(1)
    run = trainer(FakeMarket())
    for model in run.models.values():
        model.policy.predict_values = lambda obs: th.full((len(obs), 1), 2.0, device=obs.device)
        model.train = lambda: None
    run.train(4)
    for model in run.models.values():
        buffer = model.rollout_buffer
        assert not buffer.episode_starts[1:, 0].any()
        expected = buffer.rewards[-1, 0] + model.gamma * 2.0
        assert buffer.returns[-1, 0] == pytest.approx(expected)


def test_unclipped_actions_and_matching_log_probs_are_buffered():
    th.set_num_threads(1)
    market = FakeMarket()
    run = trainer(market)
    forced_action = th.tensor([[2.0, -2.0, 0.5, -0.5]])
    for model in run.models.values():
        policy = model.policy

        def forward(obs, policy=policy):
            action = forced_action.to(obs.device)
            distribution = policy.get_distribution(obs)
            return action, policy.predict_values(obs), distribution.log_prob(action)

        policy.forward = forward
        model.train = lambda: None
    run.train(4)
    for agent, model in run.models.items():
        buffer = model.rollout_buffer
        np.testing.assert_allclose(buffer.actions[0, 0], forced_action.numpy()[0])
        np.testing.assert_allclose(market.actions[0][agent][:4], [1, -1, .5, -.5])
        obs = th.as_tensor(buffer.observations[0], device=model.device)
        with th.no_grad():
            raw_log_prob = model.policy.get_distribution(obs).log_prob(forced_action)
            clipped_log_prob = model.policy.get_distribution(obs).log_prob(
                th.clamp(forced_action, -1, 1)
            )
        assert buffer.log_probs[0, 0] == pytest.approx(float(raw_log_prob.item()))
        assert abs(buffer.log_probs[0, 0] - float(clipped_log_prob.item())) > 0.01


@pytest.mark.parametrize("termination,truncation,market_closed,legacy,bootstrap", [
    ("mm_1", None, False, False, False),
    (None, "mm_1", False, False, True),
    (None, "mm_1", True, False, False),
    (None, "mm_1", True, True, True),
    ("mm_0", "mm_1", True, True, False),
])
def test_joint_episode_end_and_time_limit_bootstrap(
    termination, truncation, market_closed, legacy, bootstrap,
):
    th.set_num_threads(1)
    market = FakeMarket(
        termination=termination, truncation=truncation, market_closed=market_closed,
    )
    run = trainer(market, bootstrap_at_market_close=legacy)
    # Constant terminal value makes the expected bootstrap contribution exact.
    for model in run.models.values():
        model.policy.predict_values = lambda obs: th.full((len(obs), 1), 2.0, device=obs.device)
    run.train(4)
    assert len(market.seeds) == 5  # each joint end resets both traders together
    assert run.diagnostics[0]["risk_ends"] == (4 if termination else 0)
    assert run.diagnostics[0]["market_close_ends"] == (
        4 if market_closed and not termination else 0
    )
    assert run.diagnostics[0]["episode_summaries"][0]["end_reason"] == (
        "risk" if termination else "market_close" if market_closed else "external_truncation"
    )
    assert len(run.diagnostics[0]["episode_summaries"]) == 4
    for agent, model in run.models.items():
        norm = run.normalizers[agent]
        raw = 3.0 if agent == "mm_0" else -7.0
        # After every one-step episode the running return statistic is reset.
        assert norm.returns[0] == 0
        base = raw / np.sqrt(norm.ret_rms.var + norm.epsilon)
        base = np.clip(base, -norm.clip_reward, norm.clip_reward)
        expected = base + (model.gamma * 2 if bootstrap else 0)
        assert model.rollout_buffer.rewards[-1, 0] == pytest.approx(expected)
        assert model.rollout_buffer.episode_starts[:, 0].all()


def test_validation_and_private_market_seed_stream():
    with pytest.raises(ValueError, match="size limits"):
        ConcurrentPPOTrainer(CONFIG, 11, 1, market=FakeMarket())
    with pytest.raises(ValueError, match="gated"):
        ConcurrentPPOTrainer(dict(CONFIG, gated_actions=False), 5, 1, market=FakeMarket())
    th.set_num_threads(1)
    a = trainer(FakeMarket(termination="mm_0"))
    b = trainer(FakeMarket(termination="mm_0"))
    np.random.seed(999)
    a.train(4)
    np.random.seed(12)
    b.train(4)
    assert a.episode_seeds == b.episode_seeds


@pytest.mark.abides
def test_short_abides_joint_training_smoke():
    th.set_num_threads(1)
    config = dict(
        CONFIG, end_time="09:30:10", agent_id_width=2,
        config_kwargs={"num_noise_agents": 50, "num_value_agents": 5},
    )
    run = ConcurrentPPOTrainer(config, 5, 42, n_steps=4, batch_size=4, n_epochs=1)
    try:
        run.train(4)
        assert run.joint_steps == 4
        assert len(run.episode_seeds) == 1
        assert all(model._n_updates == 1 for model in run.models.values())
        assert all(np.isfinite(model.rollout_buffer.returns).all()
                   for model in run.models.values())
    finally:
        run.close()
