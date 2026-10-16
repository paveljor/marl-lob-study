from copy import deepcopy

import gymnasium as gym
import numpy as np
import pytest

from marl_lob.actions import resolve_quote
from marl_lob.env import MarlLobEnv
from marl_lob.independent import ConstantPolicy, SingleLearnerEnv, fixed_size_action

CONFIG = {"n_agents": 2, "gated_actions": True, "offset_span": 15, "max_size": 100}


class FakeMarket:
    possible_agents = ["mm_0", "mm_1"]

    def __init__(self):
        self.seeds = []
        self.end_other = False
        self.truncate = False
        self.market_closed = False

    def observation_space(self, agent):
        return gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float32)

    def reset(self, seed):
        self.seeds.append(seed)
        return {"mm_0": np.zeros(3, np.float32), "mm_1": np.ones(3, np.float32)}, {
            "mm_0": {}, "mm_1": {}}

    def step(self, actions):
        self.last_actions = deepcopy(actions)
        obs = {"mm_0": np.zeros(3, np.float32), "mm_1": np.ones(3, np.float32)}
        return obs, {"mm_0": 7, "mm_1": -11}, {
            "mm_0": False, "mm_1": self.end_other}, {
            "mm_0": self.truncate, "mm_1": self.truncate}, {
            "mm_0": {"market_closed": True} if self.market_closed else {},
            "mm_1": {"market_closed": True} if self.market_closed else {},
        }

    def close(self):
        pass


def test_learner_reward_and_opponent_observation_stay_separate():
    market = FakeMarket()

    class Opponent:
        def act(self, observation):
            np.testing.assert_array_equal(observation, np.ones(3))
            return fixed_size_action(np.array([-1, 1, .5, -.5]), 3)

    env = SingleLearnerEnv(CONFIG, "mm_0", Opponent(), 10, 7, market)
    env.reset()
    _, reward, term, trunc, _ = env.step(np.array([1, -1, -.2, .2]))
    assert reward == 7
    assert not term and not trunc
    np.testing.assert_allclose(market.last_actions["mm_0"], [1, -1, -.2, .2, 10, 10])
    np.testing.assert_allclose(market.last_actions["mm_1"], [-1, 1, .5, -.5, 3, 3])


def test_either_risk_termination_ends_joint_training_episode():
    market = FakeMarket()
    market.end_other = True
    env = SingleLearnerEnv(CONFIG, "mm_0", ConstantPolicy(), 10, 7, market)
    env.reset()
    _, _, term, trunc, info = env.step(np.zeros(4))
    assert term and not trunc and info["joint_termination"]


@pytest.mark.parametrize("market_closed,expected", [
    (False, (False, True)), (True, (True, False)),
])
def test_market_close_is_finite_horizon_but_external_cutoff_is_truncation(
    market_closed, expected,
):
    market = FakeMarket()
    market.truncate = True
    market.market_closed = market_closed
    env = SingleLearnerEnv(CONFIG, "mm_0", ConstantPolicy(), 10, 7, market)
    env.reset()
    _, _, term, trunc, info = env.step(np.zeros(4))
    assert (term, trunc) == expected
    assert info.get("market_closed", False) is market_closed
    assert not info["joint_termination"]


def test_episode_seed_stream_reproducible_and_independent_of_global_rng():
    def sample(global_seed):
        env = SingleLearnerEnv(CONFIG, "mm_1", ConstantPolicy(), 10, 9, FakeMarket())
        for _ in range(3):
            np.random.seed(global_seed)
            env.reset()
        return env.episode_seeds
    assert sample(1) == sample(999)
    assert len(set(sample(1))) == 3


def test_fixed_size_keeps_withdrawal_available():
    action = fixed_size_action(np.array([-1, -1, 0, 0]), 10)
    quote = resolve_quote(tuple(action), doubled_mid=20000, gated=True, offset_span=15)
    assert quote.bid_qty == quote.ask_qty == 0


@pytest.mark.parametrize("ticks", [1, 5, 16])
def test_scripted_offset_uses_same_mapping_as_learner(ticks):
    action = ConstantPolicy(ticks, 10).act(np.zeros(1))
    q = resolve_quote(tuple(action), doubled_mid=20000, gated=True, offset_span=15)
    assert (q.bid_off_cents, q.ask_off_cents, q.bid_qty, q.ask_qty) == (ticks, ticks, 10, 10)


@pytest.mark.abides
def test_independent_training_does_not_update_frozen_opponent(tmp_path):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from marl_lob.independent import FrozenPPOPolicy

    torch.set_num_threads(1)
    config = dict(CONFIG, end_time="09:30:10", agent_id_width=2,
                  config_kwargs={"num_noise_agents": 50, "num_value_agents": 5})
    first_env = SingleLearnerEnv(config, "mm_0", ConstantPolicy(), 10, 1)
    first = PPO("MlpPolicy", first_env, seed=1, n_steps=8, batch_size=8, n_epochs=1)
    frozen = {k: v.clone() for k, v in first.policy.state_dict().items()}
    second_env = SingleLearnerEnv(config, "mm_1", FrozenPPOPolicy(first, 10), 10, 2)
    vec = VecNormalize(DummyVecEnv([lambda: second_env]), norm_obs=False)
    try:
        second = PPO("MlpPolicy", vec, seed=2, n_steps=8, batch_size=8, n_epochs=1)
        initial = {k: v.clone() for k, v in second.policy.state_dict().items()}
        second.learn(16)
        assert all(torch.equal(v, frozen[k]) for k, v in first.policy.state_dict().items())
        assert any(not torch.equal(v, initial[k])
                   for k, v in second.policy.state_dict().items())
        second.save(tmp_path / "second")
        loaded = PPO.load(tmp_path / "second")
        obs = np.zeros(second_env.observation_space.shape, dtype=np.float32)
        np.testing.assert_array_equal(second.predict(obs, deterministic=True)[0],
                                      loaded.predict(obs, deterministic=True)[0])
    finally:
        first_env.close()
        vec.close()


@pytest.mark.abides
@pytest.mark.parametrize("exhaust_kernel", [False, True])
def test_session_close_preserves_accounts_and_gross_fills(exhaust_kernel):
    env = MarlLobEnv(n_agents=2, end_time="09:30:10", offset_span=15,
                     config_kwargs={"num_noise_agents": 50, "num_value_agents": 5})
    try:
        _, infos = env.reset(seed=42)
        if exhaust_kernel:
            # Exercise the generic ABIDES terminal path before exchange close.
            env.kernel.stop_time = env._mkt_open_ns + 5_000_000_000
        initial = {a: infos[a]["traj_row"] for a in env.possible_agents}
        all_fills = {a: [] for a in env.possible_agents}
        rewards_sum = dict.fromkeys(env.possible_agents, 0.0)
        for _ in range(30):
            actions = {a: ConstantPolicy(active=False).act(np.zeros(1)) for a in env.agents}
            _, rewards, _, truncs, infos = env.step(actions)
            for a in env.possible_agents:
                all_fills[a].extend(infos[a]["executions"])
                rewards_sum[a] += rewards[a]
            if any(truncs.values()):
                break
        else:
            pytest.fail("short simulation did not reach terminal path")
        for a in env.possible_agents:
            last = infos[a]["traj_row"]
            assert infos[a]["market_closed"] is not exhaust_kernel
            assert last[0] >= initial[a][0]
            assert last[1] == 0
            assert last[2] == initial[a][2] == env.starting_cash
            assert not all_fills[a]
            assert rewards_sum[a] == 0
            assert last[0] < 11  # no one-minute post-close training buffer
    finally:
        env.close()
