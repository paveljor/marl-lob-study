import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from marl_lob.independent import fixed_size_action
from marl_lob.policy_controls import GaussianPolicy, MarginalPolicy


class SpacesOnly(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (3,), np.float32)
    action_space = gym.spaces.Box(-1, 1, (4,), np.float32)


def test_private_sampling_matches_distribution_and_ignores_other_rngs():
    torch.set_num_threads(1)
    model = PPO("MlpPolicy", SpacesOnly(), seed=123, n_steps=8, batch_size=8)
    obs = np.zeros(3, np.float32)
    p = GaussianPolicy(model, 10, 42)
    with torch.no_grad():
        normal = model.policy.get_distribution(torch.tensor(obs[None])).distribution
        expected = normal.mean.numpy()[0] + normal.stddev.numpy()[0] * (
            np.random.default_rng(42).standard_normal(4)
        )
    np.testing.assert_array_equal(p.act(obs), fixed_size_action(expected, 10))
    p1, p2 = GaussianPolicy(model, 10, 9), GaussianPolicy(model, 10, 9)
    for _ in range(10):
        act = p1.act(obs)
        torch.randn(200)
        np.random.randn(100)
        np.testing.assert_array_equal(act, p2.act(obs))
    mean = GaussianPolicy(model, 10, 5, deterministic=True).act(obs)
    np.testing.assert_array_equal(
        mean, fixed_size_action(model.predict(obs, deterministic=True)[0], 10)
    )


def test_marginal_control_preserves_whole_rows_and_ignores_observations():
    actions = np.array([[1, -1, -0.5, 0.5, 10, 10], [-1, 1, 1, -1, 10, 10]])
    a, b = MarginalPolicy(actions, 42), MarginalPolicy(actions, 42)
    outputs = []
    for i in range(30):
        out = a.act(np.ones(3) * i)
        np.testing.assert_array_equal(out, b.act(np.zeros(3)))
        assert any(np.array_equal(out, row) for row in actions)
        outputs.append(tuple(out))
    assert len(set(outputs)) == 2
