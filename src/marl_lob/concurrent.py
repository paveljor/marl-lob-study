"""Independent PPO learners acting together in one shared market rollout."""
from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.running_mean_std import RunningMeanStd

from .env import MarlLobEnv
from .independent import fixed_size_action


class _SpacesOnlyEnv(gym.Env):
    """Supply SB3 with spaces; the joint market owns every transition."""

    metadata = {"render_modes": []}

    def __init__(self, observation_space: gym.Space):
        self.observation_space = observation_space
        self.action_space = gym.spaces.Box(-1, 1, shape=(4,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        raise RuntimeError("The concurrent trainer steps the joint market directly")


class _RewardNormalizer:
    """The reward path of VecNormalize, kept separately for each account."""

    def __init__(self, gamma: float, clip_reward: float = 10.0, epsilon: float = 1e-8):
        self.gamma = gamma
        self.clip_reward = clip_reward
        self.epsilon = epsilon
        self.returns = np.zeros(1, dtype=np.float64)
        self.ret_rms = RunningMeanStd(shape=())

    def normalize(self, reward: float, done: bool) -> float:
        self.returns = self.returns * self.gamma + reward
        self.ret_rms.update(self.returns)
        result = np.clip(
            reward / np.sqrt(self.ret_rms.var + self.epsilon),
            -self.clip_reward,
            self.clip_reward,
        )
        if done:
            self.returns.fill(0)
        return float(result)

    def state(self) -> dict:
        return {
            "mean": float(self.ret_rms.mean),
            "var": float(self.ret_rms.var),
            "count": float(self.ret_rms.count),
            "returns": float(self.returns[0]),
            "gamma": self.gamma,
            "clip_reward": self.clip_reward,
            "epsilon": self.epsilon,
        }


class ConcurrentPPOTrainer:
    """Train two independent PPO policies on the same joint ABIDES trajectory.

    ``total_steps`` counts joint market transitions. Like SB3 PPO, training
    finishes a whole ``n_steps`` rollout when the target falls inside one.
    The checkpoint callback receives ``(trainer, joint_steps)`` after both
    optimizers finish each rollout.
    """

    agents = ("mm_0", "mm_1")

    def __init__(
        self,
        environment_config: dict,
        quote_size: int,
        seed: int,
        *,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        bootstrap_at_market_close: bool = False,
        market=None,
        **ppo_kwargs,
    ):
        self.environment_config = deepcopy(environment_config)
        if (
            not environment_config.get("gated_actions", True)
            or not environment_config.get("offset_span")
        ):
            raise ValueError("Concurrent training requires gated, rescaled actions")
        if not 1 <= quote_size <= environment_config.get("max_size", 100):
            raise ValueError("quote_size outside environment size limits")
        if n_steps < 2 or batch_size < 2 or n_epochs < 1:
            raise ValueError("n_steps and batch_size must be >= 2; n_epochs must be positive")
        self.market = market if market is not None else MarlLobEnv(**environment_config)
        if list(self.market.possible_agents) != list(self.agents):
            raise ValueError("Concurrent training requires exactly mm_0 and mm_1")
        self.quote_size = quote_size
        self.n_steps = n_steps
        self.bootstrap_at_market_close = bootstrap_at_market_close
        self._market_seeds = np.random.default_rng(seed)
        self.episode_seeds: list[int] = []
        self.diagnostics: list[dict] = []
        self._episode_rewards = dict.fromkeys(self.agents, 0.0)
        self.joint_steps = 0
        self._observations = None
        self._episode_start = True
        self.models: dict[str, PPO] = {}
        self.normalizers: dict[str, _RewardNormalizer] = {}
        for index, agent in enumerate(self.agents):
            spaces_env = _SpacesOnlyEnv(self.market.observation_space(agent))
            model = PPO(
                "MlpPolicy", spaces_env, seed=seed + index, n_steps=n_steps,
                batch_size=batch_size, n_epochs=n_epochs,
                learning_rate=learning_rate, gamma=gamma, verbose=0,
                **ppo_kwargs,
            )
            model.set_logger(configure(folder=None, format_strings=[]))
            self.models[agent] = model
            self.normalizers[agent] = _RewardNormalizer(gamma)

    def _reset_market(self) -> None:
        market_seed = int(self._market_seeds.integers(0, 2**31 - 1))
        self.episode_seeds.append(market_seed)
        self._observations, _ = self.market.reset(seed=market_seed)
        self._episode_start = True

    @staticmethod
    def _value(model: PPO, observation: np.ndarray) -> th.Tensor:
        tensor, _ = model.policy.obs_to_tensor(observation)
        with th.no_grad():
            return model.policy.predict_values(tensor)

    def normalizer_state(self) -> dict[str, dict]:
        """Serializable reward statistics for paired checkpoint metadata."""
        return {agent: self.normalizers[agent].state() for agent in self.agents}

    def train(
        self,
        total_steps: int,
        checkpoint_callback: Callable[[ConcurrentPPOTrainer, int], None] | None = None,
    ) -> dict[str, PPO]:
        if total_steps < 1:
            raise ValueError("total_steps must be positive")
        if self._observations is None:
            self._reset_market()
        target = self.joint_steps + total_steps
        for model in self.models.values():
            model._total_timesteps = target

        while self.joint_steps < target:
            for model in self.models.values():
                model.policy.set_training_mode(False)
                model.rollout_buffer.reset()
                if model.use_sde:
                    model.policy.reset_noise(1)
            batch_rewards = dict.fromkeys(self.agents, 0.0)
            batch_normalized_rewards = dict.fromkeys(self.agents, 0.0)
            batch_episodes = 0
            batch_risk_ends = 0
            batch_time_ends = 0
            episode_summaries = []
            done = False

            for rollout_step in range(self.n_steps):
                assert self._observations is not None
                actions = {}
                samples = {}
                for agent, model in self.models.items():
                    if model.use_sde and model.sde_sample_freq > 0:
                        if rollout_step % model.sde_sample_freq == 0:
                            model.policy.reset_noise(1)
                    observation = np.asarray(self._observations[agent], dtype=np.float32)
                    tensor, _ = model.policy.obs_to_tensor(observation)
                    with th.no_grad():
                        action, value, log_prob = model.policy(tensor)
                    raw_action = action.cpu().numpy()[0]
                    actions[agent] = fixed_size_action(raw_action, self.quote_size)
                    samples[agent] = (observation, raw_action, value, log_prob)

                next_obs, rewards, terms, truncs, infos = self.market.step(actions)
                risk_end = any(terms.values())
                time_end = any(truncs.values())
                market_closed = any(info.get("market_closed", False) for info in infos.values())
                done = risk_end or time_end
                for agent, model in self.models.items():
                    observation, raw_action, value, log_prob = samples[agent]
                    reward = self.normalizers[agent].normalize(float(rewards[agent]), done)
                    # External cutoffs bootstrap like SB3 TimeLimit. Exchange
                    # close ends the finite trading task unless legacy mode is set.
                    if time_end and not risk_end and (
                        not market_closed or self.bootstrap_at_market_close
                    ):
                        terminal_value = self._value(model, next_obs[agent])
                        reward += model.gamma * float(terminal_value.item())
                    model.rollout_buffer.add(
                        observation[None], raw_action[None],
                        np.array([reward], dtype=np.float32),
                        np.array([self._episode_start]), value, log_prob,
                    )
                    batch_rewards[agent] += float(rewards[agent])
                    self._episode_rewards[agent] += float(rewards[agent])
                    batch_normalized_rewards[agent] += reward
                    model.num_timesteps += 1
                self.joint_steps += 1
                self._episode_start = done
                if done:
                    batch_episodes += 1
                    batch_risk_ends += int(risk_end)
                    batch_time_ends += int(market_closed and not risk_end)
                    episode_summaries.append({
                        "seed": self.episode_seeds[-1],
                        "end_reason": (
                            "risk" if risk_end else
                            "market_close" if market_closed else "external_truncation"
                        ),
                        "raw_reward": self._episode_rewards.copy(),
                        "final_inventory": {
                            agent: int(infos[agent]["traj_row"][1])
                            if "traj_row" in infos[agent] else None
                            for agent in self.agents
                        },
                    })
                    self._episode_rewards = dict.fromkeys(self.agents, 0.0)
                    self._reset_market()
                else:
                    self._observations = next_obs

            for agent, model in self.models.items():
                assert self._observations is not None
                last_value = self._value(model, self._observations[agent])
                model.rollout_buffer.compute_returns_and_advantage(
                    last_values=last_value, dones=np.array([done]),
                )
            # Both buffers are now complete. No policy is updated while its
            # counterpart is still collecting this shared rollout.
            for model in self.models.values():
                model._update_current_progress_remaining(self.joint_steps, target)
                model.train()
            train_scalars = {
                agent: {
                    key: float(value)
                    for key, value in model.logger.name_to_value.items()
                    if key.startswith("train/") and np.isscalar(value)
                }
                for agent, model in self.models.items()
            }
            self.diagnostics.append({
                "joint_steps": self.joint_steps,
                "episodes": batch_episodes,
                "risk_ends": batch_risk_ends,
                "market_close_ends": batch_time_ends,
                "episode_summaries": episode_summaries,
                "raw_reward": batch_rewards,
                "normalized_reward": batch_normalized_rewards,
                "train": train_scalars,
            })
            if checkpoint_callback is not None:
                checkpoint_callback(self, self.joint_steps)
        return self.models

    def close(self) -> None:
        self.market.close()
