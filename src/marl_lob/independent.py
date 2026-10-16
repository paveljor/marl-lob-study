"""One learner per optimizer, with other traders controlled by frozen policies."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Protocol

import gymnasium as gym
import numpy as np

from .env import MarlLobEnv


class TradingPolicy(Protocol):
    def act(self, observation: np.ndarray) -> np.ndarray: ...


def fixed_size_action(action: np.ndarray, quote_size: int) -> np.ndarray:
    """Four normalized choices -> six simulator fields, with size held fixed.

    Participation remains optional; fixed size applies only to open gates.
    This avoids presenting PPO with a size dimension that mostly rounds to 1.
    """
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (4,) or not np.isfinite(action).all():
        raise ValueError("Expected four finite gate/offset outputs")
    if quote_size < 1:
        raise ValueError("quote_size must be positive")
    return np.concatenate((np.clip(action, -1, 1), [quote_size, quote_size])).astype(
        np.float32
    )


@dataclass
class ConstantPolicy:
    offset_ticks: int = 5
    quote_size: int = 10
    offset_span: int = 15
    active: bool = True

    def act(self, observation: np.ndarray) -> np.ndarray:
        if not 1 <= self.offset_ticks <= 1 + self.offset_span:
            raise ValueError("Scripted offset outside the configured span")
        raw = 2 * (self.offset_ticks - 1) / self.offset_span - 1
        gate = 1.0 if self.active else -1.0
        return fixed_size_action(np.array([gate, gate, raw, raw]), self.quote_size)


@dataclass
class FrozenPPOPolicy:
    """Inference only. The model has its own weights and never receives rewards.

    The independent trainer uses raw scaled observations (norm_obs=False).
    Reward normalization is training-only and therefore absent at inference.
    """
    model: object
    quote_size: int

    def act(self, observation: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(observation, deterministic=True)
        return fixed_size_action(action, self.quote_size)


class SingleLearnerEnv(gym.Env):
    """Gym view of one account in a two-trader shared market.

    All policies act on their own *pre-step* observation. Only the selected
    trader's reward reaches its optimizer. End the joint episode on either
    trader's risk-limit breach, avoiding post-termination outstanding orders.
    """
    metadata = {"render_modes": []}

    def __init__(self, environment_config: dict, learner: str,
                 opponent: TradingPolicy, quote_size: int, seed: int,
                 market=None):
        super().__init__()
        self.environment_config = deepcopy(environment_config)
        self.market = market if market is not None else MarlLobEnv(**environment_config)
        if self.market.possible_agents != ["mm_0", "mm_1"]:
            raise ValueError("Independent training currently requires two traders")
        if learner not in self.market.possible_agents:
            raise ValueError(f"Unknown learner {learner}")
        if (not environment_config.get("gated_actions", True)
                or not environment_config.get("offset_span")):
            raise ValueError("Independent training requires gated, rescaled actions")
        if not 1 <= quote_size <= environment_config.get("max_size", 100):
            raise ValueError("quote_size outside environment size limits")
        self.learner = learner
        self.other = next(a for a in self.market.possible_agents if a != learner)
        self.opponent = opponent
        self.quote_size = quote_size
        self.action_space = gym.spaces.Box(-1, 1, shape=(4,), dtype=np.float32)
        self.observation_space = self.market.observation_space(learner)
        self._market_seeds = np.random.default_rng(seed)
        self.episode_seeds: list[int] = []
        self._observations = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._market_seeds = np.random.default_rng(seed)
        market_seed = int(self._market_seeds.integers(0, 2**31 - 1))
        self.episode_seeds.append(market_seed)
        self._observations, infos = self.market.reset(seed=market_seed)
        return self._observations[self.learner], infos[self.learner]

    def step(self, action):
        actions = {
            self.learner: fixed_size_action(action, self.quote_size),
            self.other: self.opponent.act(self._observations[self.other]),
        }
        self._observations, rewards, terms, truncs, infos = self.market.step(actions)
        terminated, truncated = any(terms.values()), any(truncs.values())
        info = dict(infos[self.learner])
        info["joint_termination"] = terminated
        if any(agent_info.get("market_closed", False) for agent_info in infos.values()):
            terminated = True
            truncated = False
        return (self._observations[self.learner], rewards[self.learner],
                terminated, truncated, info)

    def close(self):
        self.market.close()
