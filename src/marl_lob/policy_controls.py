"""Inference policies for interventions with private, reproducible randomness."""

from __future__ import annotations

import numpy as np
import torch

from .independent import fixed_size_action


class GaussianPolicy:
    """Sample the trained diagonal Gaussian using a per-trader NumPy RNG.

    This is the SB3 Gaussian policy distribution, followed by the same action
    clipping used during training. Private RNGs prevent replacing B from
    changing A's random-number stream. Models must use raw observations and
    the four-dimensional unsquashed Box action interface.
    """

    def __init__(self, model, quote_size: int, seed: int, deterministic=False):
        if model.action_space.shape != (4,) or model.policy.squash_output or model.use_sde:
            raise ValueError("Requires unsquashed four-dimensional diagonal Gaussian PPO")
        self.model = model
        self.quote_size = quote_size
        self.rng = np.random.default_rng(seed)
        self.deterministic = deterministic
        self.model.policy.set_training_mode(False)

    def act(self, observation):
        with torch.no_grad():
            obs, _ = self.model.policy.obs_to_tensor(observation)
            normal = self.model.policy.get_distribution(obs).distribution
            mean = normal.mean.cpu().numpy()[0]
            std = normal.stddev.cpu().numpy()[0]
        action = mean if self.deterministic else mean + std * self.rng.standard_normal(4)
        return fixed_size_action(action, self.quote_size)


class MarginalPolicy:
    """Draw full action rows from a disjoint calibration pool, ignoring state.

    Preserves the empirical joint distribution of gates, offsets and sizes;
    breaks association with current observations and temporal dependencies.
    Does not promise exact participation matching on another market path.
    """

    def __init__(self, actions, seed):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 6 or len(actions) == 0:
            raise ValueError("Expected a nonempty pool of six-dimensional actions")
        if not np.isfinite(actions).all():
            raise ValueError("Nonfinite action pool")
        self.actions = actions.copy()
        self.rng = np.random.default_rng(seed)

    def act(self, observation):
        return self.actions[self.rng.integers(len(self.actions))].copy()
