"""Tests for MarlLobEnv.

Two layers:

- Pure-helper tests (always fast): equity arithmetic + reward formula.
  Pin the formulas in isolation so a sign error fails fast without
  spinning up the kernel.
- Integration tests (marked `abides`): drive a real ABIDES kernel
  through reset/step and check shapes, dtypes, fill production,
  termination handling, and PettingZoo API conformance.
"""
from __future__ import annotations

import numpy as np
import pytest

from marl_lob.env import MarlLobEnv
from marl_lob.observation_extractor import obs_vector_size

# ── _equity_from_traj_row ───────────────────────────────────────────────────

def test_equity_long_inventory():
    # row = (ts, inv, cash, mid, fill_qty, fill_price)
    row = (1.0, 10, 1_000_000, 100_000, 0, 0)
    # cash + inv*mid = 1_000_000 + 10*100_000 = 2_000_000
    assert MarlLobEnv._equity_from_traj_row(row) == 2_000_000


def test_equity_short_inventory():
    row = (1.0, -5, 1_500_000, 100_000, 0, 0)
    # 1_500_000 + (-5)*100_000 = 1_000_000
    assert MarlLobEnv._equity_from_traj_row(row) == 1_000_000


def test_equity_zero_inventory_is_just_cash():
    row = (1.0, 0, 9_999_999, 100_000, 0, 0)
    assert MarlLobEnv._equity_from_traj_row(row) == 9_999_999


# ── _compute_reward ─────────────────────────────────────────────────────────

def test_reward_pure_pnl_no_inventory_no_termination():
    """No inventory, no termination → reward is just ΔPnL."""
    r = MarlLobEnv._compute_reward(
        prev_equity=1_000_000, equity=1_000_500,
        inventory=0, inventory_penalty=1e-4,
        terminated=False, termination_penalty=-1.0,
    )
    assert r == 500.0


def test_reward_inventory_penalty_subtracts():
    """ΔPnL=0, inv=100, λ=1e-4 → reward = -0.01."""
    r = MarlLobEnv._compute_reward(
        prev_equity=1_000_000, equity=1_000_000,
        inventory=100, inventory_penalty=1e-4,
        terminated=False, termination_penalty=-1.0,
    )
    assert r == -0.01


def test_reward_inventory_penalty_symmetric_long_vs_short():
    """λ·|inv| is symmetric: long 50 and short 50 incur the same penalty."""
    common = dict(prev_equity=0, equity=0, inventory_penalty=1e-3,
                  terminated=False, termination_penalty=-1.0)
    long = MarlLobEnv._compute_reward(inventory=50, **common)
    short = MarlLobEnv._compute_reward(inventory=-50, **common)
    assert long == short == -0.05


def test_reward_long_inventory_with_price_up_is_positive_pnl_minus_penalty():
    """Held 10 long, mid moved 50 cents in our favor → +500 ΔPnL minus penalty."""
    r = MarlLobEnv._compute_reward(
        prev_equity=1_000_000, equity=1_000_500,   # +500 cents
        inventory=10, inventory_penalty=1.0,        # large λ for visibility
        terminated=False, termination_penalty=-1.0,
    )
    # 500 - 1.0*10 = 490
    assert r == 490.0


def test_reward_termination_adds_penalty():
    """Termination penalty stacks on top of the ΔPnL/penalty arithmetic."""
    r = MarlLobEnv._compute_reward(
        prev_equity=1_000_000, equity=1_000_000,
        inventory=0, inventory_penalty=1e-4,
        terminated=True, termination_penalty=-1.0,
    )
    assert r == -1.0


def test_reward_termination_with_pnl_and_penalty():
    """All three terms compose additively."""
    r = MarlLobEnv._compute_reward(
        prev_equity=1_000_000, equity=999_500,    # -500 ΔPnL
        inventory=200, inventory_penalty=1e-3,    # -0.2 penalty
        terminated=True, termination_penalty=-5.0,
    )
    # -500 - 0.2 - 5.0 = -505.2
    assert r == -505.2


# ── Integration: real ABIDES kernel under the env ───────────────────────────
# These spin up RMSC03 + coordinator + children for real. ~1-3s each.

@pytest.fixture
def env():
    e = MarlLobEnv(n_agents=2)
    yield e
    e.close()


@pytest.mark.abides
def test_reset_returns_correct_obs_shapes_and_info_keys(env):
    obs, info = env.reset(seed=42)
    assert set(obs.keys()) == {"mm_0", "mm_1"}
    for v in obs.values():
        # 4*K+4 book/scalar features plus the n_agents-wide identity one-hot.
        assert v.shape == (obs_vector_size(env.k, env.n_agents),)
        assert v.dtype == np.float32
    for v in info.values():
        assert "traj_row" in v
        assert len(v["traj_row"]) == 6


@pytest.mark.abides
def test_traj_row_dtypes_match_trajectory_contract(env):
    obs, info = env.reset(seed=42)
    # Gated layout: both gates open, 2c each side, 5 shares.
    acts = {a: np.array([1.0, 1.0, 2.0, 2.0, 5.0, 5.0], dtype=np.float32)
            for a in env.agents}
    _obs, _r, _t, _tr, info = env.step(acts)
    ts, inv, cash, mid, fill_qty, fill_price = info["mm_0"]["traj_row"]
    assert isinstance(ts, float)
    assert isinstance(inv, int) and isinstance(cash, int)
    assert isinstance(mid, int) and isinstance(fill_qty, int)
    assert isinstance(fill_price, int)


@pytest.mark.abides
def test_random_policy_100_steps_produces_fills(env):
    """Smoke: 100 random steps run cleanly and at least some fills happen."""
    rng = np.random.default_rng(0)
    obs, info = env.reset(seed=42)
    fills_seen = {a: 0 for a in env.possible_agents}
    for _ in range(100):
        acts = {
            a: rng.uniform(env._act_space.low, env._act_space.high).astype(np.float32)
            for a in env.agents
        }
        obs, rewards, terms, truncs, info = env.step(acts)
        for a in env.possible_agents:
            if info[a]["traj_row"][4] != 0:
                fills_seen[a] += 1
        if any(truncs.values()):
            break
    assert sum(fills_seen.values()) > 0, "no fills in 100 random-policy steps"


@pytest.mark.abides
def test_terminated_agent_drops_out_of_active_set():
    """Force-spike mm_0's inventory; it should leave self.agents."""
    e = MarlLobEnv(n_agents=2, max_inventory=20)
    obs, info = e.reset(seed=42)
    # Aggressive bid on mm_0 — bid gate open at the tightest offset and full
    # size, ask gate shut, so inventory can only accumulate long. mm_1 keeps
    # both gates shut and quotes nothing.
    aggressive = np.array([1.0, -1.0, 0.0, 50.0, 100.0, 0.0], dtype=np.float32)
    passive = np.array([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for _ in range(20):
        if "mm_0" not in e.agents:
            break
        acts = {"mm_0": aggressive, "mm_1": passive}
        # Only pass actions for still-active agents.
        acts = {a: acts[a] for a in e.agents}
        obs, rewards, terms, truncs, info = e.step(acts)
        if any(truncs.values()):
            pytest.fail("kernel truncated before mm_0 termination")
    assert "mm_0" not in e.agents, "mm_0 never terminated despite forced spike"
    assert "mm_1" in e.agents, "mm_1 should still be active"
    e.close()


@pytest.mark.abides
def test_pettingzoo_parallel_api_conformance():
    """PettingZoo's official API check: spaces, dtypes, dict-shape conformance."""
    from pettingzoo.test import parallel_api_test
    e = MarlLobEnv(n_agents=2)
    parallel_api_test(e, num_cycles=50)
    e.close()


def test_background_logging_is_off_by_default():
    """rmsc03 enables order and book logging for every background agent, and
    nothing here reads any of it - observations come from the market-data
    subscription and metrics from traj_row. Leaving it on cost ~26% of wall
    time at full episode length (see configs/rmsc03_simple.py)."""
    from marl_lob.configs import rmsc03_simple
    captured = {}

    class _Stub:
        @staticmethod
        def build_config(**kw):
            captured.update(kw)
            return {"agents": []}

    real = rmsc03_simple.rmsc03
    rmsc03_simple.rmsc03 = _Stub
    try:
        rmsc03_simple.build_config()
        assert captured["exchange_log_orders"] is False
        assert captured["log_orders"] is False
        assert captured["book_logging"] is False

        captured.clear()
        rmsc03_simple.build_config(log_orders=True, book_logging=True)
        assert captured["log_orders"] is True, "must stay overridable"
        assert captured["book_logging"] is True
    finally:
        rmsc03_simple.rmsc03 = real
