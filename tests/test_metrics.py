from __future__ import annotations

import numpy as np

from marl_lob.metrics import (
    compute_all,
    compute_inventory_distribution,
    compute_max_drawdown,
    compute_pnl_curve,
    compute_sharpe,
)
from tests.fixtures.synthetic_trajectories import (
    constant_traj,
    empty_traj,
    linear_growth_traj,
    triangle_equity_traj,
)


def test_constant_trajectory_is_flat():
    traj = constant_traj()
    assert compute_pnl_curve(traj).tolist() == [0] * len(traj)
    assert compute_sharpe(traj) == 0.0
    mdd, peak, trough = compute_max_drawdown(traj)
    assert mdd == 0.0
    assert peak == 0 and trough == 0


def test_linear_growth_has_zero_drawdown_and_positive_sharpe():
    traj = linear_growth_traj()
    pnl = compute_pnl_curve(traj)
    assert pnl[0] == 0
    assert pnl[-1] > 0
    assert np.all(np.diff(pnl) >= 0)
    mdd, _peak, _trough = compute_max_drawdown(traj)
    assert mdd == 0.0
    sharpe = compute_sharpe(traj)
    assert np.isfinite(sharpe) and sharpe > 0


def test_triangle_drawdown_indices_match():
    peak_idx, trough_idx = 5, 10
    traj = triangle_equity_traj(peak_idx=peak_idx, trough_idx=trough_idx)
    eq = traj.equity()
    expected_mdd = (eq[peak_idx] - eq[trough_idx]) / eq[peak_idx]
    mdd, peak, trough = compute_max_drawdown(traj)
    assert peak == peak_idx
    assert trough == trough_idx
    assert mdd == expected_mdd


def test_empty_trajectory_returns_sentinels():
    traj = empty_traj()
    assert compute_pnl_curve(traj).tolist() == []
    assert compute_sharpe(traj) == 0.0
    mdd, peak, trough = compute_max_drawdown(traj)
    assert (mdd, peak, trough) == (0.0, 0, 0)
    assert compute_inventory_distribution(traj).tolist() == []


def test_inventory_distribution_returns_copy():
    traj = constant_traj()
    inv = compute_inventory_distribution(traj)
    inv[0] = 999
    # underlying trajectory unchanged
    assert traj.inventory[0] == 0


def test_compute_all_keys():
    traj = triangle_equity_traj()
    out = compute_all(traj)
    assert set(out.keys()) == {
        "pnl_curve",
        "final_pnl",
        "sharpe",
        "max_drawdown",
        "max_drawdown_peak_idx",
        "max_drawdown_trough_idx",
        "inventory",
        "n_fills",
    }
    assert out["max_drawdown"] > 0
