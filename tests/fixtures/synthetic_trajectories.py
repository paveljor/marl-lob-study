"""Synthetic trajectories with closed-form expected metrics."""
from __future__ import annotations

import numpy as np

from marl_lob.trajectory import Trajectory


def constant_traj(n: int = 100, mid: int = 10_000, cash: int = 1_000_000) -> Trajectory:
    """Flat: zero inventory, constant cash, constant mid. Equity is flat."""
    return Trajectory(
        timestamps=np.arange(n, dtype=float),
        inventory=np.zeros(n, dtype=np.int64),
        cash=np.full(n, cash, dtype=np.int64),
        mid_price=np.full(n, mid, dtype=np.int64),
        fills=[],
    )


def linear_growth_traj(n: int = 100, start: int = 1_000_000, step: int = 100) -> Trajectory:
    """Cash rises linearly; zero inventory; constant mid. Equity is linear, MDD = 0."""
    return Trajectory(
        timestamps=np.arange(n, dtype=float),
        inventory=np.zeros(n, dtype=np.int64),
        cash=np.array([start + i * step for i in range(n)], dtype=np.int64),
        mid_price=np.full(n, 10_000, dtype=np.int64),
        fills=[],
    )


def triangle_equity_traj(peak_idx: int = 5, trough_idx: int = 10, n: int = 16) -> Trajectory:
    """Equity rises to a peak then falls to a trough, then rises again.

    All movement is in `cash` (zero inventory) so equity == cash. The trough
    drawdown vs the peak has a known fraction.
    """
    base = 1_000_000
    cash = np.full(n, base, dtype=np.int64)
    for i in range(1, peak_idx + 1):
        cash[i] = base + i * 1000  # 1000-cent gain per step up to peak
    for i in range(peak_idx + 1, trough_idx + 1):
        cash[i] = cash[peak_idx] - (i - peak_idx) * 2000  # 2000-cent loss per step
    for i in range(trough_idx + 1, n):
        cash[i] = cash[trough_idx] + (i - trough_idx) * 500
    return Trajectory(
        timestamps=np.arange(n, dtype=float),
        inventory=np.zeros(n, dtype=np.int64),
        cash=cash,
        mid_price=np.full(n, 10_000, dtype=np.int64),
        fills=[],
    )


def empty_traj() -> Trajectory:
    return Trajectory.from_tuples([])
