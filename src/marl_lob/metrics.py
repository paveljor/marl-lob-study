"""Pure metric functions over a Trajectory. No ABIDES, no PettingZoo.

All amounts in cents; convert to dollars at the call site if you must display them.
"""
from __future__ import annotations

import numpy as np

from .trajectory import Trajectory

# A standard equities trading day: 6.5 hours × 3600 s × 252 trading days.
TRADING_SECONDS_PER_YEAR = 23_400 * 252


def compute_pnl_curve(traj: Trajectory) -> np.ndarray:
    """Mark-to-market P&L curve in cents: equity[t] - equity[0]."""
    if len(traj) == 0:
        return np.zeros(0, dtype=np.int64)
    eq = traj.equity()
    return eq - eq[0]


def compute_sharpe(
    traj: Trajectory,
    dt_seconds: float = 1.0,
    trading_seconds_per_year: float = TRADING_SECONDS_PER_YEAR,
) -> float:
    """Annualized Sharpe over per-step equity returns.

    Step return at t = (equity[t+1] - equity[t]) / equity[t], with a guard for
    zero equity (those steps contribute zero to mean and variance). Returns 0.0
    rather than NaN when the return series has zero variance — useful default
    for empty or constant trajectories.
    """
    if len(traj) < 2:
        return 0.0
    eq = traj.equity().astype(np.float64)
    denom = eq[:-1]
    diffs = np.diff(eq)
    valid = denom != 0
    if not np.any(valid):
        return 0.0
    rets = np.zeros_like(diffs)
    rets[valid] = diffs[valid] / denom[valid]
    std = float(rets.std(ddof=0))
    if std == 0.0:
        return 0.0
    mean = float(rets.mean())
    annualization = np.sqrt(trading_seconds_per_year / dt_seconds)
    return (mean / std) * float(annualization)


def compute_max_drawdown(traj: Trajectory) -> tuple[float, int, int]:
    """Maximum drawdown as a positive fraction, with diagnostic indices.

    Returns (mdd_fraction, peak_idx, trough_idx). `mdd_fraction` is in [0, 1].
    For non-positive running peaks the drawdown is reported as 0 at that step
    (we don't want to divide by zero). For empty / single-step trajectories we
    return (0.0, 0, 0) — defined sentinels rather than crashes.
    """
    n = len(traj)
    if n < 2:
        return 0.0, 0, 0
    eq = traj.equity().astype(np.float64)
    running_peak = np.maximum.accumulate(eq)
    safe_peak = np.where(running_peak > 0, running_peak, 1.0)
    drawdowns = np.where(running_peak > 0, (running_peak - eq) / safe_peak, 0.0)
    trough_idx = int(np.argmax(drawdowns))
    mdd = float(drawdowns[trough_idx])
    peak_idx = int(np.argmax(eq[: trough_idx + 1])) if trough_idx > 0 else 0
    return mdd, peak_idx, trough_idx


def compute_inventory_distribution(traj: Trajectory) -> np.ndarray:
    """Raw inventory time series. Caller histograms with np.histogram if needed."""
    return np.asarray(traj.inventory, dtype=np.int64).copy()


def compute_all(
    traj: Trajectory,
    dt_seconds: float = 1.0,
    trading_seconds_per_year: float = TRADING_SECONDS_PER_YEAR,
) -> dict:
    """Convenience aggregator for training callbacks (Pavel's E)."""
    pnl = compute_pnl_curve(traj)
    mdd, peak_idx, trough_idx = compute_max_drawdown(traj)
    return {
        "pnl_curve": pnl,
        "final_pnl": int(pnl[-1]) if len(pnl) else 0,
        "sharpe": compute_sharpe(
            traj, dt_seconds=dt_seconds, trading_seconds_per_year=trading_seconds_per_year
        ),
        "max_drawdown": mdd,
        "max_drawdown_peak_idx": peak_idx,
        "max_drawdown_trough_idx": trough_idx,
        "inventory": compute_inventory_distribution(traj),
        "n_fills": len(traj.fills),
    }
