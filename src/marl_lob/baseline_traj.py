"""Trajectory adapter for the constant-spread baseline.

F runs natively in ABIDES (no PettingZoo), so it never emits an
``info["traj_row"]`` like the env does. This module wraps F to record the
same 6-tuple snapshots once per quote round, then exposes them as a
``Trajectory`` for D's metrics harness.

Snapshot frequency matches F's wakeup frequency (default 10s — coarser than
the env's 1s, fine for Sharpe/MaxDD). One snapshot per successful quote
round; wakeups that skip due to missing L1 data are not snapshotted.
"""
from __future__ import annotations

from typing import Any

from .baseline_market_maker import ConstantSpreadMarketMaker
from .trajectory import Trajectory


def aggregate_step_fills(orders: list) -> tuple[int, int]:
    """Compress a list of executed Orders into (signed_qty, vwap_price_cents).

    `orders` are ABIDES `Order` objects with `.side.is_bid()`, `.quantity`,
    and `.fill_price`. Buys count positive, sells negative; the VWAP weights
    by absolute quantity. Empty input returns (0, 0).

    Mixed-side steps collapse both directions into one VWAP — same imprecision
    as `marl_agents._aggregate_fills`. Acceptable at 10s wakeups; an MM almost
    never both buys and sells in the same step.
    """
    if not orders:
        return 0, 0
    signed_qty = sum(
        int(o.quantity) * (1 if o.side.is_bid() else -1) for o in orders
    )
    abs_total = sum(int(o.quantity) for o in orders)
    if abs_total == 0:
        return 0, 0
    vwap = sum(int(o.quantity) * int(o.fill_price) for o in orders) / abs_total
    return signed_qty, int(round(vwap))


class LoggingConstantSpreadMM(ConstantSpreadMarketMaker):
    """ConstantSpreadMarketMaker that records traj_row snapshots in-place."""

    def __init__(self, *args: Any, mkt_open_ns: int = 0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.mkt_open_ns = int(mkt_open_ns)
        # Each entry: (ts_s, inv, cash, mid, fill_signed_qty, fill_price_cents)
        self.snapshots: list[tuple[float, int, int, int, int, int]] = []
        # We keep our own fill log because TradingAgent.order_executed (F's
        # parent) only updates holdings — it does NOT append to
        # `executed_orders`, despite the attribute existing. The CoreBackgroundAgent
        # lineage does, but F predates that hierarchy.
        self._fills_since_last_snap: list = []

    def order_executed(self, order) -> None:  # type: ignore[override]
        super().order_executed(order)
        self._fills_since_last_snap.append(order)

    def _place_quotes(self, current_time: int) -> None:  # type: ignore[override]
        """Snapshot inv/cash/mid/fills each successful quote round, then quote.

        The parent's _place_quotes reads L1, computes mid, and posts orders.
        We snapshot *before* that so the inventory/cash reflect state going
        into the round, not after the new orders settle (orders won't fill
        instantaneously anyway, but the convention matches C's traj_row).
        """
        bid, _, ask, _ = self.get_known_bid_ask(self.symbol)
        if bid is not None and ask is not None:
            mid = int((bid + ask) // 2)
            inv = int(self.holdings.get(self.symbol, 0))
            cash = int(self.holdings.get("CASH", 0))

            fill_qty, fill_vwap = aggregate_step_fills(self._fills_since_last_snap)
            self._fills_since_last_snap = []

            ts_s = (int(current_time) - self.mkt_open_ns) / 1e9
            self.snapshots.append(
                (ts_s, inv, cash, mid, int(fill_qty), int(fill_vwap))
            )

        super()._place_quotes(current_time)

    def to_trajectory(self) -> Trajectory:
        """Build a Trajectory from the recorded snapshots."""
        return Trajectory.from_tuples(self.snapshots)
