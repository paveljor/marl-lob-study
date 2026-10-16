"""Pure-helper tests for the baseline trajectory adapter.

Same split as test_env.py: pure helpers tested in isolation, the
ABIDES-driven `_place_quotes` override is exercised via run_baseline.py
as a smoke check (not unit-tested).
"""
from __future__ import annotations

from unittest.mock import MagicMock

from marl_lob.baseline_traj import aggregate_step_fills


def _fake_order(is_buy: bool, price: int, qty: int):
    """Match real ABIDES Order: `.side.is_bid()`, `.fill_price`, `.quantity`."""
    side = MagicMock()
    side.is_bid.return_value = is_buy
    o = MagicMock()
    o.side = side
    o.fill_price = price
    o.quantity = qty
    return o


def test_aggregate_empty_returns_zero_zero():
    assert aggregate_step_fills([]) == (0, 0)


def test_aggregate_all_buys_signed_positive_vwap_exact():
    orders = [
        _fake_order(is_buy=True, price=10_000, qty=5),
        _fake_order(is_buy=True, price=10_002, qty=10),
    ]
    qty, vwap = aggregate_step_fills(orders)
    assert qty == 15
    # (5*10000 + 10*10002) / 15 = 10001.333... → rounds to 10001
    assert vwap == 10_001


def test_aggregate_all_sells_signed_negative_vwap_exact():
    orders = [
        _fake_order(is_buy=False, price=9_998, qty=7),
        _fake_order(is_buy=False, price=9_999, qty=3),
    ]
    qty, vwap = aggregate_step_fills(orders)
    assert qty == -10
    # (7*9998 + 3*9999) / 10 = 9998.3 → rounds to 9998
    assert vwap == 9_998


def test_aggregate_mixed_sides_nets_signed_qty_and_collapses_vwap():
    """Mixed-side steps: signed sum, VWAP weighted by absolute qty."""
    orders = [
        _fake_order(is_buy=True, price=10_002, qty=10),
        _fake_order(is_buy=False, price=10_000, qty=4),
    ]
    qty, vwap = aggregate_step_fills(orders)
    assert qty == 6
    # weighted by abs qty: (10*10002 + 4*10000) / 14 = 10001.43 → 10001
    assert vwap == 10_001


def test_aggregate_single_fill_vwap_equals_fill_price():
    qty, vwap = aggregate_step_fills([_fake_order(is_buy=True, price=12_345, qty=1)])
    assert (qty, vwap) == (1, 12_345)
