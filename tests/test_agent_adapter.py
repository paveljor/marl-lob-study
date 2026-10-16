"""Tests for the agent adapter.

The mock-based test of `dispatch_intents` runs everywhere — that's the part
I (Neil) own and need to keep correct. The ABIDES-gated smoke test is for
machines that have ABIDES installed (Pavel's box).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from marl_lob.actions import CancelIntent, PlaceIntent, RestingOrder
from marl_lob.agent_adapter import (
    _ABIDES_AVAILABLE,
    dispatch_intents,
    project_resting_orders,
)


def test_project_resting_orders_extracts_fields():
    fake_order = MagicMock()
    fake_order.side.is_bid.return_value = True
    fake_order.limit_price = 9_998
    fake_order.quantity = 5
    out = project_resting_orders({42: fake_order})
    assert out == [RestingOrder(order_id=42, is_buy_order=True, limit_price=9_998, quantity=5)]


def test_dispatch_routes_cancels_and_places():
    place_fn = MagicMock()
    cancel_fn = MagicMock()
    fake_resting = MagicMock(name="fake_order_42")
    orders = {42: fake_resting}

    intents = [
        CancelIntent(order_id=42),
        PlaceIntent(symbol="ABM", quantity=10, is_buy_order=True, limit_price=9_998),
        PlaceIntent(symbol="ABM", quantity=10, is_buy_order=False, limit_price=10_002),
    ]
    dispatch_intents(intents, orders_dict=orders, place_fn=place_fn, cancel_fn=cancel_fn)

    cancel_fn.assert_called_once_with(fake_resting)
    assert place_fn.call_count == 2
    place_fn.assert_any_call("ABM", 10, True, 9_998)
    place_fn.assert_any_call("ABM", 10, False, 10_002)


def test_dispatch_skips_cancel_when_order_id_unknown():
    """If the order_id was already filled / cancelled by the exchange, skip silently."""
    place_fn = MagicMock()
    cancel_fn = MagicMock()
    dispatch_intents(
        [CancelIntent(order_id=999)],
        orders_dict={},
        place_fn=place_fn,
        cancel_fn=cancel_fn,
    )
    cancel_fn.assert_not_called()


@pytest.mark.abides
def test_marl_agent_imports_when_abides_available():  # pragma: no cover
    if not _ABIDES_AVAILABLE:
        pytest.skip("ABIDES not on path")
    from marl_lob.agent_adapter import MarlAgent  # noqa: F401
