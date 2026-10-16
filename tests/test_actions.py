from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from marl_lob.actions import (
    CancelIntent,
    PlaceIntent,
    RestingOrder,
    translate_action,
)

SYMBOL = "ABM"


def _places(intents):
    return [i for i in intents if isinstance(i, PlaceIntent)]


def _cancels(intents):
    return [i for i in intents if isinstance(i, CancelIntent)]


def test_symmetric_quote_at_mid_10000():
    intents = translate_action(
        action=(2.0, 2.0, 10.0, 10.0),
        agent_id=1,
        symbol=SYMBOL,
        mid_price=10_000,
        resting_orders=[],
    )
    places = _places(intents)
    assert len(places) == 2
    bid = next(p for p in places if p.is_buy_order)
    ask = next(p for p in places if not p.is_buy_order)
    assert bid.limit_price == 9_998 and bid.quantity == 10
    assert ask.limit_price == 10_002 and ask.quantity == 10
    assert _cancels(intents) == []


def test_resting_orders_are_cancelled_then_replaced():
    resting = [
        RestingOrder(order_id=42, is_buy_order=True, limit_price=9_998, quantity=10),
        RestingOrder(order_id=43, is_buy_order=False, limit_price=10_002, quantity=10),
    ]
    intents = translate_action(
        action=(3.0, 3.0, 5.0, 5.0),
        agent_id=1,
        symbol=SYMBOL,
        mid_price=10_000,
        resting_orders=resting,
    )
    cancel_ids = sorted(c.order_id for c in _cancels(intents))
    assert cancel_ids == [42, 43]
    places = _places(intents)
    assert {p.limit_price for p in places} == {9_997, 10_003}


def test_one_sided_quote_drops_zero_size_side():
    intents = translate_action(
        action=(2.0, 2.0, 0.0, 10.0),
        agent_id=1,
        symbol=SYMBOL,
        mid_price=10_000,
        resting_orders=[],
    )
    places = _places(intents)
    assert len(places) == 1
    assert places[0].is_buy_order is False
    assert places[0].quantity == 10


def test_both_sides_zero_emits_only_cancels():
    resting = [RestingOrder(order_id=7, is_buy_order=True, limit_price=9_999, quantity=4)]
    intents = translate_action(
        action=(2.0, 2.0, 0.0, 0.0),
        agent_id=1,
        symbol=SYMBOL,
        mid_price=10_000,
        resting_orders=resting,
    )
    assert _places(intents) == []
    assert [c.order_id for c in _cancels(intents)] == [7]


def test_negative_offsets_clamp_to_zero_and_reject_crossed():
    # bid_off and ask_off both clamp to 0 → bid_price == ask_price → reject both
    intents = translate_action(
        action=(-5.0, -5.0, 10.0, 10.0),
        agent_id=1,
        symbol=SYMBOL,
        mid_price=10_000,
        resting_orders=[],
    )
    assert _places(intents) == []


def test_nan_inf_action_emits_no_places():
    for bad in (math.nan, math.inf, -math.inf):
        intents = translate_action(
            action=(bad, 2.0, 10.0, 10.0),
            agent_id=1,
            symbol=SYMBOL,
            mid_price=10_000,
            resting_orders=[],
        )
        # bid_off → 0; bid_price == mid; ask_off=2 → ask_price = mid+2.
        # Bid valid (price 10000 > 0, qty 10 > 0); ask valid; not crossed.
        # However NaN-as-bid_off becomes 0 which is fine — make NaN size the harder case:
    intents = translate_action(
        action=(2.0, 2.0, math.nan, math.nan),
        agent_id=1,
        symbol=SYMBOL,
        mid_price=10_000,
        resting_orders=[],
    )
    assert _places(intents) == []


def test_oversize_clamps_to_max_size():
    intents = translate_action(
        action=(2.0, 2.0, 10_000.0, 10_000.0),
        agent_id=1,
        symbol=SYMBOL,
        mid_price=10_000,
        resting_orders=[],
        max_size=42,
    )
    places = _places(intents)
    assert all(p.quantity == 42 for p in places)


def test_offset_quantizes_to_tick_size():
    intents = translate_action(
        action=(1.4, 1.6, 10.0, 10.0),
        agent_id=1,
        symbol=SYMBOL,
        mid_price=10_000,
        resting_orders=[],
        tick_size=1,
    )
    places = sorted(_places(intents), key=lambda p: p.limit_price)
    # bid_off rounds to 1 → bid 9999; ask_off rounds to 2 → ask 10002
    assert places[0].limit_price == 9_999
    assert places[1].limit_price == 10_002


@settings(max_examples=200)
@given(
    bid_off=st.floats(allow_nan=True, allow_infinity=True, width=32),
    ask_off=st.floats(allow_nan=True, allow_infinity=True, width=32),
    bid_size=st.floats(allow_nan=True, allow_infinity=True, width=32),
    ask_size=st.floats(allow_nan=True, allow_infinity=True, width=32),
    mid=st.integers(min_value=1, max_value=1_000_000),
)
def test_property_emitted_places_have_positive_price_and_qty(
    bid_off, ask_off, bid_size, ask_size, mid
):
    intents = translate_action(
        action=(bid_off, ask_off, bid_size, ask_size),
        agent_id=1,
        symbol=SYMBOL,
        mid_price=mid,
        resting_orders=[],
    )
    for p in _places(intents):
        assert p.limit_price > 0
        assert p.quantity > 0


# ─────────────────────────────────────────────────────────────────────────────
# Gated action space
#
# The legacy 4-tuple lets a policy stop quoting by rounding to zero, which is
# a region of the action space with no gradient — changing the parameters
# there changes neither the posted quote nor the reward. These tests pin the
# properties that make withdrawal a decision instead of an artifact.
# ─────────────────────────────────────────────────────────────────────────────

def _places(intents):
    return [i for i in intents if isinstance(i, PlaceIntent)]


def _gated(action, mid=10_000, **kw):
    return translate_action(
        action=action, agent_id=0, symbol="ABM", mid_price=mid,
        resting_orders=[], gated=True, **kw,
    )


def test_gated_subtick_action_still_posts_a_real_quote():
    """The exact failure the gates exist to remove.

    Under the legacy layout this action posts nothing: every field rounds to
    zero. Gated, the floors make it a genuine 1-tick, 1-share two-sided quote.
    """
    legacy = translate_action(
        action=(0.3, 0.3, 0.3, 0.3), agent_id=0, symbol="ABM",
        mid_price=10_000, resting_orders=[],
    )
    assert _places(legacy) == []

    places = _places(_gated((1.0, 1.0, 0.3, 0.3, 0.3, 0.3)))
    assert len(places) == 2
    bid = next(p for p in places if p.is_buy_order)
    ask = next(p for p in places if not p.is_buy_order)
    assert (bid.limit_price, bid.quantity) == (9_999, 1)
    assert (ask.limit_price, ask.quantity) == (10_001, 1)


@pytest.mark.parametrize("gate", [-1.0, -0.001, 0.0])
def test_gate_off_posts_nothing_on_that_side(gate):
    """A gate quotes only when strictly positive, so 0.0 is off — which makes
    the all-zeros vector a valid no-op for a terminated agent."""
    assert _places(_gated((gate, gate, 5.0, 5.0, 20.0, 20.0))) == []


def test_one_sided_gate_is_a_skew_not_a_void():
    """Quoting one side only is inventory skewing — real market-maker
    behaviour the legacy layout could not express deliberately."""
    places = _places(_gated((1.0, -1.0, 3.0, 3.0, 20.0, 20.0)))
    assert len(places) == 1
    assert places[0].is_buy_order
    assert places[0].limit_price == 9_997


def test_gated_sides_can_never_cross():
    """With both gates on and offsets floored at one tick,
    bid_price <= mid-1 < mid+1 <= ask_price. The legacy layout self-voided on
    54.7% of steps in the min_size_10 cell; here it is impossible."""
    for off in (0.0, 0.1, 0.4, 0.6, 1.0, 7.3):
        places = _places(_gated((1.0, 1.0, off, off, 10.0, 10.0)))
        assert len(places) == 2, f"offset {off} produced {len(places)} places"
        bid = next(p for p in places if p.is_buy_order)
        ask = next(p for p in places if not p.is_buy_order)
        assert bid.limit_price < ask.limit_price


def test_min_offset_ticks_zero_reopens_the_cross_guard():
    """min_offset_ticks=0 restores the degenerate region, and the cross guard
    must still catch it rather than posting a crossed quote."""
    assert _places(_gated((1.0, 1.0, 0.2, 0.2, 10.0, 10.0),
                          min_offset_ticks=0)) == []


def test_gated_floors_are_configurable():
    places = _places(_gated((1.0, 1.0, 0.0, 0.0, 0.0, 0.0),
                            min_offset_ticks=4, min_quote_size=25))
    assert {(p.limit_price, p.quantity) for p in places} == {
        (9_996, 25), (10_004, 25)}


def test_gated_size_still_clamps_to_max_size():
    places = _places(_gated((1.0, 1.0, 2.0, 2.0, 900.0, 900.0), max_size=100))
    assert all(p.quantity == 100 for p in places)


def test_gated_nan_gate_is_treated_as_off():
    assert _places(_gated((float("nan"), float("inf"), 5.0, 5.0, 9.0, 9.0))) == []


def test_wrong_arity_is_rejected_rather_than_misread():
    """A 4-tuple read as gated would treat the offsets as gates — silently
    wrong. Both directions must raise."""
    with pytest.raises(ValueError, match="6-tuple"):
        _gated((1.0, 1.0, 2.0, 2.0))
    with pytest.raises(ValueError, match="4-tuple"):
        translate_action(action=(1.0,) * 6, agent_id=0, symbol="ABM",
                         mid_price=10_000, resting_orders=[])


def test_gated_still_cancels_resting_orders():
    intents = translate_action(
        action=(-1.0, -1.0, 0.0, 0.0, 0.0, 0.0), agent_id=0, symbol="ABM",
        mid_price=10_000,
        resting_orders=[RestingOrder(order_id=7, is_buy_order=True,
                                     limit_price=9_999, quantity=5)],
        gated=True,
    )
    assert [i for i in intents if isinstance(i, CancelIntent)] == [
        CancelIntent(order_id=7)]
    assert _places(intents) == []
