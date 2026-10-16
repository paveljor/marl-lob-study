"""Action translator: continuous action vector -> ABIDES order/cancel intents.

Pure Python; no ABIDES import. The thin TradingAgent adapter in
``agent_adapter`` consumes the intents and dispatches them through ABIDES's
``placeLimitOrder`` / ``cancelOrder`` helpers.

Two action-vector layouts
-------------------------
``gated=False`` (legacy, 4-tuple) is kept so that trajectories and checkpoints
produced before the gated space landed can still be replayed and analysed. New
work should use ``gated=True``.

**Gated (6-tuple, preferred):**
``(bid_gate, ask_gate, bid_offset, ask_offset, bid_size, ask_size)``.

A side quotes when its gate is ``> 0``, and posts nothing otherwise. When a
side is on, its offset is floored at ``min_offset_ticks`` and its size at
``min_quote_size``, so a quote that reaches the book is always a *real* quote.

The point of the gates is that withdrawal has to be a **decision** rather than
a rounding artifact. In the legacy layout the action space floor is 0 on every
field, sub-tick offsets and sub-unit sizes both round to zero, and a policy
initialised near zero therefore starts in a region where no parameter change
alters the posted quote - and so no parameter change alters the reward. That
region has zero gradient and the optimiser cannot feel its way out of it;
measured on the 2026-09 grid, the policy posted a genuine two-sided market on
about 2% of steps. Gating separates "do not quote" (one dimension, with a
gradient) from "quote here, this big" (the rest), so withdrawal stays available
- it is the behaviour Glosten-Milgrom predicts - while ceasing to be the
default the policy is initialised into.

A second property falls out for free: with both gates on and both offsets at
least one tick, ``bid_price <= mid - 1 < mid + 1 <= ask_price``, so the two
sides can no longer cross. The legacy layout self-voided on 54.7% of steps in
the ``min_size_10`` cell; under gating that is structurally impossible.

**Legacy (4-tuple):**
``(bid_offset, ask_offset, bid_size, ask_size)``:

* ``bid_offset``, ``ask_offset`` — continuous floats in **cents**, clipped to
  ``>= 0``, then quantized to ``tick_size`` cents via ``round()``.
* ``bid_size``, ``ask_size`` — continuous floats in ``[0, max_size]``, then
  ``int(round(...))``.
* Quote prices: ``bid_price = mid_price - bid_offset_ticks``,
  ``ask_price = mid_price + ask_offset_ticks``.
* If ``bid_price >= ask_price`` post-rounding, both sides are zeroed (no place).
* Any side with non-positive price or non-positive size is dropped.
* NaN / inf in any field is treated as 0.

Common to both layouts: any side with non-positive price or size is dropped,
NaN/inf is treated as 0, and in the legacy layout both sides are voided if the
rounded bid would cross the rounded ask.

Two orthogonal knobs on top of the layouts
------------------------------------------
``offset_span`` rescales the offset dimensions - symmetrically, over
[-1, 1] - so that the policy's whole action range maps onto distinct ticks
instead of ~99% of it collapsing onto one tick, and so that a policy
initialised at mean ~0 starts mid-range with gradient in both directions.
``mid_rounding`` chooses how the quoting reference is derived from
the touch, which decides whether a symmetric offset is actually symmetric
about the true mid. Both default to the legacy behaviour: unset span and
``floor``, reproducing every earlier run bit-for-bit.

One resolution path
-------------------
``resolve_quote`` is the only place the arithmetic lives.
``translate_action`` adds intent construction on top of it, and
``quote_analysis.effective_quotes`` replays saved actions through the same
function. Reimplementing either mapping elsewhere would let the live
behaviour and every reported behavioural metric drift apart without any test
noticing - a defect class this project has already been bitten by.

Reconciliation policy: naive cancel-all-then-replace each step. Simple and
fast in simulator time; the smarter "only cancel if price/size differs" is
left as future work.

Pavel (Module C) must import ``TICK_SIZE_CENTS`` from this module rather
than redefining it — keeps action-vector semantics in one place.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

TICK_SIZE_CENTS = 1

# Floors applied to a side that a gate has switched ON. One tick and one share
# are the smallest quantities the exchange can represent, so these are the
# weakest floors that still guarantee a gated-on side reaches the book.
DEFAULT_MIN_OFFSET_TICKS = 1
DEFAULT_MIN_QUOTE_SIZE = 1

# Offset rescale. ``None`` (or 0) keeps the legacy mapping, in which the raw
# offset is read directly as cents and then floored: with tick_size=1 and
# min_offset_ticks=1 that is ``max(round(raw), 1)``, so every raw offset in
# [0, 1.5) posts exactly one tick. Measured on the 2026-09 grid the policy
# emitted raw > 1.5 on 0.06-1.4% of steps, so on ~99% of steps its offset
# choice moved no price and so changed no reward - a second zero-gradient
# region, this time inside the gated layout.
#
# A span rescales the **symmetric** [-1, 1] action range onto
# ``min_offset_ticks + [0, span]`` ticks, so each 2/span of the range is worth
# exactly one tick and the whole range is live. Symmetric, not [0, 1],
# because SB3's Gaussian policy initialises its mean at ~0 in action-space
# units: on a [0, 1] box that puts the fresh policy *at the floor* with half
# its mass clipped into the same zero-gradient region the rescale exists to
# remove. Measured at initialisation over 20k sampled observations, [0, 1]
# left 51.8% of steps posting one tick (mean 5.67 ticks) against 17.5% at the
# floor and 17.3% at the ceiling under [-1, 1] (mean 8.47). It is also the
# reason the gates already span [-1, 1].
DEFAULT_OFFSET_SPAN: int | None = None

# How the quoting reference is derived from the touch. The background book's
# spread is exactly 1 cent on ~56-57% of snapshots, so the true mid is a
# half-integer that often, and no single rounded ``mid`` can be symmetric.
#
# ``floor``   legacy: ``mid = (bid + ask) // 2``, which lands on the bid, so a
#             symmetric offset puts the ask one tick CLOSER to the true mid
#             than the bid. Measured skew: fills-per-posted-order
#             (sell/ask - buy/bid) differs by +0.105 to +0.136 between odd and
#             even spreads (notes/fill_skew.md).
# ``ceil``    the deliberate mirror: ``mid = (bid + ask + 1) // 2``. This does
#             NOT fix the asymmetry, it reverses its direction - a causal
#             manipulation check, not a repair.
# ``outward`` the fix: work from the exact doubled mid and round each side
#             away from it, so both sides sit the same distance from the true
#             mid whatever the spread's parity.
MID_ROUNDING_MODES = ("floor", "ceil", "outward")
DEFAULT_MID_ROUNDING = "floor"


@dataclass(frozen=True)
class PlaceIntent:
    symbol: str
    quantity: int
    is_buy_order: bool
    limit_price: int          # cents
    tag: str | None = None


@dataclass(frozen=True)
class CancelIntent:
    order_id: int


@dataclass(frozen=True)
class RestingOrder:
    order_id: int
    is_buy_order: bool
    limit_price: int
    quantity: int


Intent = PlaceIntent | CancelIntent


def _safe_float(x: float) -> float:
    return x if math.isfinite(x) else 0.0


def _quantize_offset(raw: float, tick_size: int) -> int:
    raw = max(_safe_float(raw), 0.0)
    return int(round(raw / tick_size)) * tick_size


def _quantize_size(raw: float, max_size: int) -> int:
    raw = max(_safe_float(raw), 0.0)
    raw = min(raw, max_size)
    return int(round(raw))


def resolve_offset_cents(
    off_raw: float,
    *,
    tick_size: int = TICK_SIZE_CENTS,
    min_offset_ticks: int = DEFAULT_MIN_OFFSET_TICKS,
    offset_span: int | None = DEFAULT_OFFSET_SPAN,
) -> int:
    """Offset in cents for a side that is switched ON.

    Two mappings, selected by ``offset_span``:

    * ``offset_span`` falsy (``None`` or ``0``) - **legacy**. The raw value is
      read as cents, rounded to the tick, then floored at
      ``min_offset_ticks``. Bit-identical to the pre-2026-09-05 behaviour.
    * ``offset_span = span`` - **rescaled**. The raw value is a position in
      the symmetric range ``[-1, 1]``, clamped at both ends, and
      ``ticks = min_offset_ticks + round((raw + 1) / 2 * span)``. Every 2/span
      of the range is one tick, ``raw = -1`` is the floor
      ``min_offset_ticks``, ``raw = +1`` is the ceiling
      ``min_offset_ticks + span``, and ``raw = 0`` - where a fresh policy
      starts - is the midpoint ``min_offset_ticks + round(span / 2)``.

    Rounding is Python's ``round`` in both branches (banker's rounding at
    exact halves), so the two mappings tie-break the same way. A NaN or
    infinite raw is scrubbed to 0.0 like everywhere else, which in the
    rescaled mapping is mid-range rather than the floor.
    """
    if offset_span:
        raw = min(max(_safe_float(off_raw), -1.0), 1.0)
        ticks = min_offset_ticks + int(round((raw + 1.0) / 2.0 * offset_span))
        return ticks * tick_size
    return max(_quantize_offset(off_raw, tick_size), min_offset_ticks * tick_size)


def _floor_half(x: int) -> int:
    """``floor(x / 2)`` in integers, correct for negative x."""
    return x // 2


def _ceil_half(x: int) -> int:
    """``ceil(x / 2)`` in integers, correct for negative x."""
    return -((-x) // 2)


def quote_prices(
    doubled_mid: int,
    bid_off_cents: int,
    ask_off_cents: int,
    *,
    mid_rounding: str = DEFAULT_MID_ROUNDING,
) -> tuple[int, int]:
    """``(bid_price, ask_price)`` in cents from the **doubled** mid.

    ``doubled_mid`` is ``best_bid + best_ask``, always an integer, so the
    half-cent that ``(bid + ask) / 2`` loses on an odd spread survives into
    this function and every mode below is exact integer arithmetic.

    * ``floor``   - ``mid = doubled_mid // 2``, then ``mid -/+ off``.
    * ``ceil``    - ``mid = (doubled_mid + 1) // 2``, then ``mid -/+ off``.
    * ``outward`` - each side rounded away from the exact mid:
      ``bid = floor((doubled_mid - 2*off) / 2)``,
      ``ask = ceil((doubled_mid + 2*off) / 2)``.

    When the spread is even ``doubled_mid`` is even, ``floor`` and ``ceil``
    of it coincide, and all three modes return the same pair.
    """
    if mid_rounding == "floor":
        mid = _floor_half(doubled_mid)
        return mid - bid_off_cents, mid + ask_off_cents
    if mid_rounding == "ceil":
        mid = _ceil_half(doubled_mid)
        return mid - bid_off_cents, mid + ask_off_cents
    if mid_rounding == "outward":
        return (
            _floor_half(doubled_mid - 2 * bid_off_cents),
            _ceil_half(doubled_mid + 2 * ask_off_cents),
        )
    raise ValueError(
        f"mid_rounding must be one of {MID_ROUNDING_MODES}, got {mid_rounding!r}"
    )


@dataclass(frozen=True)
class ResolvedQuote:
    """What one action vector actually puts on the book.

    Offsets and quantities are zero on a side that posts nothing, so
    ``bid_qty > 0`` is the test for "this side reached the exchange".
    ``crossed`` records the legacy failure mode - both sides voided because
    the bid would have crossed the ask - which is otherwise unrecoverable from
    the zeroed result.

    Prices are ``None`` when the caller had no mid to resolve them against -
    an offline replay of a saved action array, where offsets and sizes are
    still exact but the price they were posted at is not recoverable.
    """
    bid_off_cents: int
    ask_off_cents: int
    bid_qty: int
    ask_qty: int
    bid_price: int | None
    ask_price: int | None
    crossed: bool


def _gated_side(
    gate_raw: float,
    off_raw: float,
    size_raw: float,
    *,
    max_size: int,
    tick_size: int,
    min_offset_ticks: int,
    min_quote_size: int,
    offset_span: int | None = DEFAULT_OFFSET_SPAN,
) -> tuple[int, int]:
    """Resolve one side of a gated action into ``(offset_cents, quantity)``.

    A side is off unless its gate is strictly positive. SB3's Gaussian policy
    initialises with mean ~0, so a threshold at 0 puts a fresh policy at the
    decision boundary - it quotes on roughly half of steps at initialisation
    and therefore sees reward from both choices immediately, which is exactly
    what the legacy layout denied it.

    A side that is on is floored, so it cannot round its way back to silence.
    """
    if not (_safe_float(gate_raw) > 0.0):
        return 0, 0
    off = resolve_offset_cents(
        off_raw, tick_size=tick_size, min_offset_ticks=min_offset_ticks,
        offset_span=offset_span)
    qty = min(max(_quantize_size(size_raw, max_size), min_quote_size), max_size)
    return off, qty


def resolve_quote(
    action: tuple[float, ...],
    *,
    doubled_mid: int | None = None,
    gated: bool = False,
    max_size: int = 100,
    tick_size: int = TICK_SIZE_CENTS,
    min_offset_ticks: int = DEFAULT_MIN_OFFSET_TICKS,
    min_quote_size: int = DEFAULT_MIN_QUOTE_SIZE,
    offset_span: int | None = DEFAULT_OFFSET_SPAN,
    mid_rounding: str = DEFAULT_MID_ROUNDING,
) -> ResolvedQuote:
    """The single quote-resolution path.

    Both the live quoting path (``translate_action``, which only adds intent
    construction on top) and the offline reconstruction
    (``quote_analysis.effective_quotes``, which replays saved action arrays)
    go through here. That is deliberate: the reconstruction is the basis of
    every behavioural metric this project reports, and an action semantics
    change that reached only one of the two would make every one of those
    metrics silently wrong.

    ``doubled_mid=None`` is the replay case: offsets and quantities are
    exactly as resolved, prices come back None, and the two mid-dependent
    voids degrade as follows. The ``price > 0`` void is skipped, because
    without a mid there is no price to test - on a real book the mid is ~1e5
    cents and the void only ever fired on a synthetic one. The cross test is
    evaluated on the offsets, ``-bid_off >= ask_off``, which is the exact
    price test for ``floor`` and ``ceil`` at any parity and for all three
    modes at even parity; on an odd spread ``outward`` separates the sides by
    one further cent, so this is conservative there by that one cent, and
    only at all when ``min_offset_ticks`` is 0.
    """
    expected = 6 if gated else 4
    if len(action) != expected:
        raise ValueError(
            f"gated={gated} expects a {expected}-tuple action, "
            f"got length {len(action)}"
        )
    if offset_span and not gated:
        # The legacy 4-tuple has no gate, so it has no "side is ON" state for
        # the rescale to floor at min_offset_ticks; applying the span there
        # would silently remove the legacy layout's no-quote region, which is
        # the very thing it is kept around as a control for.
        raise ValueError(
            "offset_span requires the gated action layout; "
            "pass gated=True or leave offset_span unset"
        )
    if mid_rounding not in MID_ROUNDING_MODES:
        raise ValueError(
            f"mid_rounding must be one of {MID_ROUNDING_MODES}, "
            f"got {mid_rounding!r}"
        )

    if gated:
        bid_gate, ask_gate, bid_off_raw, ask_off_raw, bid_size_raw, ask_size_raw = action
        bid_off, bid_qty = _gated_side(
            bid_gate, bid_off_raw, bid_size_raw, max_size=max_size,
            tick_size=tick_size, min_offset_ticks=min_offset_ticks,
            min_quote_size=min_quote_size, offset_span=offset_span)
        ask_off, ask_qty = _gated_side(
            ask_gate, ask_off_raw, ask_size_raw, max_size=max_size,
            tick_size=tick_size, min_offset_ticks=min_offset_ticks,
            min_quote_size=min_quote_size, offset_span=offset_span)
    else:
        bid_off_raw, ask_off_raw, bid_size_raw, ask_size_raw = action
        bid_off = _quantize_offset(bid_off_raw, tick_size)
        ask_off = _quantize_offset(ask_off_raw, tick_size)
        bid_qty = _quantize_size(bid_size_raw, max_size)
        ask_qty = _quantize_size(ask_size_raw, max_size)

    bid_price: int | None
    ask_price: int | None
    if doubled_mid is None:
        bid_price = ask_price = None
        if bid_qty <= 0:
            bid_off, bid_qty = 0, 0
        if ask_qty <= 0:
            ask_off, ask_qty = 0, 0
        crossed = bid_qty > 0 and ask_qty > 0 and (-bid_off) >= ask_off
    else:
        bid_price, ask_price = quote_prices(
            doubled_mid, bid_off, ask_off, mid_rounding=mid_rounding)
        if bid_price <= 0 or bid_qty <= 0:
            bid_off, bid_qty, bid_price = 0, 0, 0
        if ask_price <= 0 or ask_qty <= 0:
            ask_off, ask_qty, ask_price = 0, 0, 0
        # Under gating with min_offset_ticks >= 1 this can never fire, since
        # bid_price <= mid - 1 < mid + 1 <= ask_price. Kept as a guard so that
        # min_offset_ticks=0 (which reopens the legacy failure) still degrades
        # safely rather than posting a crossed quote.
        crossed = bid_qty > 0 and ask_qty > 0 and bid_price >= ask_price
    if crossed:
        bid_off, bid_qty = 0, 0
        ask_off, ask_qty = 0, 0
        if doubled_mid is not None:
            bid_price = ask_price = 0

    return ResolvedQuote(
        bid_off_cents=bid_off,
        ask_off_cents=ask_off,
        bid_qty=bid_qty,
        ask_qty=ask_qty,
        bid_price=bid_price,
        ask_price=ask_price,
        crossed=crossed,
    )


def translate_action(
    action: tuple[float, ...],
    agent_id: int,
    symbol: str,
    mid_price: int | None = None,
    resting_orders: list[RestingOrder] | None = None,
    max_size: int = 100,
    tick_size: int = TICK_SIZE_CENTS,
    *,
    gated: bool = False,
    min_offset_ticks: int = DEFAULT_MIN_OFFSET_TICKS,
    min_quote_size: int = DEFAULT_MIN_QUOTE_SIZE,
    offset_span: int | None = DEFAULT_OFFSET_SPAN,
    mid_rounding: str = DEFAULT_MID_ROUNDING,
    doubled_mid: int | None = None,
) -> list[Intent]:
    """Translate an action vector into ABIDES order/cancel intents.

    ``gated=True`` expects the 6-tuple
    ``(bid_gate, ask_gate, bid_offset, ask_offset, bid_size, ask_size)``;
    ``gated=False`` expects the legacy 4-tuple. See the module docstring for
    why the gated layout exists.

    **The quoting reference.** ``doubled_mid`` (``best_bid + best_ask``) is the
    exact form and the one callers should pass. ``mid_price`` is the legacy
    pre-rounded form; it is accepted for compatibility and is equivalent to
    ``doubled_mid = 2 * mid_price``, which is always even and therefore
    indistinguishable from an even spread. Any ``mid_rounding`` other than
    ``floor`` is a statement about odd spreads, so passing one without
    ``doubled_mid`` is rejected rather than silently degrading to ``floor``.

    The ``agent_id`` argument is accepted for forward compatibility (e.g.
    tagging orders per-agent) but is not used by the current intent shape.
    """
    if doubled_mid is None:
        if mid_price is None:
            raise TypeError("translate_action needs doubled_mid or mid_price")
        if mid_rounding != "floor":
            raise ValueError(
                f"mid_rounding={mid_rounding!r} needs the exact doubled mid; "
                "pass doubled_mid=best_bid + best_ask. A pre-rounded "
                "mid_price has already lost the half cent this mode acts on."
            )
        doubled_mid = 2 * int(mid_price)
    elif mid_price is not None and int(mid_price) != _floor_half(int(doubled_mid)):
        raise ValueError(
            f"mid_price={mid_price} is inconsistent with "
            f"doubled_mid={doubled_mid}; pass one or the other"
        )

    quote = resolve_quote(
        action,
        doubled_mid=int(doubled_mid),
        gated=gated,
        max_size=max_size,
        tick_size=tick_size,
        min_offset_ticks=min_offset_ticks,
        min_quote_size=min_quote_size,
        offset_span=offset_span,
        mid_rounding=mid_rounding,
    )

    intents: list[Intent] = [
        CancelIntent(order_id=o.order_id) for o in (resting_orders or [])
    ]

    if quote.bid_qty > 0:
        intents.append(
            PlaceIntent(
                symbol=symbol,
                quantity=quote.bid_qty,
                is_buy_order=True,
                limit_price=quote.bid_price,
            )
        )
    if quote.ask_qty > 0:
        intents.append(
            PlaceIntent(
                symbol=symbol,
                quantity=quote.ask_qty,
                is_buy_order=False,
                limit_price=quote.ask_price,
            )
        )
    return intents
