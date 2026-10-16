"""Thin TradingAgent adapter that executes ``translate_action`` intents.

The TradingAgent import is guarded so this file loads on machines without
ABIDES; the class itself only
works when ABIDES is on the path.

Intent dispatch is one method call per intent — kept deliberately small so
swapping the JPMorgan fork (or any other ABIDES variant) is cheap.
"""
from __future__ import annotations

from collections.abc import Callable

from .actions import (
    DEFAULT_MID_ROUNDING,
    DEFAULT_MIN_OFFSET_TICKS,
    DEFAULT_MIN_QUOTE_SIZE,
    DEFAULT_OFFSET_SPAN,
    CancelIntent,
    Intent,
    PlaceIntent,
    RestingOrder,
    translate_action,
)

try:  # pragma: no cover — import is environment-dependent
    from abides_markets.agents.trading_agent import TradingAgent  # type: ignore
    _ABIDES_AVAILABLE = True
except Exception:  # pragma: no cover
    TradingAgent = object  # type: ignore[misc, assignment]
    _ABIDES_AVAILABLE = False


def project_resting_orders(orders_dict: dict) -> list[RestingOrder]:
    """Convert ABIDES ``self.orders`` to ``RestingOrder`` records."""
    out: list[RestingOrder] = []
    for order_id, order in orders_dict.items():
        out.append(
            RestingOrder(
                order_id=int(order_id),
                is_buy_order=bool(order.side.is_bid()),
                limit_price=int(order.limit_price),
                quantity=int(order.quantity),
            )
        )
    return out


def doubled_mid_from_obs(obs: dict) -> int | None:
    """``best_bid + best_ask`` out of an observation dict, or None.

    The doubled mid is the exact quoting reference; ``obs["mid_price"]`` has
    already been rounded and cannot recover the half cent an odd spread
    carries. Callers that want anything other than ``mid_rounding="floor"``
    therefore have to supply one of these keys - ``translate_action`` refuses
    the combination rather than quietly flooring.
    """
    if not isinstance(obs, dict):
        return None
    if obs.get("doubled_mid") is not None:
        return int(obs["doubled_mid"])
    bid, ask = obs.get("best_bid"), obs.get("best_ask")
    if bid is not None and ask is not None:
        return int(bid) + int(ask)
    return None


def dispatch_intents(
    intents: list[Intent],
    *,
    orders_dict: dict,
    place_fn: Callable[..., None],
    cancel_fn: Callable[..., None],
) -> None:
    """Push intents through the supplied ABIDES helpers.

    Split out from ``MarlAgent.wakeup`` so it is testable without instantiating
    a real ABIDES agent — pass in mocks for ``place_fn`` / ``cancel_fn`` and a
    plain dict for ``orders_dict``.
    """
    for intent in intents:
        if isinstance(intent, CancelIntent):
            order = orders_dict.get(intent.order_id)
            if order is not None:
                cancel_fn(order)
        elif isinstance(intent, PlaceIntent):
            place_fn(
                intent.symbol,
                intent.quantity,
                intent.is_buy_order,
                intent.limit_price,
            )


class MarlAgent(TradingAgent):  # type: ignore[misc, valid-type]
    """ABIDES TradingAgent driven by a (policy, observation_extractor) pair.

    Constructed with:
      - ``policy_fn(obs) -> action_tuple`` supplies policy actions.
      - ``observation_fn(self, current_time) -> obs`` extracts observations.
      - ``symbol``, ``max_size``, ``tick_size`` — passed to ``translate_action``.

    The class body only resolves when ABIDES is importable; instantiating
    without ABIDES raises ``RuntimeError``.
    """

    def __init__(
        self,
        id: int,
        name: str,
        type: str,
        symbol: str,
        policy_fn: Callable,
        observation_fn: Callable,
        max_size: int = 100,
        tick_size: int = 1,
        gated: bool = False,
        min_offset_ticks: int = DEFAULT_MIN_OFFSET_TICKS,
        min_quote_size: int = DEFAULT_MIN_QUOTE_SIZE,
        offset_span: int | None = DEFAULT_OFFSET_SPAN,
        mid_rounding: str = DEFAULT_MID_ROUNDING,
        random_state=None,
    ) -> None:
        if not _ABIDES_AVAILABLE:
            raise RuntimeError(
                "MarlAgent requires ABIDES on the import path. "
                "Install ABIDES or run on a machine that has it."
            )
        super().__init__(id, name, type, random_state=random_state)
        self.symbol = symbol
        self.policy_fn = policy_fn
        self.observation_fn = observation_fn
        self.max_size = max_size
        self.tick_size = tick_size
        self.gated = gated
        self.min_offset_ticks = min_offset_ticks
        self.min_quote_size = min_quote_size
        self.offset_span = offset_span
        self.mid_rounding = mid_rounding

    def wakeup(self, current_time):  # pragma: no cover — exercised in ABIDES-gated test
        super().wakeup(current_time)
        obs = self.observation_fn(self, current_time)
        action = self.policy_fn(obs)
        # A bare-array observation carries no mid-price: the vector is
        # normalised and its last element is time-to-close (or, once an
        # identity one-hot is appended, an ID bit). Callers that want this
        # adapter to quote must hand back a dict with an explicit
        # "mid_price"; anything else is a caller bug, not a value to guess.
        if not isinstance(obs, dict) or "mid_price" not in obs:
            raise TypeError(
                "observation_fn must return a mapping containing 'mid_price'; "
                "the observation vector does not carry an unnormalised mid."
            )
        # Prefer the exact doubled mid when the observation carries the touch;
        # fall back to the pre-rounded mid, which translate_action accepts only
        # under mid_rounding="floor".
        doubled_mid = doubled_mid_from_obs(obs)
        intents = translate_action(
            action=tuple(action),
            agent_id=self.id,
            symbol=self.symbol,
            mid_price=None if doubled_mid is not None else int(obs["mid_price"]),
            resting_orders=project_resting_orders(self.orders),
            max_size=self.max_size,
            tick_size=self.tick_size,
            gated=self.gated,
            min_offset_ticks=self.min_offset_ticks,
            min_quote_size=self.min_quote_size,
            offset_span=self.offset_span,
            mid_rounding=self.mid_rounding,
            doubled_mid=doubled_mid,
        )
        dispatch_intents(
            intents,
            orders_dict=self.orders,
            place_fn=self.placeLimitOrder,
            cancel_fn=self.cancelOrder,
        )
